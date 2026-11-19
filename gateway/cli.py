"""kllm command-line interface."""
import json, os, sys, time, argparse, shutil, urllib.request
from . import common, orchestrator as orch, kaggle, media

def _age(ts):
    if not ts: return "-"
    s = int(time.time() - ts)
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

def cmd_models(args):
    models = common.load_models()
    cfg = common.load_config()
    for k, m in models.items():
        mark = "*" if k == cfg["default_model"] else " "
        kind = media.kind(m)
        what = f"{media.model_id(m):40s}" if kind == "chat" else f"[{kind}/{media.runtime(m)}] {m['hf']}"
        print(f"{mark} {k:22s} {what} ~{m.get('vram_gb','?')}GB"
              + (f"  ctx {m.get('ctx','?')}" if kind == "chat" else ""))
        print(f"    {m.get('desc','')}")

def cmd_model(args):
    if not args.key:
        print("default_model:", common.load_config()["default_model"]); return
    m = common.resolve_model(args.key)  # validate
    if media.kind(m) != "chat":
        print(f"error: '{args.key}' is a {media.kind(m)} model; the default model must be a chat model "
              f"(request {args.key} by name via /v1/{media.kind(m)}s/generations)")
        sys.exit(1)
    import re, pathlib
    p = common.CONFIG_FILE
    txt = p.read_text(encoding="utf-8")
    txt = re.sub(r"^default_model:.*$", f"default_model: {args.key}", txt, flags=re.M)
    p.write_text(txt, encoding="utf-8")
    print(f"default_model -> {args.key}")
    print("Run `kllm rotate` to serve it (brings up a fresh backend, then swaps).")

def cmd_add(args):
    """Add any HuggingFace model: a GGUF repo (llama.cpp chat) or a diffusers repo (image/video)."""
    from . import inspect_model as im
    try:
        cfg = common.load_config()
    except common.ConfigMissing:
        cfg = {}
    try:
        res = im.inspect_source(args.source, file=args.file, ctx=args.ctx, name=args.name,
                                token=cfg.get("hf_token") or None, taken=set(common.load_models()))
    except im.InspectError as e:
        print("error:", e); sys.exit(1)
    print("\n".join(im.summary(res)))
    if args.default and media.kind(res["entry"]) != "chat":
        print("error: only a chat model can be the default; add it without --default"); sys.exit(1)
    if args.dry_run:
        print(im.entry_yaml(res["key"], res["entry"]), end="")
        return
    if not res["fits"] and not args.force:
        print("error: does not fit the 2xT4 kernel (use --force to add it anyway)"); sys.exit(1)
    try:
        im.write_entry(res["key"], res["entry"], common.MODELS_FILE)
    except im.InspectError as e:
        print("error:", e); sys.exit(1)
    print(f"added '{res['key']}' to {common.MODELS_FILE}")
    if args.default:
        common.set_config_value("default_model", res["key"])
        print(f"default_model -> {res['key']}  (run `kllm rotate` to serve it)")
    print(f"start it:  kllm up --model {res['key']}")
    if res["kind"] != "chat":
        ep = f"/v1/{res['kind']}s/generations"
        print(f'then:      curl -s http://127.0.0.1:{cfg.get("relay_port", 8080)}{ep} -H "Content-Type: application/json" '
              f"-d '{{\"model\": \"{res['key']}\", \"prompt\": \"a lighthouse at dusk\"}}'")

def _has_accounts():
    try:
        return bool(common.load_config().get("accounts"))
    except Exception:
        return False


def cmd_open(args):
    """`kllm` / `kllm setup`: make sure the relay runs, then open the panel (the welcome wizard on a first run)."""
    import webbrowser
    if not common.ensure_serve(wait_seconds=45, log=print):
        print("kllm: the relay did not start (see logs/serve.log under", common.ROOT, ")"); sys.exit(1)
    url = common.relay_base() + "/admin" + ("" if _has_accounts() else "#welcome")
    print(f"kllm panel: {url}" + ("" if _has_accounts() else "   (first run: paste your Kaggle token there)"))
    if not getattr(args, "no_browser", False):
        try:
            webbrowser.open(url)
        except Exception:
            pass


def cmd_update(args):
    """`kllm update`: check PyPI for a newer kllm and upgrade with the tool that installed it."""
    from . import updates
    st = updates.status(force=True)
    if not st["latest"]:
        print("kllm: could not reach PyPI to check for updates (offline?). You have", st["current"]); sys.exit(1)
    if not st["update_available"]:
        print(f"kllm {st['current']} is the latest version."); return
    print(f"kllm {st['latest']} is available (you have {st['current']}). Release notes: {st['notes']}")
    if args.check:
        print(f"Update with: kllm update   (or: {st['command']})"); return
    code = updates.run_upgrade()
    if code == 0 and os.name != "nt":
        print("kllm: updated. Restart `kllm serve` (or the kllm service) to run the new version.")
    elif code == 0:
        print("kllm: when the new window says it finished, restart `kllm serve` (or the kllm service).")
    sys.exit(code)


def cmd_up(args):
    if not _has_accounts():
        print("kllm: no Kaggle account yet. Opening the setup page to add one...")
        return cmd_open(args)
    orch.launch(model_key=args.model, account=args.account,
                session_seconds=args.session, wait=not args.no_wait)

def cmd_scale(args):
    """Run N parallel backends across N different accounts."""
    orch.scale(args.count, model_key=args.model)

def cmd_rotate(args):
    """Zero-downtime swap: bring up a fresh backend on a free account, then drop the old one."""
    cfg = common.load_config()
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if args.account and allow and not any(str(args.account).startswith(p) for p in allow):
        print(f"Error: Account '{args.account}' is not permitted by account_allow_prefixes={allow}")
        return
    s = common.read_state()
    backends = s.get("backends", [])
    if args.account:
        old = next((b for b in backends if b["account"] == args.account), None)
    else:  # rotate the oldest backend
        old = min(backends, key=lambda b: b.get("started", 0)) if backends else None
    busy = {b["account"] for b in backends}
    idx, acct = orch.pick_next_account(cfg, s, skip=busy)
    print(f"rotating -> fresh backend on {acct} (replacing: {old['account'] if old else 'none'})")
    new = orch.launch(model_key=args.model or (old and old.get("model_key")), account=acct, wait=True)
    if new:
        if old:
            orch.stop(old["account"])
        print(f"swapped in {acct}. Old backend (if any) dropped; it self-expires on Kaggle.")
    else:
        print("rotation failed; keeping existing pool unchanged.")

def cmd_down(args):
    orch.stop(account=args.account)

def cmd_cache(args):
    """Build & cache llama.cpp for faster future launches (private per-account dataset)."""
    cfg = common.load_config()
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if args.all:
        accts = [a["name"] for a in cfg["accounts"]]
        if allow:
            accts = [a for a in accts if any(str(a).startswith(p) for p in allow)]
    else:
        if args.account and allow and not any(str(args.account).startswith(p) for p in allow):
            print(f"Error: Account '{args.account}' is not permitted by account_allow_prefixes={allow}")
            return
        accts = [args.account or None]
    for a in accts:
        orch.build_cache(account=a)

def cmd_status(args):
    s = common.read_state()
    backends = s.get("backends", [])
    pending = s.get("pending", {})
    print("=== kllm status ===")
    if backends:
        print(f"POOL  ({len(backends)} backend{'s' if len(backends)!=1 else ''})")
        for b in sorted(backends, key=lambda x: x.get("started", 0)):
            left = (b["started"] + b.get("session_seconds", 0)) - time.time()
            healthy = orch.health_check(b["url"], b["api_key"]) if b.get("url") else False
            print(f"  {b['account']:4s} {b['model_key']:20s} {('HEALTHY' if healthy else 'UNREACHABLE'):11s}"
                  f" up {_age(b.get('ready_at') or b['started']):>7s}  left ~{int(left//60)}m")
            print(f"       {b['url']}")
    else:
        print("POOL  empty  (run `kllm up` or `kllm scale N`)")
    for acct, p in pending.items():
        print(f"PENDING {acct}  model={p['model_key']}  status={p.get('status')}  ({_age(p['started'])})")
    lf = common.load_config().get("local_fallback", {})
    if lf.get("enabled"):
        print(f"LOCAL   fallback {lf['url']}  {'up' if _local_up(lf['url']) else 'down'}")

def _local_up(url):
    import urllib.request
    try:
        urllib.request.urlopen(url.rstrip('/')+"/health", timeout=3); return True
    except Exception:
        return False

def cmd_kstatus(args):
    cfg = common.load_config()
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if args.account:
        if allow and not any(str(args.account).startswith(p) for p in allow):
            print(f"Error: Account '{args.account}' is not permitted by account_allow_prefixes={allow}")
            return
        accts = [args.account]
    else:
        accts = [a["name"] for a in cfg["accounts"]]
        if allow:
            accts = [a for a in accts if any(str(a).startswith(p) for p in allow)]
    for acct in accts:
        user = orch.account_user(cfg, acct)
        ref = f"{user}/{cfg['kernel_slug']}"
        st = kaggle.kernel_status(acct, ref)
        print(f"{acct:4s} {ref:40s} {st}")

def cmd_watch(args):
    from .supervisor import supervise
    lock = common.single_instance("supervisor")
    if lock is None:
        print("another supervisor is already running (kllm serve, or kllm-supervisor.service); not starting a second one")
        return
    supervise(desired=args.size)

def cmd_relay(args):
    cfg = common.load_config()
    import uvicorn
    uvicorn.run("gateway.relay:app", host=cfg["relay_host"], port=cfg["relay_port"],
                log_level="info", access_log=False)

def cmd_ensure_serve(args):
    """Start `kllm serve` in the background if it isn't running (idempotent).

    For hooks and wrappers (opencode, scripts): run this before the first request
    so "select a kllm model and start a session" never needs a manual step."""
    ok = common.ensure_serve(wait_seconds=args.wait, log=print)
    print("relay up" if ok else "relay NOT up (see logs/serve.log)")
    sys.exit(0 if ok else 1)


def cmd_serve(args):
    """One process: the relay (HTTP) plus the supervisor loop (thread) and, if enabled, the tunnel."""
    import threading, uvicorn
    from . import relay
    cfg = common.load_config()
    lock = None
    if not args.no_supervisor:
        lock = common.single_instance("supervisor")      # held for the life of the process
        if lock is None:
            print("warning: another supervisor is already running; serving the relay only", flush=True)
        else:
            from .supervisor import supervise, _log
            def sup():
                relay.INPROC["supervisor"] = True
                try:
                    supervise()
                except BaseException as e:               # never take the relay down with it
                    _log("supervisor stopped:", repr(e))
                finally:
                    relay.INPROC["supervisor"] = False
            threading.Thread(target=sup, name="supervisor", daemon=True).start()
    if args.remote or cfg.get("remote_enabled"):
        from . import tunnel
        tunnel.CTL.start(log=lambda *a: print(*a, flush=True))
    print(f"kllm serve: relay on http://{cfg['relay_host']}:{cfg['relay_port']}  root={common.ROOT}", flush=True)
    print(f"kllm serve: open the panel at http://127.0.0.1:{cfg['relay_port']}/admin"
          + ("" if cfg.get("accounts") else "  (first run: add your Kaggle token there)"), flush=True)
    uvicorn.run(relay.app, host=cfg["relay_host"], port=cfg["relay_port"], log_level="info", access_log=False,
                timeout_graceful_shutdown=5)   # do not hold a restart hostage to multi-minute LLM streams (clients retry)

def cmd_remote(args):
    """Public cloudflared tunnel to the gateway (downloads cloudflared on first use)."""
    from . import tunnel
    try:
        tunnel.run_forever(port=args.port)
    except KeyboardInterrupt:
        pass

def cmd_init(args):
    from . import setup
    try:
        setup.run_init(accounts=args.account, github_token=args.github_token,
                       gist_id=args.gist_id, gist_user=args.gist_user, no_gist=args.no_gist,
                       api_key=args.api_key, force=args.force, interactive=not args.non_interactive,
                       network=not args.no_network)
    except (FileExistsError, ValueError) as e:
        print("error:", e); sys.exit(1)

def cmd_install_service(args):
    from . import service
    service.install(start=not args.no_start, uninstall=args.uninstall, print_only=args.print)

def cmd_litellm_config(args):
    from . import litellm_cfg
    out, names = litellm_cfg.generate()
    print("wrote", out, "with models:", names)
    print("run it with:  pip install 'kllm[litellm]'  &&  litellm --config", out, "--port", common.load_config().get("litellm_port", 4000))

def cmd_test(args):
    cfg = common.load_config()
    import urllib.request
    base = f"http://{cfg['relay_host']}:{cfg['relay_port']}"
    body = json.dumps({"model": "kllm", "messages":[{"role":"user","content":
            args.prompt}], "max_tokens": args.max_tokens, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
            headers={"Content-Type":"application/json"})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=args.timeout))
    except Exception as e:
        print("request failed:", e); return
    msg = d["choices"][0]["message"]["content"]
    tm = d.get("timings", {})
    print(msg)
    print(f"\n--- {time.time()-t:.1f}s"
          + (f"  gen {tm.get('predicted_per_second',0):.1f} t/s" if tm else ""))

def cmd_autoscale(args):
    """Turn the resource manager on/off, or show its settings and recent decisions."""
    from . import scheduler
    if args.mode in ("on", "off"):
        common.set_section_value("autoscale", "enabled", "true" if args.mode == "on" else "false")
    if args.max is not None:
        common.set_section_value("autoscale", "max_backends", args.max)
    s = scheduler.settings(common.load_config())
    print("autoscale:", "ON" if s["enabled"] else "OFF (manual: kllm up / kllm down)")
    for k in ("min_backends", "max_backends", "sessions_per_backend", "idle_stop_minutes",
              "queue_timeout_seconds", "max_wait_seconds"):
        print(f"  {k:24s} {s[k]}")
    log = (common.read_state().get("scheduler") or {}).get("log") or []
    if log:
        print("recent decisions:")
        for e in log[-12:]:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(e['ts']))} {e['action']:6s} {e['target']:4s}"
                  f" {e.get('model') or '':28s} {e['why']}")

def cmd_sessions(args):
    """What clients are doing right now, as the relay sees it."""
    import urllib.request
    cfg = common.load_config()
    try:
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{cfg['relay_port']}/__relay/demand", timeout=5))
    except Exception as e:
        print("relay not reachable:", e); return
    print("demand per model:", json.dumps(d.get("models") or {}))
    for label, b in (d.get("backends") or {}).items():
        print(f"  {label:12s} {b['model']:28s} in-flight {b['inflight']}  sessions {b['sessions']}  idle {b['idle_s']//60} min")
    for s in d.get("sessions") or []:
        print(f"  session -> {s['backend'] or '(waiting)':12s} {s['model']:28s} {s['requests']} requests, idle {s['idle_s']} s")

def cmd_account(args):
    """kllm account add <KAGGLE_API_TOKEN> | kllm account list"""
    from . import accounts
    cfg = common.load_config()
    if args.action == "list":
        for a in cfg["accounts"]:
            print(f"  {a['name']:5s} {a['user']}" + ("  (disabled)" if a.get("disabled") else ""))
        return
    try:
        r = accounts.add_account(args.token, args.name, cfg)
    except accounts.AccountError as e:
        print("error:", e); raise SystemExit(1)
    print(f"added {r['name']} = {r['user']}  (credentials {r['home']}/.kaggle, mode 600; wrapper {r['wrapper']})")
    print("Note: Kaggle gives GPUs only to phone-verified accounts; an unverified one is skipped automatically.")

def cmd_doctor(args):
    """End-to-end check of everything a harness needs: relay, pool, auth, context, tool calls,
    streaming, LiteLLM. Prints one line per check and what to do if it fails."""
    import urllib.request
    cfg = common.load_config()
    relay = f"http://{cfg['relay_host']}:{cfg['relay_port']}"
    lite = f"http://127.0.0.1:{cfg.get('litellm_port', 4000)}"
    bad = 0
    def check(name, ok, detail="", fix=""):
        nonlocal bad
        bad += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name:28s} {detail}" + (f"\n       -> {fix}" if not ok and fix else ""))
    def post(base, path, body, key=None, timeout=180):
        h = {"Content-Type": "application/json", "x-kllm-probe": "doctor"}
        if key: h["Authorization"] = f"Bearer {key}"
        r = urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), h), timeout=timeout)
        return json.load(r), r.headers.get("x-kllm-backend")
    print("=== kllm doctor ===")
    from . import rendezvous
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    skipped = []
    for a in cfg.get("accounts") or []:
        if allow and not any(str(a.get("name")).startswith(p) for p in allow):
            skipped.append(a.get("name"))            # kllm never launches on these: their credentials do not matter
            continue
        try:
            kind, where = kaggle.resolve_credentials(a)
            check(f"account {a['name']}", True, f"{kind}: {where}")
        except kaggle.NoCredentials as e:
            check(f"account {a.get('name')}", False, "", str(e))
    if skipped:
        check("accounts not allowed", True, f"{len(skipped)} skipped (account_allow_prefixes: {', '.join(allow)})")
    gtok = rendezvous.token(cfg)
    gid = str(cfg.get("gist_id") or "")
    if rendezvous.mode(cfg) == "ntfy":
        sec = cfg.get("rendezvous_secret")
        ok_sec = rendezvous._real(sec)
        reach = False
        if ok_sec:
            try:
                urllib.request.urlopen(urllib.request.Request(rendezvous.ntfy_server(cfg) + "/v1/health",
                                                              headers={"User-Agent": "kllm"}), timeout=8).read()
                reach = True
            except Exception:
                reach = False
        check("rendezvous ntfy", ok_sec and reach,
              f"sealed channel via {rendezvous.ntfy_server(cfg)}" if ok_sec else "rendezvous_secret not set",
              "kllm init --force   (generates rendezvous_secret)" if not ok_sec else
              f"cannot reach {rendezvous.ntfy_server(cfg)}: check your connection/proxy, or set ntfy_server")
    elif not gid or "YOUR_" in gid:
        check("rendezvous gist", False, "gist_id not set", "kllm init --github-token <token>   (creates the private gist)")
    elif gtok:
        who = rendezvous.whoami(cfg, gtok)
        check("rendezvous gist", bool(who) and rendezvous._get_gist(cfg) is not None,
              f"token for {who or '?'} reads gist {gid[:8]}...", "token needs the 'gist' scope and must own the gist")
    else:
        check("rendezvous gist", shutil.which("gh") is not None, "no token configured; using the gh CLI",
              "set gist_token in config.yaml (or GH_TOKEN)")
    from . import auth
    check("gateway_api_key", auth.real_key(cfg) is not None,
          "set (needed only for non-local callers)" if auth.real_key(cfg) else "not set: remote/LAN callers are refused",
          "kllm init --force  (generates one) or set gateway_api_key in config.yaml")
    try:
        urllib.request.urlopen(relay + "/__relay/health", timeout=5); check("relay", True, relay)
    except Exception as e:
        check("relay", False, str(e), "kllm serve   (or: kllm install-service)"); return
    s = common.read_state()
    if not s["backends"]:
        check("pool", False, "no backend", "kllm up   (or turn on always-on in the panel)"); return
    for b in s["backends"]:
        left = int((b["started"] + b.get("session_seconds", 0) - time.time()) // 60)
        check(f"backend {b['account']}", orch.health_check(b["url"], b["api_key"]),
              f"{b['model_key']}  ctx {b.get('ctx')}  {left} min left", "kllm rotate")
    for key in sorted({b["model_key"] for b in s["backends"]}):
        ctx = min(b.get("ctx") or 0 for b in s["backends"] if b["model_key"] == key)
        reserve = 13000 + (32768 if ctx >= 200000 else 16384)     # claude-kllm's autocompact buffer
        check(f"context {key}"[:28], ctx - reserve > 40000,
              f"{ctx//1024}K window, ~{(ctx - reserve)//1000}K usable in Claude Code before autocompact")
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    try:
        t = time.time()
        d, be = post(relay, "/v1/chat/completions", {"model": "kllm", "max_tokens": 200, "tools": tools,
                     "messages": [{"role": "user", "content": "Read /tmp/x.txt"}]})
        tc = (d["choices"][0]["message"].get("tool_calls") or [{}])[0].get("function", {})
        check("tool call (OpenAI)", tc.get("name") == "read_file", f"{be}  {time.time()-t:.1f}s  {tc}")
        d, be = post(relay, "/v1/messages", {"model": "kllm", "max_tokens": 200,
                     "tools": [{"name": "Read", "description": "Read a file", "input_schema": tools[0]["function"]["parameters"]}],
                     "messages": [{"role": "user", "content": "Read /tmp/x.txt"}]})
        check("tool call (Anthropic)", d.get("stop_reason") == "tool_use", f"{be}  stop_reason={d.get('stop_reason')}")
        d, be = post(relay, "/v1/chat/completions", {"model": "kllm", "max_tokens": 300,
                     "messages": [{"role": "user", "content": "Write a haiku about GPUs."}]})
        tm = d.get("timings") or {}
        check("generation speed", tm.get("predicted_per_second", 0) > 20, f"{tm.get('predicted_per_second', 0):.0f} t/s")
    except Exception as e:
        check("model requests", False, str(e)[:200])
    if cfg.get("litellm_enabled"):
        try:
            d, _ = post(lite, "/v1/chat/completions", {"model": "kllm", "max_tokens": 8,
                        "messages": [{"role": "user", "content": "Say ok"}]}, key=cfg.get("gateway_api_key"))
            check("LiteLLM (optional)", bool(d.get("choices")), lite)
        except Exception as e:
            check("LiteLLM (optional)", False, str(e)[:120], "systemctl --user restart kllm-litellm")
    print("all good" if not bad else f"{bad} problem(s)")

def cmd_spawn(args):
    """Spawn a cross-harness worker subagent powered by Kaggle GPU."""
    import asyncio
    import urllib.request
    import urllib.error
    from . import spawner

    scope = []
    if args.scope:
        for s in args.scope:
            scope.extend([x.strip() for x in s.split(",") if x.strip()])

    relay_active = common.relay_up()
    relay_url = common.relay_base()
    res = None

    if relay_active:
        # If the persistent relay daemon is active, dispatch to it via HTTP
        # This guarantees async background subagents continue running after the CLI exits!
        payload = {
            "harness": args.harness,
            "task_instructions": args.task,
            "permitted_files": scope if scope else None,
            "verification_command": args.verify,
            "timeout_seconds": args.timeout,
            "cwd": args.cwd,
            "model": args.model,
            "wait": not args.async_run,
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{relay_url}/admin/api/subagents/spawn",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            http_timeout = (args.timeout + 60) if not args.async_run else 10
            with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                if resp.status == 200:
                    res = json.loads(resp.read().decode("utf-8"))
        except Exception:
            # Fall back to local execution if relay endpoint is not yet mounted
            res = None

    if res is None:
        print(f"Spawning subagent (harness={args.harness})...")
        res = asyncio.run(spawner.FLEET.spawn(
            harness=args.harness,
            task_instructions=args.task,
            permitted_files=scope if scope else None,
            verification_command=args.verify,
            timeout_seconds=args.timeout,
            cwd=args.cwd,
            model=args.model,
            wait=not args.async_run,
        ))

    if args.async_run:
        print(f"Subagent spawned in background: {res.get('subagent_id')}")
        print(f"Check status: kllm subagents --id {res.get('subagent_id')}")
        return

    status = res.get("status")
    print(f"\n[{status}] Subagent {res.get('subagent_id')} finished in {res.get('execution_time_seconds', 0)}s")
    print(f"  Kaggle tokens consumed: {res.get('kaggle_tokens_consumed', 0)} (Cloud tokens: 0)")
    if res.get("files_modified"):
        print(f"  Files modified: {', '.join(res.get('files_modified', []))}")
    if res.get("scope_violations"):
        print(f"  [!] Scope violations: {', '.join(res.get('scope_violations', []))}")
    if res.get("verification_output"):
        print(f"  Verification ({res.get('verification_status')}):\n    {res.get('verification_output')[:300]}")
    if res.get("git_diff"):
        print(f"\n--- Unified Git Diff ---\n{res.get('git_diff')[:2000]}")
        if len(res.get('git_diff', '')) > 2000:
            print("... [diff truncated]")
    if res.get("subagent_summary"):
        print(f"\n--- Summary ---\n{res.get('subagent_summary')[:500]}")
    if res.get("error"):
        print(f"\nError: {res.get('error')}", file=sys.stderr)


def cmd_subagents(args):
    """List or inspect subagents in the fleet."""
    import urllib.request
    from . import spawner

    relay_active = common.relay_up()
    relay_url = common.relay_base()

    if args.cancel:
        cancelled = False
        if relay_active:
            try:
                req = urllib.request.Request(
                    f"{relay_url}/admin/api/subagents/{args.cancel}/cancel",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        cancelled = True
            except Exception:
                pass
        if not cancelled:
            cancelled = spawner.FLEET.cancel(args.cancel)
        print(f"Cancelled {args.cancel}: {'success' if cancelled else 'failed / not found'}")
        return

    if args.id:
        info = None
        if relay_active:
            try:
                with urllib.request.urlopen(f"{relay_url}/admin/api/subagents/{args.id}", timeout=5) as resp:
                    if resp.status == 200:
                        info = json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass
        if not info:
            info = spawner.FLEET.get(args.id)
        if not info:
            print(f"Subagent '{args.id}' not found.", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(info, indent=2))
        return

    items = None
    if relay_active:
        try:
            url = f"{relay_url}/admin/api/subagents"
            if args.status:
                url += f"?status={args.status}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    items = data.get("subagents")
        except Exception:
            pass

    if items is None:
        items = spawner.FLEET.list_all(status=args.status)

    if not items:
        print("No subagents registered.")
        return
    print(f"{'ID':28s} {'HARNESS':12s} {'STATUS':14s} {'TIME':8s} {'MODIFIED':s}")
    for item in items:
        sid = item.get("subagent_id", "")
        harness = item.get("harness", "")
        status = item.get("status", "")
        sec = f"{item.get('execution_time_seconds', 0):.1f}s"
        mods = ", ".join(item.get("files_modified", [])) or "-"
        print(f"{sid:28s} {harness:12s} {status:14s} {sec:8s} {mods}")


def cmd_mcp(args):
    """Run Model Context Protocol server or show config."""
    from . import mcp_server
    if args.config:
        print(json.dumps(mcp_server.get_mcp_config_snippet(), indent=2))
        return
    mcp_server.main()


# Commands that only read. Inside a kllm Agents session (KLLM_AGENT_SESSION set) the Access page decides how far
# `kllm` reaches: off refuses everything, read allows these, full allows all. Every command is logged on the session.
READ_ONLY = {"status", "kstatus", "models", "sessions", "doctor", "test", "litellm-config"}


def _is_read_only(args):
    if args.cmd in READ_ONLY:
        return True
    if args.cmd == "model" and not getattr(args, "key", None):
        return True
    if args.cmd == "autoscale" and not getattr(args, "mode", None) and not getattr(args, "max", None):
        return True
    if args.cmd == "account" and getattr(args, "action", None) == "list":
        return True
    if args.cmd == "subagents" and not getattr(args, "cancel", None):
        return True
    if args.cmd == "mcp" and getattr(args, "config", False):
        return True
    return False


def agent_access_ok(args, env=None):
    env = os.environ if env is None else env
    sid = env.get("KLLM_AGENT_SESSION")
    if not sid:
        return True
    level = env.get("KLLM_ACCESS", "read")
    allowed = level == "full" or (level == "read" and _is_read_only(args))
    try:
        relay = env.get("KLLM_RELAY") or "http://127.0.0.1:8080"
        req = urllib.request.Request(f"{relay}/admin/api/agents/sessions/{sid}/kllm_action",
                                     data=json.dumps({"worker": env.get("KLLM_AGENT_WORKER"),
                                                      "action": "kllm " + " ".join(sys.argv[1:] or [args.cmd]),
                                                      "allowed": allowed, "via": "cli"}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).read()
    except Exception:
        pass
    if not allowed:
        print(f"kllm: this agent session has '{level}' access to kllm commands, so `kllm {args.cmd}` is refused. "
              "Change it on the Access page in kllm Agents.", file=sys.stderr)
    return allowed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kllm", description="Kaggle GPU as a local LLM gateway")
    sub = ap.add_subparsers(dest="cmd")       # no command: open the panel (setup wizard on a first run)
    s = sub.add_parser("update", help="check PyPI for a newer kllm and upgrade this install")
    s.add_argument("--check", action="store_true", help="only report whether an update is available")
    s.set_defaults(fn=cmd_update)
    s = sub.add_parser("setup", help="start kllm and open the setup/control panel in your browser (same as `kllm`)")
    s.add_argument("--no-browser", action="store_true", help="only print the panel URL")
    s.set_defaults(fn=cmd_open)

    s = sub.add_parser("add", help="add a model from HuggingFace (GGUF chat model, or diffusers image/video model)")
    s.add_argument("source", help="repo id (org/name) or a huggingface.co URL; org/name:Q5_K_M picks a quant")
    s.add_argument("--name", help="key in models.yaml (default: derived from the repo name)")
    s.add_argument("--file", help="GGUF file or quant to use (default: best quant that fits 2xT4)")
    s.add_argument("--ctx", type=int, help="context length (default: from GGUF metadata, bounded by VRAM)")
    s.add_argument("--default", action="store_true", help="make it the default model (chat models only)")
    s.add_argument("--dry-run", action="store_true", help="inspect and print the entry, do not write models.yaml")
    s.add_argument("--force", action="store_true", help="add even if it does not look like it fits 2xT4")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("up", help="launch a backend on the next account")
    s.add_argument("--model"); s.add_argument("--account"); s.add_argument("--session", type=int)
    s.add_argument("--no-wait", action="store_true"); s.set_defaults(fn=cmd_up)

    s = sub.add_parser("scale", help="run N parallel backends across N accounts")
    s.add_argument("count", type=int); s.add_argument("--model"); s.set_defaults(fn=cmd_scale)

    s = sub.add_parser("rotate", help="bring up a fresh backend then swap (zero downtime)")
    s.add_argument("--model"); s.add_argument("--account"); s.set_defaults(fn=cmd_rotate)

    s = sub.add_parser("down", help="stop backend(s): all, or one --account (frees Kaggle GPU quota)")
    s.add_argument("--account"); s.set_defaults(fn=cmd_down)
    s = sub.add_parser("status", help="show active/pending/local state"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("watch", help="supervise: health, failover, pre-expiry rotation")
    s.add_argument("--size", type=int, help="desired pool size to maintain (default: current)")
    s.set_defaults(fn=cmd_watch)
    s = sub.add_parser("relay", help="run the local stable endpoint (reverse proxy)"); s.set_defaults(fn=cmd_relay)
    s = sub.add_parser("serve", help="run relay + supervisor (+ tunnel if enabled) in ONE process (default install)")
    s.add_argument("--no-supervisor", action="store_true", help="relay only")
    s.add_argument("--remote", action="store_true", help="also open the public tunnel")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("ensure-serve", help="start `kllm serve` in the background if down (idempotent; for hooks/wrappers)")
    s.add_argument("--wait", type=int, default=30, help="seconds to wait for the relay to answer")
    s.set_defaults(fn=cmd_ensure_serve)
    s = sub.add_parser("init", help="setup wizard: config.yaml, accounts, gist, gateway key, prerequisite checks")
    s.add_argument("--non-interactive", action="store_true", help="no prompts; use the flags below")
    s.add_argument("--account", action="append", default=[], metavar="[NAME=]KAGGLE_JSON",
                   help="path to an account's kaggle.json (repeatable; names default to kg1, kg2...)")
    s.add_argument("--github-token", help="GitHub token with only the gist scope (creates the private gist)")
    s.add_argument("--gist-id"); s.add_argument("--gist-user")
    s.add_argument("--no-gist", action="store_true", help="do not create a gist")
    s.add_argument("--api-key", help="gateway_api_key (default: random)")
    s.add_argument("--force", action="store_true", help="overwrite an existing config.yaml (a .bak is kept)")
    s.add_argument("--no-network", action="store_true", help="skip the network reachability checks")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("install-service", help="install the background service (systemd user unit / launchd agent)")
    s.add_argument("--no-start", action="store_true", help="only write the file")
    s.add_argument("--uninstall", action="store_true"); s.add_argument("--print", action="store_true", help="print, do not write")
    s.set_defaults(fn=cmd_install_service)
    s = sub.add_parser("remote", help="public cloudflared tunnel to the gateway (optional; downloads cloudflared)")
    s.add_argument("port", nargs="?", type=int, help="local port to expose (default: relay, or LiteLLM if enabled)")
    s.set_defaults(fn=cmd_remote)
    s = sub.add_parser("litellm-config", help="(optional LiteLLM) regenerate litellm/config.yaml from models.yaml")
    s.set_defaults(fn=cmd_litellm_config)
    s = sub.add_parser("models", help="list the model registry"); s.set_defaults(fn=cmd_models)

    s = sub.add_parser("model", help="show/set default model"); s.add_argument("key", nargs="?")
    s.set_defaults(fn=cmd_model)

    s = sub.add_parser("kstatus", help="kaggle kernel status per account"); s.add_argument("account", nargs="?")
    s.set_defaults(fn=cmd_kstatus)

    s = sub.add_parser("cache", help="pre-build llama.cpp into a per-account dataset (faster launches)")
    s.add_argument("account", nargs="?"); s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_cache)

    s = sub.add_parser("autoscale", help="resource manager: on | off | (show); --max N backends")
    s.add_argument("mode", nargs="?", choices=["on", "off"]); s.add_argument("--max", type=int)
    s.set_defaults(fn=cmd_autoscale)
    s = sub.add_parser("sessions", help="live demand: sessions, load per backend")
    s.set_defaults(fn=cmd_sessions)

    s = sub.add_parser("account", help="Kaggle accounts: add <KAGGLE_API_TOKEN> [--name kgN] | list")
    s.add_argument("action", choices=["add", "list"]); s.add_argument("token", nargs="?"); s.add_argument("--name")
    s.set_defaults(fn=cmd_account)

    s = sub.add_parser("doctor", help="end-to-end check: relay, pool, context, tool calls, speed, LiteLLM")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("test", help="send a prompt through the relay")
    s.add_argument("prompt", nargs="?", default="Reply with exactly: kllm online.")
    s.add_argument("--max-tokens", type=int, default=64); s.add_argument("--timeout", type=int, default=300)
    s.set_defaults(fn=cmd_test)

    s = sub.add_parser("spawn", help="cross-harness subagent: dispatch task to Claude Code / OpenCode / Shell on Kaggle GPU")
    s.add_argument("task", help="task instructions for the worker subagent")
    s.add_argument("--harness", default="claude-code", choices=["claude-code", "claude-kllm", "claude", "opencode", "shell", "bash", "agy"],
                   help="worker harness environment (default: claude-code)")
    s.add_argument("--scope", action="append", default=[], help="permitted files/dirs the subagent may edit (repeatable or comma-separated)")
    s.add_argument("--verify", help="verification command to run after edits (e.g. 'pytest tests/...')")
    s.add_argument("--timeout", type=int, default=600, help="max execution timeout in seconds (default: 600)")
    s.add_argument("--cwd", help="working directory (default: workspace root)")
    s.add_argument("--model", help="model key override")
    s.add_argument("--async", dest="async_run", action="store_true", help="run asynchronously in background and return subagent ID")
    s.set_defaults(fn=cmd_spawn)

    s = sub.add_parser("subagents", help="fleet manager: list, inspect, or cancel worker subagents")
    s.add_argument("--id", help="inspect a specific subagent deliverable")
    s.add_argument("--status", choices=["PENDING", "RUNNING", "COMPLETED", "NEEDS_REVIEW", "FAILED", "TIMEOUT", "CANCELLED"],
                   help="filter by status")
    s.add_argument("--cancel", help="cancel a running subagent by ID")
    s.set_defaults(fn=cmd_subagents)

    s = sub.add_parser("mcp", help="run Model Context Protocol server (stdio bridge for Antigravity / Claude Desktop)")
    s.add_argument("--config", action="store_true", help="print JSON snippet for IDE mcp_config.json")
    s.set_defaults(fn=cmd_mcp)

    args = ap.parse_args(argv)
    if args.cmd is None:
        args.cmd, args.fn, args.no_browser = "setup", cmd_open, False
    if not agent_access_ok(args):
        sys.exit(3)
    # zero-config: the first command that needs a config creates it from the Kaggle credentials already on this
    # machine (~/.kaggle/kaggle.json, ~/.kaggle/access_token or KAGGLE_* env). `init`/`add` handle config themselves.
    if args.cmd not in ("init", "add", "mcp") and (not common.CONFIG_FILE.exists()
                                                    or args.cmd in ("serve", "up", "ensure-serve", "setup")):
        from . import setup as _setup
        if not _setup.ensure_config(common.ROOT, log=lambda *a: print(*a, file=sys.stderr), allow_empty=True):
            sys.exit(2)
        common.CONFIG_FILE = common.ROOT / "config.yaml"
    # update notice: cached result only (no network wait); a stale cache is refreshed in the background
    if args.cmd not in ("mcp", "update"):
        try:
            from . import updates
            msg = updates.notice()
            if msg:
                print(msg, file=sys.stderr)
            updates.refresh_in_background()
        except Exception:
            pass
    try:
        args.fn(args)
    except common.ConfigMissing as e:
        print("error:", e, file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
