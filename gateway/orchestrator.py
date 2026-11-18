"""Launch, rotate, and supervise the Kaggle-hosted llama-server backends."""
import base64
import json, time, secrets, shutil, datetime, urllib.request
from . import common, kaggle, rendezvous, media

KROOT = common.data_path("kernel")
BUILD_ROOT = common.ROOT / "state" / "builds"
BUILD_ROOT.mkdir(parents=True, exist_ok=True)

def _now(): return int(time.time())
def _ts(t): return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")

def account_user(cfg, name):
    for a in cfg["accounts"]:
        if a["name"] == name:
            return a["user"]
    raise KeyError(name)

def pick_next_account(cfg, state, skip=()):
    """Round-robin over the account pool, skipping any in `skip`."""
    from .scheduler import disabled_accounts
    accts = [a["name"] for a in cfg["accounts"]]
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if allow:
        accts = [a for a in accts if any(str(a).startswith(p) for p in allow)]
    if not accts:
        raise RuntimeError(f"No accounts configured matching account_allow_prefixes={allow}")
    start = (state.get("last_account_index", -1) + 1) % len(accts)
    off = disabled_accounts(cfg)
    for skipped in (set(skip) | off, set(skip)):          # disabled accounts only as a last resort
        for i in range(len(accts)):
            idx = (start + i) % len(accts)
            if accts[idx] not in skipped:
                return idx, accts[idx]
    return start, accts[start]

def render_kernel(cfg, account, model, api_key, session_seconds, dataset_sources=None,
                  build_only=False, stage_idx=None, stages=None):
    """Write a per-account build dir containing server_kernel.py + kernel-metadata.json."""
    user = account_user(cfg, account)
    # Per-account decorrelation overrides (inert unless set on the account entry):
    # distinct slug/title and per-account shared identifiers so no two accounts'
    # kernels carry the same correlation strings.
    _acct = next((a for a in cfg["accounts"] if a.get("name") == account), {})
    _ov = lambda f, d: _acct.get(f, d)
    slug = _ov("kernel_slug", cfg.get("kernel_slug", "kllm-server"))
    title = _ov("kernel_title", "kllm-server")
    ref = f"{user}/{slug}"
    bdir = BUILD_ROOT / account
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir(parents=True)

    kcfg = {
        "account": account,
        "model_hf": model["hf"],
        "model_quant": model.get("quant"),
        "model_file": model.get("file"),
        "chat_template": (common.data_path(model["template"]).read_text(encoding="utf-8") if model.get("template") else None),
        "model_ctx": model.get("ctx", 32768),
        "model_args": model.get("args", []),
        # real multimodal: llama-server --mmproj-url (model entry fields: mmproj, optional mmproj_repo)
        "mmproj_url": (f"https://huggingface.co/{model.get('mmproj_repo') or model['hf']}/resolve/main/{model['mmproj']}"
                       if model.get("mmproj") else None),
        "image_max_tokens": model.get("image_max_tokens", 1024),
        "extra_server_args": cfg.get("extra_server_args", []),
        "api_key": _ov("api_key", api_key),
        # rendezvous: ntfy (default) needs only the shared secret; the gist token goes into the kernel ONLY in
        # gist mode, so a GitHub token never reaches Kaggle unless the user chose the gist backend.
        "rendezvous": rendezvous.mode(cfg),
        "rendezvous_secret": _ov("rendezvous_secret", cfg.get("rendezvous_secret", "")),
        "ntfy_server": rendezvous.ntfy_server(cfg),
        "gist_id": _ov("gist_id", cfg.get("gist_id", "")) if rendezvous.mode(cfg) == "gist" else "",
        "gist_token": _ov("gist_token", cfg.get("gist_token", "")) if rendezvous.mode(cfg) == "gist" else "",
        "gist_user": _ov("gist_user", cfg.get("gist_user", "")) if rendezvous.mode(cfg) == "gist" else "",
        "session_seconds": session_seconds,
        "llamacpp_ref": model.get("llamacpp_ref") or cfg.get("llamacpp_ref", "master"),
        # prebuilt llama.cpp release tag; a model entry may pin its own (A/B a newer build before rolling it out)
        "llamacpp_build": model.get("llamacpp_build") or cfg.get("llamacpp_build", "b11425"),
        "llamacpp_repo": model.get("llamacpp_repo") or cfg.get("llamacpp_repo", "https://github.com/ggml-org/llama.cpp"),
        "prefer_cached_binary": model.get("prefer_cached_binary", False),
        "port": 8000,
        "build_only": build_only,
        # pluggable runtime (llama.cpp text models by default; diffusers for image/video; swarm for distributed MoE)
        "runtime": media.runtime(model),
        "kind": media.kind(model),
        "model_key": model.get("key"),
        "opts": dict(model.get("opts") or {}, **({"pipeline": model["pipeline"]} if model.get("pipeline") else {})),
        "hf_token": _ov("hf_token", cfg.get("hf_token", "")) if media.runtime(model) == "diffusers" else "",
        "hub_url": _ov("hub_url", (cfg.get("swarm") or {}).get("hub_url")),
        "stage_idx": int(stage_idx if stage_idx is not None else 0),
        "stages": int(stages if stages is not None else model.get("stages", 6)),
        "total_layers": model.get("total_layers"),
        "hidden_dim": model.get("hidden_dim"),
        "num_heads": model.get("num_heads"),
        "num_kv_heads": model.get("num_kv_heads"),
        "head_dim": model.get("head_dim", 128),
        "num_experts": model.get("num_experts"),
        "vocab_size": model.get("vocab_size"),
        "layers_per_stage": model.get("layers_per_stage"),
        "dataset": model.get("dataset"),
        "draft_model": model.get("draft_model"),
        "draft_file": model.get("draft_file"),
        "draft_hf": model.get("draft_hf"),
        "draft_quant": model.get("draft_quant"),
        "draft_ngld": model.get("draft_ngld"),
        "draft_args": model.get("draft_args"),
        "spec_k": model.get("spec_k") or (cfg.get("swarm") or {}).get("spec_k", 6),
        "tunnel_name": _ov("tunnel_name", cfg.get("tunnel_name")),
        "tunnel_creds_file": _ov("tunnel_creds_file", cfg.get("tunnel_credentials_file")),
        "tunnel_hostname": _ov("tunnel_hostname", cfg.get("tunnel_hostname")),
    }
    if kcfg["runtime"] == "comfyui":
        # the ComfyUI video runtime + its custom nodes travel inside the kernel config (Kaggle takes one file)
        # base64: the config is embedded in a raw \'\'\' literal, so source text must not appear verbatim
        kcfg["_comfy_video_b64"] = base64.b64encode((KROOT / "comfy_video.py").read_bytes()).decode()
        kcfg["_kllm_nodes_b64"] = base64.b64encode((KROOT / "comfy_kllm_nodes.py").read_bytes()).decode()
        kcfg["hf_token"] = _ov("hf_token", cfg.get("hf_token", ""))
    tmpl = (KROOT / "server_kernel.tmpl.py").read_text(encoding="utf-8")
    code = tmpl.replace("__KLLM_CONFIG_JSON__", json.dumps(kcfg))
    (bdir / "server_kernel.py").write_text(code, encoding="utf-8")

    meta = (KROOT / "kernel-metadata.tmpl.json").read_text(encoding="utf-8")
    ds = ", ".join(json.dumps(d) for d in (dataset_sources or []))
    meta = (meta.replace("__USER__", user).replace("__SLUG__", slug)
                .replace("__DATASET_SOURCES__", ds))
    if title != "kllm-server":
        meta = meta.replace('"title": "kllm-server"', f'"title": "{title}"')
    (bdir / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    return bdir, ref

def launch_swarm_cluster(model_key=None, session_seconds=None, wait=True, log=print, origin="manual", accounts=None):
    """Launch an N-node distributed MoE pipeline cluster across pooled Kaggle accounts."""
    cfg = common.load_config()
    state = common.read_state()
    model = common.resolve_model(model_key or cfg["default_model"])
    total_stages = int(model.get("stages", 6))

    all_accts = [a["name"] for a in cfg["accounts"]]
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if allow:
        all_accts = [a for a in all_accts if any(str(a).startswith(p) for p in allow)]
    busy = {b["account"] for b in state.get("backends", [])} | set(state.get("pending", {}).keys())
    free = [a for a in all_accts if a not in busy]

    if accounts:
        if allow:
            disallowed = [a for a in accounts if not any(str(a).startswith(p) for p in allow)]
            if disallowed:
                log(f"Error: Swarm accounts {disallowed} not permitted by account_allow_prefixes={allow}")
                return None
        chosen = list(accounts)[:total_stages]
    elif len(free) >= total_stages:
        chosen = free[:total_stages]
    else:
        chosen = free + [a for a in all_accts if a not in free][:max(0, total_stages - len(free))]

    if len(chosen) < total_stages:
        log(f"Error: need {total_stages} accounts for swarm model '{model['key']}', but found {len(chosen)}")
        return None

    log(f"Launching swarm model '{model['key']}' across {total_stages} accounts: {chosen}")
    pending_list = []
    min_gap = int((cfg.get("swarm") or {}).get("push_min_gap", 5))

    for idx, acct in enumerate(chosen):
        p = launch(model_key=model["key"], account=acct, session_seconds=session_seconds,
                   wait=False, log=log, origin=origin, stage_idx=idx, stages=total_stages)
        if p:
            pending_list.append(p)
        if idx < total_stages - 1 and min_gap > 0:
            time.sleep(min_gap)

    if not wait:
        log(f"Swarm cluster launched async across {chosen}; track via `kllm status`.")
        return pending_list

    deadline = time.time() + 2700
    while time.time() < deadline:
        s = common.read_state()
        active = {b["account"]: b for b in s.get("backends", []) if b.get("model_key") == model["key"]}
        if all(a in active for a in chosen):
            log(f"Swarm cluster '{model['key']}' is fully active across all {total_stages} stages!")
            return active.get(chosen[0]) or True
        time.sleep(10)

    log(f"Timed out waiting for swarm cluster '{model['key']}' to become ready")
    return None


def launch(model_key=None, account=None, session_seconds=None, wait=True, log=print, origin="manual",
           stage_idx=None, stages=None):
    """Render + push a server kernel, wait for its tunnel URL, record it as active."""
    cfg = common.load_config()
    state = common.read_state()
    model = common.resolve_model(model_key or cfg["default_model"])
    bad = media.problems(model)
    if bad:
        log(f"model '{model['key']}' is not launchable: " + "; ".join(bad))
        return None
    session_seconds = session_seconds or cfg["session_seconds"]

    if media.runtime(model) == "swarm" and stage_idx is None:
        return launch_swarm_cluster(model_key=model["key"], session_seconds=session_seconds,
                                    wait=wait, log=log, origin=origin)

    reserved = False
    if account is None:
        # Pick AND reserve atomically, never an account that already runs a backend or has a launch pending: pushing to
        # it replaces that kernel (a live backend and its sessions die). Concurrent `kllm up` calls also raced onto the
        # same account and overwrote each other's push (seen 2026-11-07: two launches -> kg9 v1, v2).
        with common.mutate_state() as s:
            busy = {b.get("account") for b in s.get("backends", [])} | set((s.get("pending") or {}).keys())
            idx, account = pick_next_account(cfg, s, skip=busy)
            if account in busy:
                log("no free Kaggle account: every account already runs a backend or has a launch pending")
                return None
            s.setdefault("pending", {})[account] = {"account": account, "model_key": model["key"], "origin": origin,
                                                    "started": _now(), "since": _now() - 5, "reserving": True}   # since: ignore events of earlier kernels on it
            s["last_account_index"] = idx
        reserved = True
    else:
        allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
        if allow and not any(str(account).startswith(p) for p in allow):
            log(f"Error: Account '{account}' is not permitted by account_allow_prefixes={allow}")
            return None
        idx = next((i for i, a in enumerate(cfg["accounts"]) if a["name"] == account), -1)

    api_key = "kllm-" + secrets.token_hex(16)
    # Attach model dataset if specified (e.g. pre-split GGUF or quantized safetensors)
    ds_sources = []
    if model.get("dataset"):
        if isinstance(model["dataset"], list):
            ds_sources.extend(model["dataset"])
        else:
            ds_sources.append(model["dataset"])
    if model.get("datasets"):
        for d in model["datasets"]:
            if d not in ds_sources:
                ds_sources.append(d)
    cache_ds = (state.get("build_cache") or {}).get(account)
    if cache_ds and cache_ds not in ds_sources:
        ds_sources.append(cache_ds)
    # Attach tunnel credentials dataset if configured (for named cloudflared tunnels)
    tunnel_ds = cfg.get("tunnel_dataset")
    if tunnel_ds and tunnel_ds not in ds_sources:
        ds_sources.append(tunnel_ds)

    def release():
        if reserved:
            with common.mutate_state() as s:
                if (s.get("pending") or {}).get(account, {}).get("reserving"):
                    s["pending"].pop(account, None)
    try:
        bdir, ref = render_kernel(cfg, account, model, api_key, session_seconds, ds_sources,
                                  stage_idx=stage_idx, stages=stages)
        since = _now() - 5
        log(f"[{account}] pushing {ref}  model={model['key']} ({media.model_id(model)})"
            + (f"  +cache {cache_ds}" if cache_ds else "")
            + (f"  stage={stage_idx}/{stages}" if stage_idx is not None else ""))
        ok, msg = kaggle.push_kernel(account, str(bdir))
    except BaseException:
        release()                                   # never leave a reservation blocking the account
        raise
    if not ok:
        quota_msg = "weekly GPU quota" in msg
        if quota_msg:
            # Kaggle's own count beats kllm's estimate — suppress this account until the weekly reset
            until = common.mark_quota_exhausted(account)
            log(f"[{account}] Kaggle reports the weekly GPU quota is used up; "
                f"suppressing this account until {time.strftime('%Y-%m-%d %H:%M', time.localtime(until))}")
        common.set_pending(account, {
            "account": account, "user": account_user(cfg, account),
            "model_key": model["key"], "origin": origin,
            "started": _now(), "since": _now(),
            "status": "failed", "failed": True,
            "note": msg[:240], "kind": "image" if model.get("kind") == "image" else ("video" if model.get("kind") == "video" else "chat"),
        })
        log(f"[{account}] push failed: {msg}")
        return None
    log(f"[{account}] pushed: {msg}")
    try:
        rendezvous.publish_control(cfg, account, "run")
    except Exception:
        pass

    pending = {
        "account": account, "user": account_user(cfg, account), "kernel_ref": ref,
        "model": media.model_id(model), "model_key": model["key"],
        "ctx": model.get("ctx"), "api_key": api_key, "url": None,
        "started": _now(), "since": since, "session_seconds": session_seconds,
        "status": "launching", "origin": origin,
        "stage_idx": stage_idx,
    }
    common.set_pending(account, pending)
    common.update_state(last_account_index=idx)

    if not wait:
        log(f"[{account}] launched async; use `kllm watch` or `kllm status` to track.")
        return pending

    return await_ready(cfg, pending, log=log)

def await_ready(cfg, pending, log=print, overall_timeout=2700):
    """Block until the kernel reports a ready URL in the gist (or fails), then promote."""
    account = pending["account"]
    deadline = time.time() + overall_timeout
    last_stage = None
    while time.time() < deadline:
        for ev in rendezvous.latest_events(cfg):
            if ev.get("account") not in (account, None) or ev.get("ts", 0) < pending.get("since", 0):
                continue
            e = ev.get("event")
            if e in ("boot", "binary", "building", "downloading", "serving") and e != last_stage:
                log(f"[{account}] {e}: {ev.get('note') or ev.get('stage') or ev.get('gpu') or ''}")
                last_stage = e
            if e in ("error", "crash"):
                log(f"[{account}] kernel failed: {ev.get('stage') or ''} {ev.get('err') or ''}")
                return None
        if try_promote_pending(cfg, common.read_state().get("pending", {}).get(account, pending), log=log):
            for b in common.read_state().get("backends", []):
                if b["account"] == account:
                    return b
            return True
        time.sleep(10)
    log(f"[{account}] timed out waiting for ready. Check `kllm kstatus {account}`.")
    return None

def try_promote_pending(cfg, pending, log=print, events=None):
    """Non-blocking: read this pending's current status from the gist; if it reports a ready
    URL and the backend answers a real completion, promote it into the pool. Returns True if
    promoted. Lets a fire-and-forget launch (CLI --no-wait or the web UI) auto-register."""
    if not pending:
        return False
    account = pending["account"]
    evs = events if events is not None else rendezvous.latest_events(cfg)
    ready = None
    since = pending.get("since", 0)
    for e in evs:
        # ignore the previous session on this account: its last 'alive' still carries an old
        # url (and that kernel has a different api key)
        if e.get("ts", 0) < since:
            continue
        # a serving kernel reports 'ready' then 'alive' heartbeats — both carry the url
        if e.get("event") in ("ready", "alive") and e.get("url") and e.get("account") in (account, None):
            ready = e
        if e.get("event") in ("crash", "error") and e.get("account") in (account, None):
            log(f"[{account}] kernel reported {e.get('event')}: {e.get('stage') or e.get('err','')}")
    if not ready and (pending.get("stage_idx") == 0 or pending.get("model_key") == "qwen3-235b-swarm"):
        hub_st_url = (cfg.get("swarm") or {}).get("hub_url") or ""     # experimental swarm: your own hub only
        hub_st_url = hub_st_url.replace("ws://", "http://").replace("wss://", "https://")
        if hub_st_url and not hub_st_url.endswith("/status"):
            hub_st_url = f"{hub_st_url.rstrip('/')}/status"
        try:
            if not hub_st_url:
                raise ValueError("no swarm hub configured")
            with urllib.request.urlopen(hub_st_url, timeout=3) as r:
                hub_st = json.loads(r.read())
                s0 = hub_st.get("stage0_url")
                if s0:
                    ready = {"url": s0, "account": account, "event": "ready"}
        except Exception:
            pass
    if not ready:
        return False
    url = ready["url"]
    if not health_check(url, pending["api_key"]):
        return False
    backend = dict(pending)
    backend.update(url=url, status="active", ready_at=_now())
    common.upsert_backend(backend)
    log(f"[{account}] REGISTERED {url} ({backend['model_key']})")
    return True

def publish_stop(account):
    """Ask a running kernel (by account) to shut down so it stops using GPU quota."""
    try:
        return rendezvous.publish_control(common.load_config(), account, "stop")
    except Exception:
        return False

def health_check(url, api_key, timeout=25):
    try:
        # /v1/models, not /health: llama-server serves /health without auth, so a wrong key or
        # another session's kernel would pass. This proves the backend accepts OUR key.
        req = urllib.request.Request(url.rstrip("/") + "/v1/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False

def build_cache(account=None, log=print, model_key=None):
    """Compile llama.cpp once on an account and snapshot the binaries into that account's
    own PRIVATE Kaggle dataset, then register it for that account so future launches there
    skip the ~15-min build. Run per account (datasets are private & per-owner)."""
    cfg = common.load_config()
    s = common.read_state()
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if account is not None:
        if allow and not any(str(account).startswith(p) for p in allow):
            log(f"Error: Account '{account}' is not permitted by account_allow_prefixes={allow}")
            return None
    else:
        _, account = pick_next_account(cfg, s)
    user = account_user(cfg, account)
    mkey = model_key or cfg["default_model"]
    model = common.resolve_model(mkey)
    api_key = "build-" + secrets.token_hex(4)
    bdir, ref = render_kernel(cfg, account, model, api_key, 3600, build_only=True)
    since = _now() - 5
    log(f"[{account}] pushing build-only kernel {ref} (compiles llama.cpp, then exits)")
    ok, msg = kaggle.push_kernel(account, str(bdir))
    if not ok:
        log(f"[{account}] push failed: {msg}"); return None
    log(f"[{account}] pushed: {msg}  — waiting for build to finish (~15 min)...")

    ev = None
    deadline = time.time() + 2400
    while time.time() < deadline:
        for e in rendezvous.latest_events(cfg):
            if e.get("account") == account and e.get("event") == "built":
                ev = e; break
        if ev:
            break
        time.sleep(10)
    if not ev:
        log(f"[{account}] build did not complete cleanly"); return None

    st = "UNKNOWN"
    for _ in range(40):
        st = kaggle.kernel_status(account, ref)
        if st in ("COMPLETE", "ERROR", "CANCELED"):
            break
        time.sleep(15)
    log(f"[{account}] kernel {st}; fetching compiled binaries")

    dsdir = common.ROOT / "state" / "cache_build"
    if dsdir.exists():
        shutil.rmtree(dsdir)
    dsdir.mkdir(parents=True)
    kaggle.kernel_output(account, ref, str(dsdir), pattern="llamacpp-bin/*")
    binp = None
    for root, _dirs, files in __import__("os").walk(dsdir):
        if "llama-server" in files:
            binp = True; break
    if not binp:
        log(f"[{account}] could not find llama-server in fetched output"); return None

    slug = f"kllm-{model['key']}-bin" if model_key else "kllm-llamacpp-t4"
    ds_id = f"{user}/{slug}"
    meta = {"title": f"kllm llama.cpp CUDA ({model['key']})" if model_key else "kllm llama.cpp CUDA (T4/sm_75)",
            "id": ds_id, "licenses": [{"name": "other"}]}
    (dsdir / "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # private by default (no --public); create first time, else version
    rc, out = kaggle.run(account, "datasets", "create", "-p", str(dsdir), "-r", "zip", timeout=600)
    if "already exists" in out.lower() or "409" in out or rc != 0:
        rc, out = kaggle.run(account, "datasets", "version", "-p", str(dsdir),
                             "-m", "rebuild", "-r", "zip", timeout=600)
    log(f"[{account}] dataset {ds_id}: {out.strip().splitlines()[-1] if out.strip() else rc}")

    with common.mutate_state() as s:
        s["build_cache"][account] = ds_id
    log(f"[{account}] cache registered -> {ds_id}. Future launches on {account} skip the build.")
    return ds_id

def scale(count, model_key=None, log=print):
    """Bring the pool up to `count` parallel backends, one per account, launched together.
    Returns the list of accounts that became ready."""
    cfg = common.load_config()
    s = common.read_state()
    have = {b["account"] for b in s["backends"]}
    accts = [a["name"] for a in cfg["accounts"]]
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if allow:
        accts = [a for a in accts if any(str(a).startswith(p) for p in allow)]
    # choose accounts not already serving, in rotation order
    free = [a for a in accts if a not in have]
    need = max(0, count - len(have))
    chosen = free[:need]
    if not chosen:
        log(f"pool already has {len(have)} backend(s): {sorted(have)}"); return []
    log(f"scaling pool +{len(chosen)}: launching {chosen} in parallel")
    import threading
    results = {}
    def worker(acct):
        results[acct] = launch(model_key=model_key, account=acct, wait=True,
                               log=lambda *a: log(f"  {acct}:", *a))
    threads = [threading.Thread(target=worker, args=(a,)) for a in chosen]
    for t in threads: t.start()
    for t in threads: t.join()
    ready = [a for a, r in results.items() if r]
    log(f"pool ready: {ready}  (failed: {[a for a in chosen if a not in ready]})")
    return ready

def _free_account(cfg, account, log, kernel_ref=None):
    """Stop a running kernel on one account: graceful (gist control) + hard (delete frees
    the GPU slot immediately, since Kaggle has no cancel)."""
    publish_stop(account)                      # graceful: kernel closes tunnel and exits
    ref = kernel_ref or f"{account_user(cfg, account)}/{cfg.get('kernel_slug', 'kllm-server')}"
    rc, out = kaggle.delete_kernel(account, ref)
    log(f"[{account}] stop: graceful sent; delete {'ok' if 'deleted successfully' in out.lower() else out.strip()[:80]}")

def stop(account=None, log=print):
    """Stop a backend (or all) and free its Kaggle GPU slot. Drops it from routing too."""
    cfg = common.load_config()
    s = common.read_state()

    if account is None:
        try:
            rendezvous.publish_control(cfg, "all", "stop")
        except Exception:
            pass

    ref_map = {}
    for b in s.get("backends", []):
        if b.get("account") and b.get("kernel_ref"):
            ref_map[b["account"]] = b["kernel_ref"]
    for a, p in (s.get("pending") or {}).items():
        if isinstance(p, dict) and p.get("kernel_ref"):
            ref_map[a] = p["kernel_ref"]

    accounts = ([account] if account else
                list({b["account"] for b in s.get("backends", [])} | set(s.get("pending", {}).keys())))
    if not accounts:
        log("nothing running to stop"); return

    # Clean local state immediately so UI and CLI status update without delay
    for acct in accounts:
        common.remove_backend(acct)

    # Free accounts in parallel so stopping all backends finishes in seconds
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(accounts))) as ex:
        futures = {ex.submit(_free_account, cfg, acct, log, ref_map.get(acct)): acct for acct in accounts}
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                log(f"[{futures[f]}] error stopping: {e}")

