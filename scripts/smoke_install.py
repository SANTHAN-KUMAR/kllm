"""Clean-room smoke test of a built kllm package, the way a user gets it.

    python scripts/smoke_install.py dist/kllm_gateway-X.Y.Z-py3-none-any.whl

Creates a fresh virtual environment with no access to globally installed packages, installs the wheel into it, and
then runs what a user runs: imports every module (catches undeclared dependencies), starts `kllm serve`, checks the
panel and the first-request behaviour, and exercises update, stop and uninstall --dry-run. Exits non-zero on the
first failure. Used by CI on Linux, macOS and Windows before every release.
"""
import json, os, pathlib, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

PORT = 18765
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def main(wheel):
    real_token = os.environ.get("KLLM_SMOKE_KAGGLE_TOKEN", "").strip()
    base = pathlib.Path(tempfile.mkdtemp(prefix="kllm-smoke-"))
    venv, home, root = base / "venv", base / "home", base / "kllm"
    home.mkdir()
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    kllm = bindir / ("kllm.exe" if os.name == "nt" else "kllm")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KAGGLE_", "KLLM_", "PYTHON"))}
    env.update(HOME=str(home), USERPROFILE=str(home), KLLM_ROOT=str(root), KLLM_NO_UPDATE_CHECK="1",
               KLLM_NO_BROWSER="1", PYTHONIOENCODING="utf-8", PATH=str(bindir) + os.pathsep + env.get("PATH", ""))

    def run_cmd(cmd, **kw):
        kw.setdefault("env", env)
        kw.setdefault("capture_output", True)
        kw.setdefault("text", True)
        kw.setdefault("encoding", "utf-8")
        kw.setdefault("errors", "replace")
        return subprocess.run(cmd, **kw)

    print("install:", flush=True)
    r = run_cmd([str(py), "-m", "pip", "install", "-q", str(wheel)])
    check("pip install the wheel", r.returncode == 0, r.stderr[-500:])
    if r.returncode:
        return finish(base)
    dists = run_cmd([str(py), "-m", "pip", "list", "--format=json"]).stdout
    names = {d["name"].lower() for d in json.loads(dists or "[]")}
    check("no obsolete 'typing' backport pulled in", "typing" not in names)

    print("imports:", flush=True)
    code = ("import importlib, pkgutil, gateway, sys\n"
            "bad = []\n"
            "for m in pkgutil.walk_packages(gateway.__path__, 'gateway.'):\n"
            "    try: importlib.import_module(m.name)\n"
            "    except Exception as e: bad.append(f'{m.name}: {type(e).__name__}: {e}')\n"
            "print('\\n'.join(bad)); sys.exit(1 if bad else 0)\n")
    r = run_cmd([str(py), "-c", code], cwd=str(base))
    check("every gateway module imports in a clean environment", r.returncode == 0, (r.stdout + r.stderr)[-800:])

    print("kaggle:", flush=True)
    IMPORT_ERR = ("ModuleNotFoundError", "ImportError", "SyntaxError")
    fake = "KGAT_" + "0" * 32
    kenv = {**env, "KAGGLE_API_TOKEN": fake}
    r = run_cmd([str(py), "-c", "from kaggle.api.kaggle_api_extended import KaggleApi"], env=kenv, cwd=str(base), timeout=120)
    check("the kaggle library imports (all its dependencies are installed)",
          r.returncode == 0 and not any(e in r.stderr for e in IMPORT_ERR), r.stderr[-500:])
    kcli = bindir / ("kaggle.exe" if os.name == "nt" else "kaggle")
    r = run_cmd([str(kcli), "--version"], env=kenv, cwd=str(base), timeout=120)
    check("the kaggle command runs", not any(e in (r.stdout + r.stderr) for e in IMPORT_ERR), (r.stdout + r.stderr)[-500:])
    # kllm's own credential check, with tokens Kaggle will reject: it must say "rejected/did not accept", never crash
    probe_root = base / "probe-root"
    for label, cred in (
        ("KGAT token", fake),
        ("kaggle.json", json.dumps({"username": "nobody_fake_xyz_999", "key": "0" * 32})),
    ):
        code = ("import os, pathlib, sys\n"
                "root = pathlib.Path(os.environ['KLLM_ROOT']); root.mkdir(parents=True, exist_ok=True)\n"
                "(root / 'config.yaml').write_text('accounts: []\\n', encoding='utf-8')\n"
                "from gateway import accounts\n"
                "try:\n"
                "    accounts.add_account(os.environ['PROBE_CRED'], name='kgsmoke')\n"
                "    print('ACCEPTED')\n"
                "except accounts.AccountError as e:\n"
                "    print('MSG:', e)\n")
        r = run_cmd([str(py), "-c", code], env={**env, "KLLM_ROOT": str(probe_root), "PROBE_CRED": cred},
                    cwd=str(base), timeout=180)
        msg = r.stdout.strip()
        check(f"credential check runs and rejects a fake {label} cleanly",
              msg.startswith("MSG:") and "could not run its Kaggle check" not in msg and "Traceback" not in msg
              and r.returncode == 0, (msg + r.stderr)[-500:])

    print("first run:", flush=True)
    r = run_cmd([str(kllm), "account", "list"], cwd=str(base), timeout=120)
    cfg = root / "config.yaml"
    check("first command writes a starter config without credentials", r.returncode == 0 and cfg.exists(), r.stderr[-400:])
    if not cfg.exists():
        return finish(base)
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("relay_port: 8080", f"relay_port: {PORT}"), encoding="utf-8")
    # what the panel does after Kaggle accepts a token: append the first account to this exact starter config
    code = ("import os, pathlib, yaml\nfrom gateway import yamledit\n"
            "p = pathlib.Path(os.environ['KLLM_ROOT']) / 'config.yaml'\n"
            "t = yamledit.append_account(p.read_text(encoding='utf-8'), {'name': 'kg1', 'user': 'smoke'})\n"
            "assert [a['name'] for a in yaml.safe_load(t)['accounts']] == ['kg1']\nprint('OK')\n")
    r = run_cmd([str(py), "-c", code], cwd=str(base), timeout=60)
    check("the first account can be added to the starter config", r.stdout.strip() == "OK", r.stderr[-500:])

    print("serve:", flush=True)
    log = open(base / "serve.log", "wb")
    proc = subprocess.Popen([str(kllm), "serve"], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(base),
                            stdin=subprocess.DEVNULL)
    up = False
    for _ in range(90):
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/__relay/health", timeout=2).read(); up = True; break
        except Exception:
            time.sleep(1)
    log.flush()
    check("kllm serve starts and answers", up, (base / "serve.log").read_text(encoding="utf-8", errors="replace")[-1500:])
    if up:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        try:
            urllib.request.build_opener(NoRedirect).open(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/", headers={"Accept": "text/html"}), timeout=5)
            code_, loc = 200, ""
        except urllib.error.HTTPError as e:
            code_, loc = e.code, e.headers.get("location", "")
        check("browser at / is sent to the panel", code_ == 307 and loc == "/admin", f"{code_} {loc}")
        try:
            html = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/admin", timeout=5).read().decode("utf-8", errors="replace")
            check("panel loads", "kllm" in html.lower())
        except Exception as e:
            check("panel loads", False, str(e))
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", method="POST",
                                     data=json.dumps({"model": "kllm", "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            urllib.request.urlopen(req, timeout=30); body, status = "", 200
        except urllib.error.HTTPError as e:
            body, status = e.read().decode("utf-8", errors="replace"), e.code
        except Exception as e:
            body, status = str(e), 0
        check("chat with no account fails fast with a pointer to the panel",
              status == 503 and "no_account" in body and time.time() - t0 < 15, f"{status} {body[:200]}")
        try:
            api = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/admin/api/update", timeout=10).read())
            check("update endpoint answers", "enabled" in api)
        except Exception as e:
            check("update endpoint answers", False, str(e))
        if real_token:
            def panel_api(method, path, payload=None):
                hdrs = {"X-Kllm-Panel": "1"}
                data = None
                if payload is not None:
                    hdrs["Content-Type"] = "application/json"
                    data = json.dumps(payload).encode("utf-8")
                rq = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers=hdrs, method=method)
                try:
                    with urllib.request.urlopen(rq, timeout=120) as resp:
                        return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read().decode("utf-8", errors="replace") or "{}")
                except Exception as e:
                    return 0, {"msg": str(e)}

            st, d = panel_api("POST", "/admin/api/accounts/add", {"token": real_token})
            check("real token adds first account through panel API", st == 200 and d.get("ok") is True and bool(d.get("user")),
                  f"{st} {d.get('msg', '')}")
            st, d = panel_api("GET", "/admin/api/accounts")
            accts = d.get("accounts") or []
            check("real account appears in /admin/api/accounts with valid credentials",
                  st == 200 and any(a.get("name") == "kg1" and (a.get("credentials") or {}).get("ok") is True for a in accts),
                  f"{st} {accts}")
            st, d = panel_api("POST", "/admin/api/accounts/sync-quota", {})
            check("real account sync-quota succeeds", st == 200 and "kg1" in (d.get("synced") or []),
                  f"{st} {d.get('msg', '')}")
            st, d = panel_api("POST", "/admin/api/accounts/remove",
                              {"name": "kg1", "delete_credentials": True, "confirm_credentials": "kg1"})
            check("real account removes cleanly", st == 200 and d.get("ok") is True, f"{st} {d.get('msg', '')}")

    print("commands:", flush=True)
    r = run_cmd([str(kllm), "update", "--check"], env={**env, "KLLM_NO_UPDATE_CHECK": "0"}, cwd=str(base), timeout=60)
    check("kllm update --check runs", r.returncode in (0, 1) and "Traceback" not in r.stderr, r.stderr[-400:])
    r = run_cmd([str(kllm), "uninstall", "--dry-run"], cwd=str(base), timeout=60)
    check("kllm uninstall --dry-run lists the plan", r.returncode == 0 and "Uninstall the kllm-gateway package" in r.stdout,
          (r.stdout + r.stderr)[-400:])
    r = run_cmd([str(kllm), "shortcut", "--help"], cwd=str(base), timeout=60)
    check("kllm shortcut is available", r.returncode == 0)
    if up:
        r = run_cmd([str(kllm), "stop"], cwd=str(base), timeout=60)
        gone = proc.wait(timeout=30) is not None if r.returncode == 0 else False
        check("kllm stop ends the relay", gone, r.stdout + r.stderr)
    if proc.poll() is None:
        proc.kill()
    log.close()
    return finish(base)


def finish(base):
    shutil.rmtree(base, ignore_errors=True)
    print(f"\n{'SMOKE PASSED' if not FAILS else 'SMOKE FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1]).resolve()))
