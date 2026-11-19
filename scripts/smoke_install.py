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
    base = pathlib.Path(tempfile.mkdtemp(prefix="kllm-smoke-"))
    venv, home, root = base / "venv", base / "home", base / "kllm"
    home.mkdir()
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    kllm = bindir / ("kllm.exe" if os.name == "nt" else "kllm")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KAGGLE_", "KLLM_", "PYTHON"))}
    env.update(HOME=str(home), USERPROFILE=str(home), KLLM_ROOT=str(root), KLLM_NO_UPDATE_CHECK="1",
               KLLM_NO_BROWSER="1", PATH=str(bindir) + os.pathsep + env.get("PATH", ""))

    print("install:", flush=True)
    r = subprocess.run([str(py), "-m", "pip", "install", "-q", str(wheel)], env=env, capture_output=True, text=True)
    check("pip install the wheel", r.returncode == 0, r.stderr[-500:])
    if r.returncode:
        return finish(base)
    dists = subprocess.run([str(py), "-m", "pip", "list", "--format=json"], env=env, capture_output=True, text=True).stdout
    names = {d["name"].lower() for d in json.loads(dists or "[]")}
    check("no obsolete 'typing' backport pulled in", "typing" not in names)

    print("imports:", flush=True)
    code = ("import importlib, pkgutil, gateway, sys\n"
            "bad = []\n"
            "for m in pkgutil.walk_packages(gateway.__path__, 'gateway.'):\n"
            "    try: importlib.import_module(m.name)\n"
            "    except Exception as e: bad.append(f'{m.name}: {type(e).__name__}: {e}')\n"
            "print('\\n'.join(bad)); sys.exit(1 if bad else 0)\n")
    r = subprocess.run([str(py), "-c", code], env=env, capture_output=True, text=True, cwd=str(base))
    check("every gateway module imports in a clean environment", r.returncode == 0, (r.stdout + r.stderr)[-800:])

    print("first run:", flush=True)
    r = subprocess.run([str(kllm), "account", "list"], env=env, capture_output=True, text=True, cwd=str(base), timeout=120)
    cfg = root / "config.yaml"
    check("first command writes a starter config without credentials", r.returncode == 0 and cfg.exists(), r.stderr[-400:])
    if not cfg.exists():
        return finish(base)
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("relay_port: 8080", f"relay_port: {PORT}"), encoding="utf-8")

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
    check("kllm serve starts and answers", up, (base / "serve.log").read_text(errors="replace")[-1500:])
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
            html = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/admin", timeout=5).read().decode()
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
            body, status = e.read().decode(errors="replace"), e.code
        except Exception as e:
            body, status = str(e), 0
        check("chat with no account fails fast with a pointer to the panel",
              status == 503 and "no_account" in body and time.time() - t0 < 15, f"{status} {body[:200]}")
        try:
            api = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/admin/api/update", timeout=10).read())
            check("update endpoint answers", "enabled" in api)
        except Exception as e:
            check("update endpoint answers", False, str(e))

    print("commands:", flush=True)
    r = subprocess.run([str(kllm), "update", "--check"], env={**env, "KLLM_NO_UPDATE_CHECK": "0"}, capture_output=True,
                       text=True, cwd=str(base), timeout=60)
    check("kllm update --check runs", r.returncode in (0, 1) and "Traceback" not in r.stderr, r.stderr[-400:])
    r = subprocess.run([str(kllm), "uninstall", "--dry-run"], env=env, capture_output=True, text=True, cwd=str(base), timeout=60)
    check("kllm uninstall --dry-run lists the plan", r.returncode == 0 and "Uninstall the kllm-gateway package" in r.stdout,
          (r.stdout + r.stderr)[-400:])
    r = subprocess.run([str(kllm), "shortcut", "--help"], env=env, capture_output=True, text=True, cwd=str(base), timeout=60)
    check("kllm shortcut is available", r.returncode == 0)
    if up:
        r = subprocess.run([str(kllm), "stop"], env=env, capture_output=True, text=True, cwd=str(base), timeout=60)
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
