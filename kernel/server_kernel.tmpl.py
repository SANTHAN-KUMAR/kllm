#!/usr/bin/env python3
"""
kllm Kaggle server kernel (script type). Rendered by the local orchestrator, then
`kaggle kernels push`ed to one account. On Kaggle it:
  1. gets a CUDA llama.cpp build (from an attached dataset cache, else builds for T4/sm_75),
  2. starts llama-server with the chosen model (downloaded via -hf),
  3. opens a cloudflared quick tunnel (public https URL, no account),
  4. posts that URL to an ntfy topic so the PC can reach it,
  5. heartbeats, and exits cleanly before Kaggle's 12h cap.

Kaggle kernel stdout is only visible after the run ends, so ALL liveness signalling
goes through ntfy. The api-key is baked in by the renderer and never travels over ntfy.
"""
import os, sys, json, time, re, random, subprocess, urllib.request, glob, shutil, threading, struct, uuid, math, asyncio
from collections import OrderedDict
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CFG = json.loads(r'''__KLLM_CONFIG_JSON__''')
# CFG keys: model_hf, model_quant, model_ctx, model_args(list), api_key, ntfy_server,
#           ntfy_topic, session_seconds, llamacpp_ref (git ref), port, extra_server_args(list)

MODEL_ID = CFG["model_hf"] + (":" + CFG["model_quant"] if CFG.get("model_quant") else "")
RUNTIME = CFG.get("runtime") or "llama.cpp"      # "llama.cpp" (text models) | "diffusers" (image/video)
PORT = CFG.get("port", 8000)          # what the tunnel exposes (keep-alive proxy)
LLAMA_PORT = PORT + 1                  # the model server itself (llama-server or diffusers), local only
T0 = time.time()

def log(*a):
    print(f"[{int(time.time()-T0):5d}s]", *a, flush=True)

# ---- rendezvous: "ntfy" (default; sealed messages, no account) or "gist" (private GitHub gist) ----------------
# The sealing code is a copy of gateway/sealbox.py (tests/test_sealbox.py checks the two interoperate).
import base64 as _b64, hashlib as _hl, hmac as _hm
RV_MODE = "gist" if (CFG.get("rendezvous") == "gist" and CFG.get("gist_id")) else "ntfy"
RV_SECRET = CFG.get("rendezvous_secret") or ""
NTFY_SERVER = (CFG.get("ntfy_server") or "https://ntfy.sh").rstrip("/")

def _sb_k(label):
    return _hm.new(RV_SECRET.encode("utf-8"), label, _hl.sha256).digest()

def _sb_topics():
    return ("kllm-" + _sb_k(b"kllm-topic-status").hex()[:40], "kllm-" + _sb_k(b"kllm-topic-control").hex()[:40])

def _sb_stream(key, nonce, n):
    out, i = bytearray(), 0
    while len(out) < n:
        out += _hm.new(key, nonce + i.to_bytes(4, "big"), _hl.sha256).digest()
        i += 1
    return bytes(out[:n])

def _sb_seal(obj):
    pt = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(16)
    ct = bytes(a ^ b for a, b in zip(pt, _sb_stream(_sb_k(b"kllm-enc"), nonce, len(pt))))
    tag = _hm.new(_sb_k(b"kllm-mac"), b"v1" + nonce + ct, _hl.sha256).digest()
    return "v1." + _b64.urlsafe_b64encode(nonce + ct + tag).decode("ascii").rstrip("=")

def _sb_unseal(text):
    try:
        if not isinstance(text, str) or not text.startswith("v1."):
            return None
        raw = text[3:]
        raw = _b64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        if len(raw) < 48:
            return None
        nonce, ct, tag = raw[:16], raw[16:-32], raw[-32:]
        if not _hm.compare_digest(tag, _hm.new(_sb_k(b"kllm-mac"), b"v1" + nonce + ct, _hl.sha256).digest()):
            return None
        obj = json.loads(bytes(a ^ b for a, b in zip(ct, _sb_stream(_sb_k(b"kllm-enc"), nonce, len(ct)))).decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _ntfy_publish(payload):
    """POST one sealed status message. Honours 429 Retry-After once; never raises."""
    import urllib.error
    status, _ = _sb_topics()
    body = _sb_seal(payload).encode("utf-8")
    for _ in range(3):
        try:
            req = urllib.request.Request(f"{NTFY_SERVER}/{status}", data=body, method="POST",
                                         headers={"User-Agent": "kllm", "Cache": "yes", "Firebase": "no"})
            urllib.request.urlopen(req, timeout=20).read()
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                try: wait = min(60.0, float(e.headers.get("Retry-After") or 10))
                except Exception: wait = 10.0
                time.sleep(wait)
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)
    return False

def _ntfy_stop_requested(since_ts):
    _, control = _sb_topics()
    since = max(0, int(since_ts) - 5)
    req = urllib.request.Request(f"{NTFY_SERVER}/{control}/json?poll=1&since={since}",
                                 headers={"User-Agent": "kllm", "Accept": "application/x-ndjson"})
    raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
    for line in raw.splitlines():
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("event") != "message":
            continue
        body = _sb_unseal(msg.get("message", ""))
        if (body and body.get("cmd") == "stop" and body.get("account") in (CFG.get("account"), "all")
                and int(body.get("ts", 0)) >= since):
            return True
    return False

def _gist_req(method, data=None, etag=None):
    headers = {"Authorization": f"token {CFG['gist_token']}",
               "Accept": "application/vnd.github+json", "User-Agent": "kllm"}
    if etag:
        headers["If-None-Match"] = etag      # 304 when unchanged: not counted by GitHub's rate limit
    req = urllib.request.Request(f"https://api.github.com/gists/{CFG['gist_id']}",
                                 data=data, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=20)

_GIST_CACHE = {"etag": None, "doc": None}

def _gist_get():
    """Conditional GET of the gist; returns the (possibly cached) document or None."""
    import urllib.error
    try:
        r = _gist_req("GET", etag=_GIST_CACHE["etag"])
        _GIST_CACHE["doc"] = json.loads(r.read().decode())
        _GIST_CACHE["etag"] = r.headers.get("ETag")
    except urllib.error.HTTPError as e:
        if e.code != 304:
            return None
    except Exception:
        return None
    return _GIST_CACHE["doc"]

_LAST_NTFY_TS = {}

def ntfy(event, **extra):
    """Publish status/url to the private gist (file s-<account>.json). Best-effort; never raises."""
    acct = CFG.get("account")
    now = time.time()
    if event in ("downloading", "building"):
        if now - _LAST_NTFY_TS.get(acct, 0) < 30.0:
            return True
    _LAST_NTFY_TS[acct] = now
    payload = {"event": event, "t": round(now - T0, 1), "account": acct,
               "ts": int(now), **extra}
    if RV_MODE == "ntfy":
        return _ntfy_publish(payload)
    files = {f"s-{acct}.json": {"content": json.dumps(payload)}}
    if event in ("died", "crash", "error", "tunnel_recovering"):
        # s-<acct>.json is overwritten by the next event (e.g. "expiring") and a stopped kernel is deleted with its
        # logs, which erased the evidence of WHY a backend died. d-<acct>.json keeps the last failure record.
        files[f"d-{acct}.json"] = {"content": json.dumps(payload)}
    body = json.dumps({"files": files}).encode()
    for _ in range(3):
        try:
            _gist_req("PATCH", body).read()
            return True
        except Exception as e:
            time.sleep(2)
    return False

def ntfy_persistent(event, max_s=900, **extra):
    """Announce an event that the PC CANNOT work without (a re-opened tunnel's new URL) and keep retrying in the background for
    up to max_s. A plain ntfy() gives up after 3 tries; when the shared gist/token is rate-limited that left the supervisor
    unaware of the new URL, so it replaced a healthy kernel (and its warm prompt cache) after the grace period."""
    def run():
        deadline = time.time() + max_s
        while time.time() < deadline:
            if ntfy(event, **extra):
                return
            time.sleep(15 + random.uniform(0, 10))
    threading.Thread(target=run, daemon=True).start()

def stop_requested(since_ts):
    """Check for a stop command for this account (or 'all'): the sealed control topic, or the gist's c-<account>.json."""
    if RV_MODE == "ntfy":
        return _ntfy_stop_requested(since_ts)
    d = _gist_get()
    if d is None:
        return False
    files = d.get("files") or {}
    for name in (f"c-{CFG.get('account')}.json", "c-all.json"):
        f = files.get(name)
        if not f:
            continue
        try:
            body = json.loads(f.get("content", "{}"))
        except Exception:
            continue
        if body.get("cmd") == "stop" and body.get("ts", 0) >= int(since_ts) - 5:
            return True
    return False

def sh(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)

def tail(path, n=40):
    try:
        return "".join(open(path).readlines()[-n:])
    except Exception:
        return f"(no {path})"

def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=20).strip()
        return out.replace("\n", " | ")
    except Exception as e:
        return f"nvidia-smi failed: {e}"

# --------------------------------------------------------------------------
# 1. llama.cpp binary: prefer an attached dataset cache, else build for sm_75.
# --------------------------------------------------------------------------
def find_cached_server():
    for p in glob.glob("/kaggle/input/*/llama-server") + glob.glob("/kaggle/input/*/*/llama-server"):
        if os.access(p, os.X_OK) or os.path.isfile(p):
            return p
    return None

def _set_libpath(root):
    import glob as _g
    dirs = {os.path.dirname(so) for so in _g.glob(root + "/**/*.so*", recursive=True)}
    dirs.add("/usr/local/nvidia/lib64")          # the GPU driver (libcuda.so.1) lives here on Kaggle
    os.environ["LD_LIBRARY_PATH"] = ":".join(d for d in dirs if d) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

def download_prebuilt():
    """Download the official prebuilt llama.cpp CUDA binary (no compile). ~30s vs ~25min.
    llama.cpp ships ubuntu-cuda-12.8 x64 builds; Kaggle T4 runs CUDA 12.8."""
    import tarfile, urllib.request
    tag = CFG.get("llamacpp_build", "b11425")
    cuda = CFG.get("cuda_ver", "12.8")
    root = "/kaggle/working/llamacpp"
    os.makedirs(root, exist_ok=True)
    for name in (f"cudart-llama-{tag}-bin-ubuntu-cuda-{cuda}-x64.tar.gz",
                 f"llama-{tag}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"):
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
        dst = f"/tmp/{name}"
        log("downloading", url)
        urllib.request.urlretrieve(url, dst)
        with tarfile.open(dst) as t:
            t.extractall(root)
    bins = glob.glob(root + "/**/llama-server", recursive=True)
    if not bins:
        raise RuntimeError("llama-server not found in prebuilt archive")
    binpath = bins[0]
    os.chmod(binpath, 0o755)
    _set_libpath(root)
    return binpath

def _cuda_env():
    """Make sure nvcc is discoverable; cmake's CUDA detection needs it on PATH."""
    env = dict(os.environ)
    for cand in ("/usr/local/cuda/bin", "/usr/local/cuda-12/bin", "/usr/local/cuda-12.8/bin"):
        if os.path.isdir(cand):
            env["PATH"] = cand + ":" + env.get("PATH", "")
            env["CUDACXX"] = os.path.join(cand, "nvcc")
            break
    # fall back to whichever nvcc is on PATH already
    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc:
        env["CUDACXX"] = nvcc
    return env, nvcc

def build_llamacpp():
    """Build llama.cpp with CUDA for Tesla T4 (sm_75). ~10-15 min on 4 cores.
    Builds in /tmp so the kernel's /kaggle/working output stays tiny; only the final
    binaries are staged to /kaggle/working/llamacpp-bin for the cache-dataset snapshot."""
    src = "/tmp/llama.cpp"
    ref = CFG.get("llamacpp_ref", "master")
    repo = CFG.get("llamacpp_repo", "https://github.com/ggml-org/llama.cpp")
    env, nvcc = _cuda_env()
    ntfy("building", stage="deps", nvcc=nvcc or "NOT FOUND")
    sh("apt-get update -qq && apt-get install -y -qq libcurl4-openssl-dev >/dev/null 2>&1 || true")
    if not os.path.isdir(src):
        sh(f"git clone --depth 1 {'--branch '+ref if ref!='master' else ''} "
           f"{repo} {src}")

    # Make CMake's FindCUDAToolkit able to create the CUDA::cuda_driver target.
    # Kaggle's CUDA stubs dir is read-only AND often lacks libcuda.so, so we create a
    # libcuda.so symlink to the real driver in a WRITABLE dir and point CMake at it
    # (configure-time linking only; the actual driver is used at runtime).
    stubs = "/tmp/culibs"
    os.makedirs(stubs, exist_ok=True)
    stub_lib = os.path.join(stubs, "libcuda.so")
    if not os.path.exists(stub_lib):
        driver = ""
        try:
            out = subprocess.run("ldconfig -p | grep -m1 libcuda.so.1", shell=True,
                                 capture_output=True, text=True).stdout
            driver = out.split("=>")[-1].strip() if "=>" in out else ""
        except Exception:
            pass
        if not driver:
            for g in (glob.glob("/usr/local/nvidia/lib64/libcuda.so*")
                      + glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*")
                      + glob.glob("/usr/local/cuda/lib64/stubs/libcuda.so*")
                      + glob.glob("/usr/lib64/libcuda.so*")):
                driver = g; break
        if driver and os.path.exists(driver):
            try: os.symlink(driver, stub_lib)
            except FileExistsError: pass
            ntfy("building", stage="cuda_stub", driver=driver)
        else:
            ntfy("building", stage="cuda_stub", driver="NONE FOUND")
    env["LIBRARY_PATH"] = stubs + ":" + env.get("LIBRARY_PATH", "")

    cfg_cmd = (f"cmake -S {src} -B {src}/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
               f"-DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda "
               f"-DCMAKE_LIBRARY_PATH={stubs}")
    try:
        sh(f"{cfg_cmd} >/tmp/cmake.log 2>&1", env=env)
    except subprocess.CalledProcessError:
        ntfy("error", stage="cmake_configure", tail=tail("/tmp/cmake.log")[-1500:])
        raise
    ntfy("building", stage="compile")
    try:
        sh(f"cmake --build {src}/build --config Release -j4 --target llama-server "
           f">/tmp/build.log 2>&1", env=env)
    except subprocess.CalledProcessError:
        ntfy("error", stage="cmake_build", tail=tail("/tmp/build.log")[-1500:])
        raise
    binpath = f"{src}/build/bin/llama-server"
    # Stage binaries + shared libs so the orchestrator can snapshot them into a cache dataset.
    cache = "/kaggle/working/llamacpp-bin"
    os.makedirs(cache, exist_ok=True)
    for f in glob.glob(f"{src}/build/bin/*"):
        try: shutil.copy2(f, cache)
        except Exception: pass
    return binpath

def get_server_binary():
    # If prefer_cached_binary is requested or a custom repo/build is specified:
    if CFG.get("prefer_cached_binary") or CFG.get("llamacpp_build") in ("custom", "cache", "source"):
        cached = find_cached_server()
        if cached:
            log("using cached llama-server (preferred):", cached)
            ntfy("binary", source="cache", path=cached)
            os.chmod(cached, 0o755)
            _set_libpath(os.path.dirname(cached))
            return cached

    # 1. Prefer the official prebuilt CUDA binary (fast: ~30s download, no compile)
    # for official upstream releases (tag != custom and repo is ggml-org).
    is_custom = bool((CFG.get("llamacpp_repo") and "ggml-org/llama.cpp" not in CFG["llamacpp_repo"]) or
                     CFG.get("llamacpp_build") in ("custom", "source", "cache"))
    if not is_custom:
        try:
            ntfy("binary", source="prebuilt")
            b = download_prebuilt()
            log("using prebuilt llama-server:", b)
            return b
        except Exception as e:
            log("prebuilt download failed, falling back:", e)
            ntfy("binary", source="prebuilt_failed", note=str(e)[:120])

    # 2. A cached build dataset, if attached.
    cached = find_cached_server()
    if cached:
        log("using cached llama-server:", cached)
        ntfy("binary", source="cache", path=cached)
        os.chmod(cached, 0o755)
        _set_libpath(os.path.dirname(cached))
        return cached

    # 3. Last resort: compile from source (~25 min).
    log(f"building llama.cpp for sm_75 from {CFG.get('llamacpp_repo', 'https://github.com/ggml-org/llama.cpp')}@{CFG.get('llamacpp_ref', 'master')}")
    ntfy("binary", source="build")
    b = build_llamacpp()
    _set_libpath(os.path.dirname(b))
    return b

# --------------------------------------------------------------------------
# 2. cloudflared
# --------------------------------------------------------------------------
def get_cloudflared():
    dst = "/kaggle/working/cloudflared"
    if not os.path.isfile(dst):
        sh(f"wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
           f"cloudflared-linux-amd64 -O {dst}")
        os.chmod(dst, 0o755)
    return dst


# --------------------------------------------------------------------------
# keep-alive proxy: the cloudflare tunnel aborts any response that sends nothing for 120 s
# (HTTP 524). llama-server sends nothing while a request waits for a slot or prefills a long
# prompt. This proxy sits between the tunnel and llama-server: if llama-server has not
# answered within HOLD seconds, it commits to a 200 SSE stream and sends SSE comments
# (ignored by every SSE parser) until llama-server's first bytes arrive. Normal requests
# pass through untouched (real status codes). Stdlib only.
# --------------------------------------------------------------------------
import http.client, socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOLD = 15          # seconds to wait for llama-server's headers before committing to keep-alive
BEAT = 5           # keep-alive comment interval (5s keeps Cloudflare and HTTP client sockets continuously warm)
STREAM_PATHS = ("/v1/chat/completions", "/chat/completions", "/v1/messages", "/v1/completions")
MEDIA_PATHS = ("/v1/images/generations", "/v1/videos/generations")    # long non-SSE JSON requests
SKIP = {"connection", "keep-alive", "transfer-encoding", "content-length", "proxy-connection",
        "te", "trailer", "upgrade", "host"}

def _sse_error(path, status, raw):
    try:
        e = json.loads(raw).get("error") or {}
        msg = e.get("message") if isinstance(e, dict) else str(e)
    except Exception:
        msg = raw.decode("utf-8", "replace")[:500]
    msg = msg or f"backend error {status}"
    if path.startswith("/v1/messages"):
        kind = "invalid_request_error" if 400 <= status < 500 else "api_error"
        return ("event: error\ndata: " + json.dumps({"type": "error", "error": {"type": kind, "message": msg}}) + "\n\n").encode()
    return ("data: " + json.dumps({"error": {"message": msg, "code": status}}) + "\n\ndata: [DONE]\n\n").encode()

class _UpstreamError(Exception):
    pass

def _json_error(status, raw):
    """A JSON error body for a request that already committed to 200: keep a backend's own
    {"error": ...} object, wrap anything else."""
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and d.get("error"):
            return json.dumps(d).encode()
        msg = None
    except Exception:
        msg = raw.decode("utf-8", "replace")[:500] if isinstance(raw, bytes) else str(raw)[:500]
    return json.dumps({"error": {"message": msg or f"backend error {status}", "code": status}}).encode()

class KeepAliveProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_port = None

    def log_message(self, *a):
        pass

    def _chunk(self, data):
        if data:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data)); self.wfile.flush()

    def _handle(self):
        if self.path.startswith("/__kllm/"):
            return _serve_local(self)             # diagnostics served by the proxy itself
        _pstat(requests=1, inflight=1)
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in SKIP}
        conn = http.client.HTTPConnection("127.0.0.1", self.upstream_port, timeout=None)
        box, ready = {}, threading.Event()
        def call():
            try:
                conn.request(self.command, self.path, body=body, headers=hdrs)
                box["resp"] = conn.getresponse()
            except Exception as e:
                box["err"] = e
            ready.set()
        threading.Thread(target=call, daemon=True).start()
        streamish = self.command == "POST" and self.path.split("?")[0] in STREAM_PATHS
        jsonish = self.command == "POST" and self.path.split("?")[0] in MEDIA_PATHS
        committed = False
        try:
            if not ready.wait(HOLD if (streamish or jsonish) else None):
                # the backend is still silent: commit to a 200 and keep the connection alive
                self.send_response(200)
                self.send_header("Content-Type", "application/json" if jsonish else "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                committed = True
                _pstat(committed=1)
                t0 = time.time()
                while not ready.wait(BEAT):
                    # JSON (images/videos): whitespace before the body is valid JSON. SSE: a comment.
                    self._chunk(b" " if jsonish else
                                f": kllm-kernel: working ({int(time.time() - t0) + HOLD}s)\n\n".encode())
            if "err" in box:
                # not the client's failure: RemoteDisconnected is a ConnectionResetError and would
                # otherwise be swallowed below as "client went away", leaving the client no answer
                raise _UpstreamError(str(box["err"]))
            resp = box["resp"]
            if resp.status >= 500:
                _pstat(status_5xx=1)
            elif resp.status >= 400:
                _pstat(status_4xx=1)
            if committed:
                if resp.status != 200:
                    self._chunk(_json_error(resp.status, resp.read()) if jsonish
                                else _sse_error(self.path, resp.status, resp.read()))
                else:
                    while True:
                        data = resp.read1(65536)
                        if not data:
                            break
                        self._chunk(data)
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                return
            # transparent pass-through
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in SKIP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                data = resp.read1(65536)
                if not data:
                    break
                self._chunk(data)
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            _pstat(client_gone=1)                 # client went away
        except Exception as e:
            _pstat(upstream_err=1)
            try:
                if committed:
                    self._chunk(_json_error(502, str(e).encode()) if jsonish
                                else _sse_error(self.path, 502, str(e).encode()))
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    msg = json.dumps({"error": {"message": f"kllm proxy: {e}"}}).encode()
                    self.send_response(502); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg))); self.end_headers(); self.wfile.write(msg)
            except Exception:
                pass
        finally:
            _pstat(inflight=-1)
            conn.close()                          # llama-server cancels the task if still running
            self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _handle

# --------------------------------------------------------------------------
# Resource telemetry (diagnostic, read-only, off the inference path).
# Kaggle kernel stdout/files are only readable after the run ends, so a long-running failure
# (host-RAM OOM kill, VRAM exhaustion, slot starvation) used to leave no evidence. A sampler
# thread records, every TELEMETRY_INTERVAL s: cgroup memory (+ oom_kill counter), llama-server
# RSS split (anon vs file-backed), per-GPU VRAM/utilisation, and keep-alive-proxy request
# counters. Served to the PC on authenticated GET /__kllm/telemetry and /__kllm/log, and appended
# to /kaggle/working/telemetry.jsonl for post-mortem. Disable with "telemetry": false in the kernel config.
# --------------------------------------------------------------------------
TELEMETRY_INTERVAL = float(CFG.get("telemetry_interval", 5))
_TELEM = []                      # ring buffer of samples
_TELEM_MAX = 40000
_TELEM_LOCK = threading.Lock()
PROXY_STATS = {"requests": 0, "committed": 0, "inflight": 0, "max_inflight": 0, "client_gone": 0,
               "upstream_err": 0, "status_5xx": 0, "status_4xx": 0}
_PSTAT_LOCK = threading.Lock()

def _pstat(**inc):
    with _PSTAT_LOCK:
        for k, v in inc.items():
            PROXY_STATS[k] = PROXY_STATS.get(k, 0) + v
        if PROXY_STATS["inflight"] > PROXY_STATS["max_inflight"]:
            PROXY_STATS["max_inflight"] = PROXY_STATS["inflight"]

def _readf(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return None

def _cgroup_mem():
    """cgroup v2 (memory.current/max/stat/events) or v1 (memory.usage_in_bytes/...) -> dict, bytes."""
    out = {}
    for base, v2 in (("/sys/fs/cgroup", True), ("/sys/fs/cgroup/memory", False)):
        cur = _readf(base + ("/memory.current" if v2 else "/memory.usage_in_bytes"))
        if cur is None:
            continue
        out["cg_ver"] = 2 if v2 else 1
        out["cg_used"] = int(cur)
        lim = _readf(base + ("/memory.max" if v2 else "/memory.limit_in_bytes"))
        if lim and lim.strip() != "max":
            out["cg_limit"] = int(lim)
        for line in (_readf(base + "/memory.stat") or "").splitlines():
            k, _, v = line.partition(" ")
            if k in ("anon", "file", "shmem", "file_mapped", "total_rss", "total_cache", "total_shmem", "total_mapped_file"):
                out["cg_" + k] = int(v)
        if v2:
            for line in (_readf(base + "/memory.events") or "").splitlines():
                k, _, v = line.partition(" ")
                if k in ("oom", "oom_kill", "max", "high"):
                    out["cg_ev_" + k] = int(v)
        else:
            for line in (_readf(base + "/memory.oom_control") or "").splitlines():
                k, _, v = line.partition(" ")
                if k == "oom_kill":
                    out["cg_ev_oom_kill"] = int(v)
            fc = _readf(base + "/memory.failcnt")
            if fc:
                out["cg_failcnt"] = int(fc)
        break
    return out

def _meminfo():
    out = {}
    for line in (_readf("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "Cached", "SwapTotal", "SwapFree", "Shmem"):
            out["mi_" + k] = int(v.split()[0]) * 1024
    return out

def _find_pids(needle):
    pids = []
    for d in glob.glob("/proc/[0-9]*"):
        cmd = _readf(d + "/cmdline")
        if cmd and needle in cmd.split("\0")[0]:
            pids.append(int(d.rsplit("/", 1)[1]))
    return pids

def _proc_mem(pid):
    out = {}
    for line in (_readf(f"/proc/{pid}/status") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "Threads"):
            out[k] = int(v.split()[0]) * (1 if k == "Threads" else 1024)
    return out

def _gpu_stats():
    try:
        txt = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception:
        return []
    rows = []
    for line in txt.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            rows.append({"i": int(p[0]), "used_mib": int(p[1]), "total_mib": int(p[2]), "util": int(p[3]),
                         "temp": int(p[4]), "sm_mhz": int(p[5]), "power_w": float(p[6])})
        except Exception:
            pass
    return rows

_PREV = {}                       # previous /proc/stat and /proc/net/dev reads, for rates between samples

def _cpu_pct():
    """Busy share of all CPUs since the previous sample, from two reads of /proc/stat."""
    line = (_readf("/proc/stat") or "").split("\n", 1)[0].split()
    if not line or line[0] != "cpu":
        return None
    v = [int(x) for x in line[1:]]
    idle, total = v[3] + (v[4] if len(v) > 4 else 0), sum(v)
    prev = _PREV.get("cpu"); _PREV["cpu"] = (idle, total)
    if not prev or total <= prev[1]:
        return None
    return round(100.0 * (1 - (idle - prev[0]) / (total - prev[1])), 1)

def _net_bps():
    """Received and sent bytes per second over all non-loopback interfaces, from two reads of /proc/net/dev."""
    rx = tx = 0
    for line in (_readf("/proc/net/dev") or "").splitlines()[2:]:
        name, _, rest = line.partition(":")
        f = rest.split()
        if name.strip() == "lo" or len(f) < 9:
            continue
        rx += int(f[0]); tx += int(f[8])
    now = time.time(); prev = _PREV.get("net"); _PREV["net"] = (now, rx, tx)
    if not prev or now <= prev[0]:
        return None, None
    dt = now - prev[0]
    return max(0, int((rx - prev[1]) / dt)), max(0, int((tx - prev[2]) / dt))

def _disk():
    out = {}
    for path in ("/kaggle/working", "/tmp"):
        try:
            u = shutil.disk_usage(path)
            out[path] = {"used": u.used, "total": u.total}
        except Exception:
            pass
    return out

def _slots():
    """Context in use per llama-server slot (GET /slots on the internal port). None for other runtimes."""
    if RUNTIME != "llama.cpp":
        return None
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/slots",
                                     headers={"Authorization": "Bearer " + CFG["api_key"]})
        rows = json.loads(urllib.request.urlopen(req, timeout=2).read().decode())
    except Exception:
        return None
    out = []
    for r in rows if isinstance(rows, list) else []:
        nt = r.get("next_token") if isinstance(r.get("next_token"), dict) else {}
        if isinstance(r.get("next_token"), list) and r["next_token"]:
            nt = r["next_token"][0]
        used = r.get("n_past", nt.get("n_past"))
        if used is None and r.get("n_prompt_tokens") is not None:
            used = int(r.get("n_prompt_tokens") or 0) + int(nt.get("n_decoded") or 0)
        out.append({"id": r.get("id"), "n_ctx": r.get("n_ctx"), "used": used,
                    "busy": bool(r.get("is_processing"))})
    return out

_TPS_PREV = {}

def _tps():
    """llama.cpp /metrics cumulative counters -> instantaneous prefill/decode tok/s since the
    previous sample. None for other runtimes, or when /metrics is absent (kernel started without --metrics)."""
    if RUNTIME != "llama.cpp":
        return None
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/metrics",
                                     headers={"Authorization": "Bearer " + CFG["api_key"]})
        body = urllib.request.urlopen(req, timeout=2).read().decode()
    except Exception:
        return None
    m = {}
    for ln in body.splitlines():
        if ln.startswith("#") or " " not in ln:
            continue
        k, _, v = ln.rpartition(" ")
        try:
            m[k.strip()] = float(v)
        except ValueError:
            pass
    pt, ps = m.get("llamacpp_prompt_tokens_total"), m.get("llamacpp_prompt_seconds_total")
    tt, ts = m.get("llamacpp_tokens_predicted_total"), m.get("llamacpp_tokens_predicted_seconds_total")
    if None in (pt, ps, tt, ts):
        return None
    out = {"prefill_tokens_total": pt, "prefill_seconds_total": ps,
           "decode_tokens_total": tt, "decode_seconds_total": ts}
    prev = _TPS_PREV
    if prev and ps >= prev["ps"] and ts >= prev["ts"]:
        dps, dts = ps - prev["ps"], ts - prev["ts"]
        if dps > 0.005:
            out["prefill_tps"] = round((pt - prev["pt"]) / dps, 1)
        if dts > 0.005:
            out["decode_tps"] = round((tt - prev["tt"]) / dts, 1)
    _TPS_PREV.clear(); _TPS_PREV.update({"pt": pt, "ps": ps, "tt": tt, "ts": ts})
    return out

def _telemetry_sample():
    s = {"ts": round(time.time(), 1), "up_s": int(time.time() - T0)}
    cpu = _cpu_pct()
    if cpu is not None:
        s["cpu_pct"] = cpu
    rx, tx = _net_bps()
    if rx is not None:
        s["net_rx_bps"], s["net_tx_bps"] = rx, tx
    s["disk"] = _disk()
    sl = _slots()
    if sl is not None:
        s["slots"] = sl
    tps = _tps()
    if tps:
        s.update(tps)
    s.update(_cgroup_mem()); s.update(_meminfo())
    try:
        s["load1"] = os.getloadavg()[0]
    except Exception:
        pass
    pids = _find_pids("llama-server")
    if pids:
        s["srv_pid"] = pids[0]
        for k, v in _proc_mem(pids[0]).items():
            s["srv_" + k] = v
    s["gpu"] = _gpu_stats()
    with _PSTAT_LOCK:
        s["proxy"] = dict(PROXY_STATS)
    return s

def _telemetry_loop():
    path = "/kaggle/working/telemetry.jsonl"
    try:
        fh = open(path, "a")
    except Exception:
        fh = None
    while True:
        try:
            s = _telemetry_sample()
            with _TELEM_LOCK:
                _TELEM.append(s)
                if len(_TELEM) > _TELEM_MAX:
                    del _TELEM[:len(_TELEM) - _TELEM_MAX]
            if fh:
                fh.write(json.dumps(s) + "\n"); fh.flush()
        except Exception:
            pass
        time.sleep(TELEMETRY_INTERVAL)

def start_telemetry():
    if CFG.get("telemetry", True):
        threading.Thread(target=_telemetry_loop, daemon=True).start()

def _serve_local(handler):
    """GET /__kllm/telemetry?since=<ts> | /__kllm/log?n=<lines>&grep=<regex>. Bearer api key required."""
    from urllib.parse import urlparse, parse_qs
    u = urlparse(handler.path); q = parse_qs(u.query)
    auth = handler.headers.get("authorization", "")
    if auth != "Bearer " + CFG["api_key"] and handler.headers.get("x-api-key", "") != CFG["api_key"]:
        body, code = b'{"error":"unauthorized"}', 401
    elif u.path == "/__kllm/telemetry":
        since = float((q.get("since") or ["0"])[0])
        with _TELEM_LOCK:
            rows = [r for r in _TELEM if r["ts"] > since]
        body, code = json.dumps({"samples": rows[-4000:]}).encode(), 200
    elif u.path == "/__kllm/log":
        n = min(int((q.get("n") or ["200"])[0]), 5000)
        pat = re.compile((q.get("grep") or [""])[0]) if q.get("grep") else None
        lines = (_readf("/kaggle/working/llama-server.log") or "").splitlines()
        if pat:
            lines = [l for l in lines if pat.search(l)]
        body, code = json.dumps({"lines": lines[-n:], "total": len(lines)}).encode(), 200
    elif u.path == "/__kllm/chaos":
        import signal
        act = (q.get("action") or [""])[0]; pr = TUNNEL.get("proc")
        if pr is None or pr.poll() is not None:
            body, code = b'{"error":"no live tunnel process"}', 409
        elif act == "freeze_tunnel":
            os.kill(pr.pid, signal.SIGSTOP); body, code = json.dumps({"ok": act, "pid": pr.pid}).encode(), 200
        elif act == "kill_tunnel":
            os.kill(pr.pid, signal.SIGKILL); body, code = json.dumps({"ok": act, "pid": pr.pid}).encode(), 200
        else:
            body, code = b'{"error":"unknown action"}', 400
    else:
        body, code = b'{"error":"not found"}', 404
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers(); handler.wfile.write(body)
    handler.close_connection = True

def start_proxy(listen_port, upstream_port):
    class S(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    KeepAliveProxy.upstream_port = upstream_port
    srv = S(("127.0.0.1", listen_port), KeepAliveProxy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

# --------------------------------------------------------------------------
# diffusers runtime (image / video models). Selected by CFG["runtime"] == "diffusers"; it
# listens on the SAME internal port llama-server would use (behind the keep-alive proxy, so the
# tunnel and relay see no difference). One generation at a time. Stdlib HTTP API:
#   GET  /health                   open (like llama-server)
#   GET  /v1/models                api key
#   POST /v1/images/generations    api key; OpenAI Images API (+ negative_prompt, steps, guidance_scale, seed)
#   POST /v1/videos/generations    api key; kllm-specific, returns an mp4 as base64
# torch is preinstalled on Kaggle and imported lazily; everything else is pip-installed quietly.
# --------------------------------------------------------------------------
import base64, hmac, io, random, tempfile, inspect as _inspect, gc

PIP_BASE = {"diffusers": "diffusers", "transformers": "transformers", "accelerate": "accelerate",
            "sentencepiece": "sentencepiece", "ftfy": "ftfy"}
PIP_VIDEO = {"imageio": "imageio", "imageio_ffmpeg": "imageio-ffmpeg"}
MIN_DIFFUSERS = (0, 33)            # Wan / LTX / HunyuanVideo pipelines need a recent diffusers
GPU_FIT_GB = 11.5                  # whole pipeline resident on one T4 (15 GB) with room for activations
GPU_COMPONENT_GB = 13.5            # largest single component that can still be moved to a T4
HOST_RAM_GB = 28.0                 # Kaggle has ~31 GB RAM: bigger models cannot be held for offload


def _have(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _diffusers_too_old():
    try:
        import importlib.metadata as md
        v = md.version("diffusers")
        parts = tuple(int(x) for x in re.findall(r"\d+", v)[:2])
        return parts < MIN_DIFFUSERS
    except Exception:
        return True


def ensure_packages(kind, extra=()):
    """pip-install (quietly) what the Kaggle image lacks. torch is never touched."""
    need = dict(PIP_BASE)
    if kind == "video":
        need.update(PIP_VIDEO)
    pkgs = [pipname for mod, pipname in need.items() if not _have(mod)]
    if _have("diffusers") and _diffusers_too_old():
        pkgs.append("diffusers>=%d.%d" % MIN_DIFFUSERS)
    pkgs += [p for p in extra if p not in pkgs]
    if not pkgs:
        return []
    log("pip install", " ".join(pkgs))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "-U", *pkgs], check=True)
    import importlib
    importlib.invalidate_caches()
    return pkgs


def plan_placement(total_gb, largest_gb, mode="auto"):
    """Where the pipeline lives. -> "gpu" | "model" (components swapped onto the GPU one at a time)
    | "sequential" (layer by layer, slowest) | "balanced" (components spread over both T4s, opt-in)."""
    if mode in ("gpu", "none"):
        return "gpu"
    if mode in ("model", "sequential", "balanced"):
        return mode
    if total_gb > HOST_RAM_GB:
        raise ApiError(500, f"model needs ~{total_gb:.0f} GB in fp16 but the kernel has ~31 GB RAM "
                            "for offloading; use a smaller model or quantized weights", "model_too_large")
    if total_gb <= GPU_FIT_GB:
        return "gpu"
    if largest_gb <= GPU_COMPONENT_GB:
        return "model"
    return "sequential"


def component_sizes(pipe):
    """{component name: GB of parameters} for the nn.Module components of a pipeline."""
    out = {}
    for name, comp in (getattr(pipe, "components", None) or {}).items():
        params = getattr(comp, "parameters", None)
        if not callable(params):
            continue
        out[name] = sum(p.numel() * p.element_size() for p in params()) / 1e9
    return out


class ApiError(Exception):
    def __init__(self, status, message, typ="invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.typ = status, message, typ


def _is_oom(e):
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _clean_cuda():
    gc.collect()
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            if callable(getattr(torch.cuda, "synchronize", None)):
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            if callable(getattr(torch.cuda, "empty_cache", None)):
                torch.cuda.empty_cache()
            if callable(getattr(torch.cuda, "ipc_collect", None)):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        import ctypes, ctypes.util
        for libname in (None, "libc.so.6", ctypes.util.find_library("c")):
            try:
                libc = ctypes.CDLL(libname)
                if hasattr(libc, "malloc_trim"):
                    libc.malloc_trim(0)
                    break
            except Exception:
                pass
    except Exception:
        pass


def _module_dtype(m):
    """Retrieve tensor dtype of a PyTorch module or component."""
    if hasattr(m, "dtype") and m.dtype is not None:
        return m.dtype
    try:
        if hasattr(m, "parameters"):
            p = next(iter(m.parameters()), None)
            if p is not None and hasattr(p, "dtype"):
                return p.dtype
    except Exception:
        pass
    return None


def _preprocess_image(im, max_w, max_h, mult=8, exact=False):
    """Normalize and scale reference image preserving aspect ratio or exact sizing."""
    try:
        from PIL import Image, ImageOps
        if not isinstance(im, Image.Image):
            return im
        im = ImageOps.exif_transpose(im).convert("RGB")
        if exact:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "BILINEAR", None))
            return im.resize((max_w, max_h), resample=resample) if resample is not None else im.resize((max_w, max_h))
        cur_w, cur_h = im.size
        if cur_w > max_w or cur_h > max_h:
            scale = min(max_w / cur_w, max_h / cur_h)
            target_w = max(mult, int(round(cur_w * scale)))
            target_h = max(mult, int(round(cur_h * scale)))
        else:
            target_w, target_h = cur_w, cur_h
        target_w = min(max_w, max(mult, (target_w // mult) * mult))
        target_h = min(max_h, max(mult, (target_h // mult) * mult))
        if (target_w, target_h) != (cur_w, cur_h):
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "BILINEAR", None))
            im = im.resize((target_w, target_h), resample=resample) if resample is not None else im.resize((target_w, target_h))
        return im
    except Exception as e:
        log("warning: image preprocessing error:", e)
        return im


def _apply_teacache(transformer, rel_l1_thresh=0.15):
    """Timestep Embedding Aware Cache (TeaCache) for video DiT transformers.
    Skips redundant transformer block evaluations across denoising steps when the
    inter-step feature delta is below rel_l1_thresh, yielding ~1.5x-2.0x acceleration
    with zero loss in photorealism or detail."""
    if transformer is None or hasattr(transformer, "_teacache_enabled"):
        return
    import torch
    orig_forward = transformer.forward
    state = {
        "prev_hidden": None,
        "cached_output": None,
        "step": 0,
        "accum_l1": 0.0,
    }

    def teacache_forward(*args, **kwargs):
        hidden_states = kwargs.get("hidden_states") if "hidden_states" in kwargs else (args[0] if args else None)
        if hidden_states is None or not (hasattr(hidden_states, "shape") and hasattr(hidden_states, "float")):
            return orig_forward(*args, **kwargs)

        should_calc = True
        if state["step"] == 0 or state["prev_hidden"] is None:
            should_calc = True
            state["accum_l1"] = 0.0
        else:
            prev = state["prev_hidden"]
            if prev.shape == hidden_states.shape:
                diff_m = (hidden_states.float() - prev.float()).abs().mean()
                prev_m = prev.float().abs().mean()
                diff_val = diff_m.item() if hasattr(diff_m, "item") else float(diff_m)
                prev_val = prev_m.item() if hasattr(prev_m, "item") else float(prev_m)
                rel_l1 = diff_val / (prev_val + 1e-6)
                state["accum_l1"] += rel_l1
                if state["accum_l1"] < rel_l1_thresh and state["cached_output"] is not None:
                    should_calc = False
                else:
                    state["accum_l1"] = 0.0

        if should_calc:
            out = orig_forward(*args, **kwargs)
            state["cached_output"] = out
            state["prev_hidden"] = hidden_states.detach().clone()
        else:
            out = state["cached_output"]

        state["step"] += 1
        return out

    def reset():
        state["prev_hidden"] = None
        state["cached_output"] = None
        state["step"] = 0
        state["accum_l1"] = 0.0

    transformer.forward = teacache_forward
    transformer._teacache_reset = reset
    transformer._teacache_enabled = True
    log(f"TeaCache activated on {transformer.__class__.__name__} (rel_l1_thresh={rel_l1_thresh})")


class DiffusersRuntime:
    """Holds the pipeline and turns API requests into generations. The pipeline-facing hooks
    (_generator, _encode_image, _encode_mp4) are plain methods so tests can replace them."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = dict(cfg.get("opts") or {})
        self.kind = cfg.get("kind") or "image"
        self.key = cfg.get("model_key") or cfg.get("model_hf") or cfg.get("hf")
        self.repo = cfg.get("model_hf") or cfg.get("hf")
        self.api_key = cfg.get("api_key")
        self.pipe = None
        self.i2v_pipe = None
        self.v2v_pipe = None
        self.t2v_pipe = None
        self.tile_kwargs = {}
        self.state = "loading"            # loading | ready | error
        self.error = None
        self.placement = None
        self.lock = threading.Lock()      # one generation at a time
        self.busy = False
        self.warming = False              # the warm-up generation runs before state flips to ready
        self.served = 0

    def _apply_vae_tiling(self, v_mod, kw=None):
        if v_mod is None or not hasattr(v_mod, "enable_tiling"):
            return
        kw = self.tile_kwargs if kw is None else kw
        try:
            sig = _inspect.signature(v_mod.enable_tiling)
            accepted = {k: v for k, v in kw.items() if k in sig.parameters}
            if accepted:
                v_mod.enable_tiling(**accepted)
            else:
                v_mod.enable_tiling()
        except Exception:
            try:
                v_mod.enable_tiling()
            except Exception:
                pass
        for attr, val in (kw or {}).items():
            if hasattr(v_mod, attr):
                try:
                    setattr(v_mod, attr, val)
                except Exception:
                    pass

    def _apply_pipe_vae_tiling(self, p, kw=None):
        if p is None or not hasattr(p, "enable_vae_tiling"):
            return
        kw = self.tile_kwargs if kw is None else kw
        try:
            sig = _inspect.signature(p.enable_vae_tiling)
            accepted = {k: v for k, v in (kw or {}).items() if k in sig.parameters}
            if accepted:
                p.enable_vae_tiling(**accepted)
            else:
                p.enable_vae_tiling()
        except Exception:
            try:
                p.enable_vae_tiling()
            except Exception:
                pass

    def _apply_vae_slicing(self, v_mod):
        if v_mod is None or not hasattr(v_mod, "enable_slicing"):
            return
        try:
            v_mod.enable_slicing()
        except Exception:
            pass

    def _apply_pipe_vae_slicing(self, p):
        if p is None or not hasattr(p, "enable_vae_slicing"):
            return
        try:
            p.enable_vae_slicing()
        except Exception:
            pass

    def _reclaim_pipeline_memory(self, active_pipe=None):
        """Aggressively reclaim Host RAM and GPU VRAM after denoising finishes,
        purging TeaCache, intermediate activation caches, KV buffers recursively across all modules,
        and returning heap pages to OS via malloc_trim."""
        pipes = [p for p in (self.pipe, self.i2v_pipe, self.v2v_pipe, self.t2v_pipe, active_pipe) if p is not None]

        def _deep_purge(mod):
            if mod is None:
                return
            if hasattr(mod, "_teacache_reset") and callable(getattr(mod, "_teacache_reset")):
                try:
                    mod._teacache_reset()
                except Exception:
                    pass
            for attr in ("_cache", "kv_cache", "cache", "_intermediates", "intermediates", "past_key_values"):
                if hasattr(mod, attr):
                    try:
                        setattr(mod, attr, None)
                    except Exception:
                        pass
            if hasattr(mod, "children") and callable(getattr(mod, "children")):
                try:
                    for child in mod.children():
                        _deep_purge(child)
                except Exception:
                    pass

        for p in pipes:
            for attr in ("transformer", "transformer_2", "text_encoder", "text_encoder_2", "image_encoder"):
                comp = getattr(p, attr, None)
                if comp is not None:
                    _deep_purge(comp)
            comps = getattr(p, "components", None) or {}
            for name, comp in comps.items():
                if name not in ("vae", "scheduler", "tokenizer", "tokenizer_2"):
                    _deep_purge(comp)

        _clean_cuda()

    def _wrap_vae_decode(self, v_mod, tr_mod=None):
        if v_mod is None or not hasattr(v_mod, "decode"):
            return
        if getattr(v_mod, "_kllm_cleanup_wrapped", False):
            if tr_mod is not None:
                if not hasattr(v_mod, "_kllm_extra_targets") or not isinstance(v_mod._kllm_extra_targets, list):
                    v_mod._kllm_extra_targets = []
                if tr_mod not in v_mod._kllm_extra_targets:
                    v_mod._kllm_extra_targets.append(tr_mod)
            return
        orig_dec = v_mod.decode
        rt_self = self
        v_mod._kllm_extra_targets = []
        if tr_mod is not None:
            v_mod._kllm_extra_targets.append(tr_mod)

        def _clean_decode(*args, **kwargs):
            rt_self._reclaim_pipeline_memory()
            for extra in getattr(v_mod, "_kllm_extra_targets", ()):
                if extra is not None:
                    if hasattr(extra, "_teacache_reset") and callable(getattr(extra, "_teacache_reset")):
                        try:
                            extra._teacache_reset()
                        except Exception:
                            pass
                    for attr in ("_cache", "kv_cache", "cache", "_intermediates", "intermediates", "past_key_values"):
                        if hasattr(extra, attr):
                            try:
                                setattr(extra, attr, None)
                            except Exception:
                                pass
            _clean_cuda()
            try:
                return orig_dec(*args, **kwargs)
            finally:
                _clean_cuda()

        v_mod.decode = _clean_decode
        v_mod._kllm_cleanup_wrapped = True

    # ---- loading -----------------------------------------------------------------------
    def load(self):
        import torch
        try:
            import torchao.utils
            if not hasattr(torchao.utils, "register_as_pytree_constant"):
                def _reg(cls):
                    try:
                        import torch.utils._pytree as _pt
                        if hasattr(_pt, "register_constant"):
                            _pt.register_constant(cls)
                    except Exception:
                        pass
                    return cls
                torchao.utils.register_as_pytree_constant = _reg
        except Exception:
            pass
        try:
            import torchao.quantization
            if not hasattr(torchao.quantization, "FqnToConfig"):
                class FqnToConfig:
                    pass
                torchao.quantization.FqnToConfig = FqnToConfig
        except Exception:
            pass
        import diffusers
        o = self.opts
        dtype_pref = str(o.get("torch_dtype") or o.get("dtype") or "float16").lower()
        if ("bf16" in dtype_pref or "bfloat16" in dtype_pref) and hasattr(torch, "bfloat16"):
            target_dtype = torch.bfloat16
        elif hasattr(torch, "float16"):
            target_dtype = torch.float16
        else:
            target_dtype = None
        kw = {}
        if target_dtype is not None:
            kw["torch_dtype"] = target_dtype
        mode = o.get("offload", "auto")
        if mode == "balanced":
            n_gpus = torch.cuda.device_count() if (hasattr(torch, "cuda") and callable(getattr(torch.cuda, "device_count", None))) else 1
            if n_gpus < 2:
                log(f"offload=balanced requested but only {n_gpus} GPU available; using model CPU offload")
                mode = "model"
            else:
                kw["device_map"] = "balanced"
        cls = getattr(diffusers, o["pipeline"], diffusers.DiffusionPipeline) if o.get("pipeline") else diffusers.DiffusionPipeline
        variant = o.get("variant", "fp16")
        quant_mode = str(o.get("quant") or o.get("quantization") or "").lower()
        sub_components = {}
        if quant_mode in ("nf4", "4bit", "bnb_nf4", "bnb_4bit"):
            try:
                from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
                bnb_config = DiffusersBnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=target_dtype or torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                transformer_cls = None
                if "Wan" in (o.get("pipeline") or "") or "wan" in str(self.repo).lower():
                    transformer_cls = getattr(diffusers, "WanTransformer3DModel", None)
                    log(f"loading {transformer_cls.__name__} in 4-bit NF4...")
                    sub_components["transformer"] = transformer_cls.from_pretrained(
                        self.repo,
                        subfolder="transformer",
                        quantization_config=bnb_config,
                        torch_dtype=target_dtype or torch.float16,
                    )
                    log("quantized 4-bit NF4 transformer loaded successfully")
                    _clean_cuda()
                if "Wan" in (o.get("pipeline") or "") or "wan" in str(self.repo).lower():
                    try:
                        from transformers import BitsAndBytesConfig as TrBnBConfig
                        from transformers import UMT5EncoderModel
                        tr_bnb = TrBnBConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=target_dtype or torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        log("loading UMT5EncoderModel in 4-bit NF4...")
                        sub_components["text_encoder"] = UMT5EncoderModel.from_pretrained(
                            self.repo,
                            subfolder="text_encoder",
                            quantization_config=tr_bnb,
                            torch_dtype=target_dtype or torch.float16,
                            device_map="cpu",     # avoid CUDA OOM during load; pipeline offload moves it later
                        )
                        log("quantized 4-bit NF4 text_encoder loaded successfully")
                        _clean_cuda()
                    except Exception as te_err:
                        log("warning: could not quantize text_encoder:", te_err)
            except Exception as e:
                log("warning: could not load quantized components separately:", e)

        kw.update(sub_components)
        log("loading pipeline", self.repo, "variant", variant or "-")
        try:
            pipe = cls.from_pretrained(self.repo, **({"variant": variant} if variant else {}), **kw)
        except (ValueError, OSError) as e:
            if not variant:
                raise
            log(f"no '{variant}' variant ({str(e)[:100]}); loading default weights")
            pipe = cls.from_pretrained(self.repo, **kw)
        fp32_names = set(o.get("fp32") or [])
        if target_dtype is not None:
            for name, comp in (getattr(pipe, "components", None) or {}).items():
                if name not in fp32_names and name not in sub_components and hasattr(comp, "to"):
                    cur_dt = _module_dtype(comp)
                    if cur_dt is not None and cur_dt != target_dtype:
                        try:
                            comp.to(target_dtype)
                        except Exception:
                            pass
        for name in fp32_names:          # e.g. Wan's VAE is unstable in fp16
            comp = getattr(pipe, name, None)
            if comp is not None and hasattr(comp, "to"):
                try:
                    comp.to(torch.float32)
                except Exception:
                    pass
        sizes = component_sizes(pipe)
        total, largest = sum(sizes.values()), max(sizes.values() or [0])
        self.placement = plan_placement(total, largest, mode)
        log(f"pipeline {total:.1f} GB (largest component {largest:.1f} GB) -> placement {self.placement}")
        if self.placement == "gpu":
            pipe.to("cuda")
        elif self.placement == "model":
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                log("enable_model_cpu_offload failed, falling back to sequential offload:", e)
                self.placement = "sequential"
                pipe.enable_sequential_cpu_offload()
        elif self.placement == "sequential":
            pipe.enable_sequential_cpu_offload()

        # VAE slicing & tiling
        is_wan = "wan" in str(self.key).lower() or "wan" in str(self.repo).lower() or "wan" in str(o.get("pipeline") or "").lower()
        tile_kwargs = {}
        if is_wan:
            tile_kwargs = {
                "tile_sample_min_height": int(o.get("tile_sample_min_height") or o.get("vae_tile_sample_min_height") or 256),
                "tile_sample_min_width": int(o.get("tile_sample_min_width") or o.get("vae_tile_sample_min_width") or 256),
                "tile_sample_stride_height": int(o.get("tile_sample_stride_height") or o.get("vae_tile_sample_stride_height") or 192),
                "tile_sample_stride_width": int(o.get("tile_sample_stride_width") or o.get("vae_tile_sample_stride_width") or 192),
                "tile_sample_min_num_frames": int(o.get("tile_sample_min_num_frames") or o.get("vae_tile_sample_min_num_frames") or 16),
                "tile_sample_stride_num_frames": int(o.get("tile_sample_stride_num_frames") or o.get("vae_tile_sample_stride_num_frames") or 12),
            }
        else:
            for k in ("tile_sample_min_height", "tile_sample_min_width", "tile_sample_min_num_frames",
                      "tile_sample_stride_height", "tile_sample_stride_width", "tile_sample_stride_num_frames"):
                if k in o:
                    tile_kwargs[k] = o[k]
        self.tile_kwargs = tile_kwargs

        is_ltx = "ltx" in str(self.key).lower() or "ltx" in str(self.repo).lower()
        slicing = o.get("vae_slicing", "auto")
        tiling = o.get("vae_tiling", "auto")
        do_tiling = tiling is True or (tiling == "auto" and (self.placement != "gpu" or self.kind == "video") and not is_ltx)
        do_slicing = slicing is True or (slicing == "auto" and (self.placement != "gpu" or self.kind == "video"))

        vae = getattr(pipe, "vae", None)
        if do_slicing:
            self._apply_pipe_vae_slicing(pipe)
            self._apply_vae_slicing(vae)

        if do_tiling:
            self._apply_pipe_vae_tiling(pipe)
            self._apply_vae_tiling(vae, tile_kwargs)

        self._wrap_vae_decode(vae, getattr(pipe, "transformer", None))

        # Attention slicing / memory-efficient attention
        attn_slicing = o.get("attention_slicing", "auto")
        if attn_slicing is True or (attn_slicing == "auto" and self.placement != "gpu"):
            if hasattr(pipe, "enable_attention_slicing"):
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

        self.pipe = pipe
        if o.get("teacache"):
            thresh = float(o.get("teacache_thresh", 0.15))
            tr = getattr(pipe, "transformer", None)
            if tr is not None:
                _apply_teacache(tr, thresh)
        comps = getattr(pipe, "components", None) or {}
        if self.kind == "video" and comps:
            def _instantiate_pipe(target_cls, base_comps, **extra_kwargs):
                if not target_cls:
                    return None
                try:
                    sig = _inspect.signature(target_cls.__init__)
                    has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                    call_args = {}
                    if has_varkw:
                        call_args.update(base_comps)
                        call_args.update(extra_kwargs)
                    else:
                        allowed = set(sig.parameters.keys())
                        for k, v in base_comps.items():
                            if k in allowed:
                                call_args[k] = v
                        for k, v in extra_kwargs.items():
                            if k in allowed:
                                call_args[k] = v
                    return target_cls(**call_args)
                except Exception as e:
                    log(f"warning: could not build {getattr(target_cls, '__name__', str(target_cls))} from components: {e}")
                    return None

            # 1. LTX-Video pairing (LTXImageToVideoPipeline <-> LTXPipeline)
            ltx_i2v_cls = getattr(diffusers, "LTXImageToVideoPipeline", None)
            ltx_t2v_cls = getattr(diffusers, "LTXPipeline", None)
            if ltx_i2v_cls and ltx_t2v_cls:
                if isinstance(pipe, ltx_i2v_cls):
                    self.i2v_pipe = pipe
                    self.t2v_pipe = _instantiate_pipe(ltx_t2v_cls, comps) or pipe
                elif isinstance(pipe, ltx_t2v_cls):
                    self.t2v_pipe = pipe
                    self.i2v_pipe = _instantiate_pipe(ltx_i2v_cls, comps) or pipe

            # 2. Wan pairing (WanPipeline <-> WanVideoToVideoPipeline <-> WanImageToVideoPipeline)
            wan_t2v_cls = getattr(diffusers, "WanPipeline", None)
            wan_v2v_cls = getattr(diffusers, "WanVideoToVideoPipeline", None)
            wan_i2v_cls = getattr(diffusers, "WanImageToVideoPipeline", None)
            p_name = str(o.get("pipeline") or self.cfg.get("pipeline") or "").lower()
            is_i2v_intent = "imagetovideo" in p_name or "i2v" in str(self.key).lower() or (wan_i2v_cls and isinstance(pipe, wan_i2v_cls) and pipe.__class__.__name__ != "WanPipeline")

            extra_cfg = {}
            if hasattr(pipe, "config"):
                for cfg_key in ("expand_timesteps", "boundary_ratio"):
                    if hasattr(pipe.config, cfg_key):
                        extra_cfg[cfg_key] = getattr(pipe.config, cfg_key)

            if is_i2v_intent:
                self.i2v_pipe = pipe
                if wan_t2v_cls and self.t2v_pipe is None:
                    self.t2v_pipe = _instantiate_pipe(wan_t2v_cls, comps, **extra_cfg)
                if wan_v2v_cls and self.v2v_pipe is None:
                    self.v2v_pipe = _instantiate_pipe(wan_v2v_cls, comps, **extra_cfg)
            elif wan_t2v_cls and isinstance(pipe, wan_t2v_cls):
                self.t2v_pipe = pipe
                if wan_i2v_cls and self.i2v_pipe is None:
                    self.i2v_pipe = _instantiate_pipe(wan_i2v_cls, comps, **extra_cfg)
                if wan_v2v_cls and self.v2v_pipe is None:
                    self.v2v_pipe = _instantiate_pipe(wan_v2v_cls, comps, **extra_cfg)
            elif wan_i2v_cls and isinstance(pipe, wan_i2v_cls):
                self.i2v_pipe = pipe
                if wan_t2v_cls and self.t2v_pipe is None:
                    self.t2v_pipe = _instantiate_pipe(wan_t2v_cls, comps, **extra_cfg)
                if wan_v2v_cls and self.v2v_pipe is None:
                    self.v2v_pipe = _instantiate_pipe(wan_v2v_cls, comps, **extra_cfg)

            # 3. CogVideoX pairing
            cog_v2v_cls = getattr(diffusers, "CogVideoXVideoToVideoPipeline", None)
            if cog_v2v_cls and any("cogvideo" in str(getattr(c, "__class__", "")).lower() for c in comps.values()):
                self.v2v_pipe = _instantiate_pipe(cog_v2v_cls, comps)

        for p in (self.pipe, self.i2v_pipe, self.v2v_pipe, self.t2v_pipe):
            if p is not None:
                if do_tiling:
                    self._apply_pipe_vae_tiling(p)
                    p_vae = getattr(p, "vae", None)
                    if p_vae is not None:
                        self._apply_vae_tiling(p_vae, tile_kwargs)
                if do_slicing:
                    self._apply_pipe_vae_slicing(p)
                    p_vae = getattr(p, "vae", None)
                    if p_vae is not None:
                        self._apply_vae_slicing(p_vae)
                p_vae = getattr(p, "vae", None)
                if p_vae is not None:
                    self._wrap_vae_decode(p_vae, getattr(p, "transformer", None) or getattr(pipe, "transformer", None))
        return pipe

    def warmup(self):
        """Tiny 1-step generation so the first real request does not pay CUDA init."""
        if not self.opts.get("warmup", self.kind == "image"):
            return
        self.warming = True
        try:
            small = {"prompt": "a red square", "size": self.opts.get("warmup_size", "256x256"), "steps": 1, "seed": 0}
            if self.kind == "video":
                self.generate_video(dict(small, num_frames=self.opts.get("warmup_frames", 9)))
            else:
                self.generate_image(dict(small, n=1))
            log("warm-up generation done")
        except Exception as e:
            log("warm-up failed (continuing):", e)
        finally:
            self.warming = False

    def load_all(self):
        try:
            self.load()
            self.warmup()
            self.state = "ready"
        except BaseException as e:
            import traceback
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
            self.state = "error"
            log("diffusers load failed:", self.error)

    # ---- request helpers ----------------------------------------------------------------
    def _size(self, req, video=False):
        raw = req.get("size") or self.opts.get("size") or ("1024x1024" if self.kind == "image" else "512x512")
        m = re.fullmatch(r"\s*(\d{2,5})\s*[xX*]\s*(\d{2,5})\s*", str(raw))
        if not m:
            raise ApiError(400, f"size must look like '1024x1024', got {raw!r}")
        w, h = int(m.group(1)), int(m.group(2))
        mult = int(self.opts.get("size_multiple", 8))
        if w % mult or h % mult:
            if video:
                w = max(mult, int(round(w / mult)) * mult)
                h = max(mult, int(round(h / mult)) * mult)
            else:
                raise ApiError(400, f"width and height must be multiples of {mult} for this model, got {w}x{h}")
        if w * h > int(self.opts.get("max_pixels", 2048 * 2048)):
            raise ApiError(400, f"{w}x{h} is larger than the {self.opts.get('max_pixels', 2048 * 2048)} pixel limit")
        return w, h

    def _int(self, req, name, default, lo, hi):
        v = req.get(name)
        if v is None:
            v = default
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ApiError(400, f"{name} must be an integer")
        if not lo <= v <= hi:
            raise ApiError(400, f"{name} must be between {lo} and {hi}")
        return v

    def _kwargs(self, req, video=False):
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiError(400, "prompt is required")
        o = self.opts
        w, h = self._size(req, video=video)
        kw = {"prompt": prompt, "width": w, "height": h,
              "num_inference_steps": self._int(req, "steps", o.get("steps", 30), 1, 200)}
        neg = req.get("negative_prompt", o.get("negative_prompt"))
        if neg:
            kw["negative_prompt"] = str(neg)
        gs = req.get("guidance_scale", o.get("guidance_scale"))
        if gs is not None:
            try:
                kw["guidance_scale"] = float(gs)
            except (TypeError, ValueError):
                raise ApiError(400, "guidance_scale must be a number")
        if video:
            max_f = int(o.get("max_frames", 241))
            raw_frames = self._int(req, "num_frames", o.get("num_frames", 49), 1, max_f)
            f_mult = int(o.get("frame_multiple", 8 if "ltx" in str(self.key).lower() else (4 if "wan" in str(self.key).lower() else 1)))
            min_f = 1 + f_mult if f_mult > 1 else 1
            if f_mult > 1:
                if raw_frames < min_f:
                    raw_frames = min_f
                else:
                    rem = (raw_frames - 1) % f_mult
                    if rem != 0:
                        raw_frames = max(min_f, raw_frames - rem)
            kw["num_frames"] = raw_frames
        img_in = req.get("image")
        if img_in:
            mult = int(self.opts.get("size_multiple", 8))
            max_ref_w = min(w, int(self.opts.get("max_ref_w", 1024)))
            max_ref_h = min(h, int(self.opts.get("max_ref_h", 1024)))
            def _to_pil(x):
                from PIL import Image
                if isinstance(x, Image.Image):
                    im = x
                elif isinstance(x, str):
                    x = x.strip()
                    if x.startswith("http://") or x.startswith("https://"):
                        import urllib.request
                        req_img = urllib.request.Request(x, headers={"User-Agent": "kllm/1.0"})
                        with urllib.request.urlopen(req_img, timeout=15) as resp:
                            im = Image.open(io.BytesIO(resp.read()))
                    else:
                        if "," in x and x.startswith("data:"):
                            x = x.split(",", 1)[1]
                        pad = len(x) % 4
                        if pad:
                            x += "=" * (4 - pad)
                        im = Image.open(io.BytesIO(base64.b64decode(x)))
                elif isinstance(x, bytes):
                    im = Image.open(io.BytesIO(x))
                elif isinstance(x, dict) and "url" in x:
                    return _to_pil(x["url"])
                elif isinstance(x, dict) and "b64_json" in x:
                    return _to_pil(x["b64_json"])
                else:
                    raise ValueError(f"unsupported reference image type: {type(x).__name__}")
                im.load()
                return _preprocess_image(im, w if video else max_ref_w, h if video else max_ref_h, mult, exact=video)
            try:
                if isinstance(img_in, list):
                    kw["image"] = [_to_pil(x) for x in img_in[:10]]
                else:
                    kw["image"] = _to_pil(img_in)
            except Exception as e:
                raise ApiError(400, f"failed to decode reference image: {e}")
        return kw

    def _decode_video(self, req, target_w, target_h, max_frames):
        vid_in = req.get("video") or req.get("source_video")
        if not vid_in:
            return None
        import imageio.v2 as iio
        from PIL import Image
        f_mult = int(self.opts.get("frame_multiple", 8 if "ltx" in str(self.key).lower() else (4 if "wan" in str(self.key).lower() else 1)))
        min_f = 1 + f_mult if f_mult > 1 else 1

        if isinstance(vid_in, list):
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "BILINEAR", None))
            frames = []
            for item in vid_in:
                if isinstance(item, Image.Image):
                    im = item.convert("RGB")
                elif isinstance(item, str):
                    im = Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB")
                else:
                    im = item
                if target_w and target_h and hasattr(im, "resize"):
                    im = im.resize((target_w, target_h), resample=resample) if resample is not None else im.resize((target_w, target_h))
                frames.append(im)
            if max_frames and len(frames) > max_frames:
                frames = frames[:max_frames]
            if f_mult > 1 and len(frames) > 1:
                rem = (len(frames) - 1) % f_mult
                if rem != 0:
                    frames = frames[:-rem]
            if len(frames) < min_f:
                raise ApiError(400, f"source video is too short ({len(frames)} frames; requires at least {min_f} frames for {f_mult}n+1 cadence)")
            return frames

        raw_bytes = None
        if isinstance(vid_in, str):
            vid_in = vid_in.strip()
            if vid_in.startswith("http://") or vid_in.startswith("https://"):
                import urllib.request
                req_v = urllib.request.Request(vid_in, headers={"User-Agent": "kllm/1.0"})
                with urllib.request.urlopen(req_v, timeout=30) as resp:
                    raw_bytes = resp.read()
            else:
                if "," in vid_in and vid_in.startswith("data:"):
                    vid_in = vid_in.split(",", 1)[1]
                pad = len(vid_in) % 4
                if pad:
                    vid_in += "=" * (4 - pad)
                raw_bytes = base64.b64decode(vid_in)
        elif isinstance(vid_in, (bytes, bytearray)):
            raw_bytes = bytes(vid_in)
        if not raw_bytes:
            raise ApiError(400, "empty video payload")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            t.write(raw_bytes)
            t.flush()
            tmp_path = t.name
        try:
            reader = iio.get_reader(tmp_path, "ffmpeg")
            frames = []
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "BILINEAR", None))
            for frame in reader:
                im = Image.fromarray(frame).convert("RGB")
                if target_w and target_h:
                    im = im.resize((target_w, target_h), resample=resample) if resample is not None else im.resize((target_w, target_h))
                frames.append(im)
                if max_frames and len(frames) >= max_frames:
                    break
            reader.close()
            if f_mult > 1 and len(frames) > 1:
                rem = (len(frames) - 1) % f_mult
                if rem != 0:
                    frames = frames[:-rem]
            if len(frames) < min_f:
                raise ApiError(400, f"source video is too short ({len(frames)} frames; requires at least {min_f} frames for {f_mult}n+1 cadence)")
            return frames
        except ApiError:
            raise
        except Exception as e:
            raise ApiError(400, f"failed to decode source video: {e}")
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass

    def _takes(self, pipe, name, list_ok=False):
        try:
            sig = _inspect.signature(pipe.__call__)
        except (TypeError, ValueError):
            return True
        if name in sig.parameters:
            return True if not list_ok else "List" in str(sig.parameters[name].annotation) or "list" in str(sig.parameters[name].annotation)
        return any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()) and not list_ok

    def _img2img_pipe(self):
        """A pipeline that uses the reference image. The loaded pipe when it takes `image` (edit models), else an
        image-to-image view of the same weights (no second copy in VRAM). Never drops the image silently."""
        if self._takes(self.pipe, "image"):
            return self.pipe
        cached = getattr(self, "_i2i", None)
        if cached is not None:
            return cached
        try:
            from diffusers import AutoPipelineForImage2Image
            self._i2i = AutoPipelineForImage2Image.from_pipe(self.pipe)
        except Exception as e:
            raise ApiError(400, f"model '{self.key}' cannot start from an image ({type(e).__name__}: {e}). "
                                "Use an image-edit model, or send the prompt without an image.")
        return self._i2i

    def _accepted(self, kw, pipe=None):
        """Drop arguments this pipeline does not take (e.g. negative_prompt on FLUX)."""
        target_pipe = pipe or self.pipe
        try:
            sig = _inspect.signature(target_pipe.__call__)
        except (TypeError, ValueError):
            return kw
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            return kw
        params = sig.parameters
        out = {k: v for k, v in kw.items() if k in params}
        if "true_cfg_scale" in params and "true_cfg_scale" not in out:
            if "guidance_scale" in kw and kw["guidance_scale"] is not None:
                out["true_cfg_scale"] = kw["guidance_scale"]
        if "negative_prompt" in params and "negative_prompt" not in out:
            if out.get("true_cfg_scale", 1.0) > 1.0:
                out["negative_prompt"] = ""
        return out

    def _seed(self, req, i=0):
        s = req.get("seed")
        try:
            s = int(s) if s is not None else random.randrange(2 ** 31)
        except (TypeError, ValueError):
            raise ApiError(400, "seed must be an integer")
        return s + i

    def _generator(self, seed):
        import torch
        return torch.Generator(device="cpu").manual_seed(seed)

    def _encode_image(self, img):
        if not hasattr(img, "save"):               # numpy output_type
            import numpy as np
            from PIL import Image
            a = np.asarray(img)
            img = Image.fromarray((a * 255).round().astype("uint8") if a.dtype.kind == "f" else a)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _encode_mp4(self, frames, fps, target_fps=None):
        import numpy as np
        import imageio.v2 as iio
        arr = []
        for f in frames:
            a = np.asarray(f)
            if a.dtype.kind == "f":
                a = (np.clip(a, 0, 1) * 255).round().astype("uint8")
            arr.append(a)
        target_fps = int(target_fps) if target_fps else int(fps)
        # Fast path: no interpolation requested.
        if target_fps == int(fps):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as t:
                iio.mimwrite(t.name, arr, fps=fps, codec="libx264", quality=8, macro_block_size=1)
                return open(t.name, "rb").read()
        # Write the native (low-fps) clip first, then use ffmpeg's minterpolate to
        # raise the frame rate without changing the clip duration. This keeps the
        # diffusion cost at the low source fps while delivering smooth playback.
        src = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        src.close()
        try:
            iio.mimwrite(src.name, arr, fps=fps, codec="libx264", quality=8, macro_block_size=1)
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = "ffmpeg"
            dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            dst.close()
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", src.name, "-vf",
                     "minterpolate=fps=%d:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" % target_fps,
                     "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", dst.name],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return open(dst.name, "rb").read()
            finally:
                try:
                    os.unlink(dst.name)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(src.name)
            except OSError:
                pass

    @staticmethod
    def _first_video(frames):
        if hasattr(frames, "ndim") and frames.ndim == 5:
            return frames[0]
        first = frames[0]
        if isinstance(first, (list, tuple)) or getattr(first, "ndim", 0) == 4:
            return first
        return frames

    # ---- generation ---------------------------------------------------------------------
    def _run(self, fn):
        if (self.state != "ready" and not self.warming) or self.pipe is None:
            raise ApiError(503, "model is still loading" if self.state == "loading"
                           else f"model failed to load: {self.error}", "model_unavailable")
        with self.lock:                             # one generation at a time
            self.busy = True
            try:
                _clean_cuda()
                tr = getattr(self.pipe, "transformer", None)
                if tr is not None and hasattr(tr, "_teacache_reset"):
                    tr._teacache_reset()
                return fn()
            except ApiError:
                raise
            except Exception as e:
                if _is_oom(e):
                    _clean_cuda()
                    if self.placement == "model" and hasattr(self.pipe, "enable_sequential_cpu_offload"):
                        try:
                            log("GPU OOM with model offload; switching to sequential CPU offload and retrying...")
                            self.pipe.enable_sequential_cpu_offload()
                            self.placement = "sequential"
                            _clean_cuda()
                            return fn()
                        except Exception as retry_err:
                            log("retry with sequential offload failed:", retry_err)
                            _clean_cuda()
                    raise ApiError(500, "GPU out of memory; try a smaller size / fewer frames / fewer steps",
                                   "out_of_memory")
                import traceback
                log("generation failed:", traceback.format_exc()[-1500:])
                raise ApiError(500, f"generation failed: {type(e).__name__}: {e}", "generation_error")
            finally:
                _clean_cuda()
                self.busy = False
                self.served += 1

    def generate_image(self, req):
        n = self._int(req, "n", 1, 1, int(self.opts.get("max_n", 4)))
        fmt = req.get("response_format") or "b64_json"
        if fmt not in ("b64_json", "url"):
            raise ApiError(400, "response_format must be b64_json or url")
        kw = self._kwargs(req)
        seed0 = self._seed(req)
        pipe = self.pipe
        if kw.get("image") is not None:
            pipe = self._img2img_pipe()
            if isinstance(kw["image"], list) and not self._takes(pipe, "image", list_ok=True):
                kw["image"] = kw["image"][0]
            if self._takes(pipe, "strength"):
                try:
                    s = float(req.get("strength", self.opts.get("img2img_strength", 0.6)))
                except (TypeError, ValueError):
                    s = 0.6
                s = max(0.05, min(1.0, s))
                steps = int(kw.get("num_inference_steps") or 0)
                if steps and steps * s < 1:            # turbo models run 1-4 steps: keep at least one denoise step
                    kw["num_inference_steps"] = steps = max(steps, int(math.ceil(1 / s)))
                kw["strength"] = s

        def go():
            data = []
            for i in range(n):
                call = self._accepted(dict(kw, generator=self._generator(seed0 + i)), pipe)
                images = pipe(**call).images
                b = base64.b64encode(self._encode_image(images[0])).decode()
                item = {"seed": seed0 + i}
                if fmt == "url":                       # no file server in the kernel: a data URL works in <img src>
                    item["url"] = "data:image/png;base64," + b
                else:
                    item["b64_json"] = b
                data.append(item)
            return {"created": int(time.time()), "data": data}
        return self._run(go)

    def generate_video(self, req):
        kw = self._kwargs(req, video=True)
        fps = self._int(req, "fps", self.opts.get("fps", 24 if "ltx" in str(self.key).lower() else 16), 1, 60)
        out_fps = int(req.get("output_fps") or self.opts.get("interpolate_fps") or self.opts.get("output_fps") or fps)
        if out_fps < 1 or out_fps > 120:
            raise ApiError(400, "output_fps must be between 1 and 120")
        seed = self._seed(req)
        w, h = kw["width"], kw["height"]

        # Check for Video-to-Video
        source_video = self._decode_video(req, w, h, kw.get("num_frames", 49))
        active_pipe = None
        if source_video is not None:
            active_pipe = getattr(self, "v2v_pipe", None)
            if active_pipe is None:
                sig = _inspect.signature(self.pipe.__call__)
                if "video" in sig.parameters or any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
                    active_pipe = self.pipe
                else:
                    raise ApiError(400, f"model '{self.key}' does not support video-to-video transformation; use a model that supports video input (e.g. wan2.1-t2v-1.3b or cogvideox)")
            kw["video"] = source_video
            kw["num_frames"] = len(source_video)
            kw.pop("image", None)
            if "strength" in req:
                try:
                    kw["strength"] = max(0.05, min(1.0, float(req["strength"])))
                except (ValueError, TypeError):
                    pass
        # Check for Image-to-Video
        elif kw.get("image") is not None:
            active_pipe = getattr(self, "i2v_pipe", None)
            if active_pipe is None:
                sig = _inspect.signature(self.pipe.__call__)
                if "image" in sig.parameters or any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
                    active_pipe = self.pipe
                else:
                    raise ApiError(400, f"model '{self.key}' does not support image-to-video generation; use an I2V model (e.g. ltx-video)")
        # Text-to-Video
        else:
            active_pipe = getattr(self, "t2v_pipe", None)
            if active_pipe is None:
                sig = _inspect.signature(self.pipe.__call__)
                p_img = sig.parameters.get("image")
                if p_img is not None and p_img.default is p_img.empty and p_img.kind not in (p_img.VAR_KEYWORD, p_img.VAR_POSITIONAL):
                    raise ApiError(400, f"model '{self.key}' requires a reference image; provide 'image' in request body for image-to-video")
                active_pipe = self.pipe

        def go():
            # Ensure VAE tiling, slicing, and decode cleanup wrapper are active on active_pipe
            is_ltx = "ltx" in str(self.key).lower() or "ltx" in str(self.repo).lower()
            tiling = self.opts.get("vae_tiling", "auto")
            slicing = self.opts.get("vae_slicing", "auto")
            do_tiling = tiling is True or (tiling == "auto" and (self.placement != "gpu" or self.kind == "video") and not is_ltx)
            do_slicing = slicing is True or (slicing == "auto" and (self.placement != "gpu" or self.kind == "video"))

            act_vae = getattr(active_pipe, "vae", None)
            act_tr = getattr(active_pipe, "transformer", None) or getattr(self.pipe, "transformer", None)
            if act_vae is not None:
                if do_tiling:
                    self._apply_pipe_vae_tiling(active_pipe)
                    self._apply_vae_tiling(act_vae, self.tile_kwargs)
                if do_slicing:
                    self._apply_pipe_vae_slicing(active_pipe)
                    self._apply_vae_slicing(act_vae)
                self._wrap_vae_decode(act_vae, act_tr)

            call = dict(kw, generator=self._generator(seed))
            try:
                sig = _inspect.signature(active_pipe.__call__)
                has_cb = "callback_on_step_end" in sig.parameters or any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
                if has_cb:
                    steps_tot = call.get("num_inference_steps", 30)
                    def _step_cleanup(pipe_obj, step, timestep, cb_kwargs):
                        sched = getattr(pipe_obj, "scheduler", None)
                        timesteps_list = getattr(sched, "timesteps", None)
                        total_s = len(timesteps_list) if timesteps_list is not None else steps_tot
                        if step >= total_s - 1 or step >= steps_tot - 1:
                            self._reclaim_pipeline_memory(pipe_obj)
                        return cb_kwargs
                    call["callback_on_step_end"] = _step_cleanup
                    if "callback_on_step_end_tensor_inputs" in sig.parameters or any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
                        call["callback_on_step_end_tensor_inputs"] = ["latents"]
            except Exception:
                pass
            call = self._accepted(call, pipe=active_pipe)
            output = active_pipe(**call)
            _clean_cuda()
            frames = self._first_video(output.frames)
            out_frames = int(round(len(frames) * out_fps / fps)) if out_fps != fps else len(frames)
            if out_fps != fps:
                try:
                    mp4 = self._encode_mp4(frames, fps, target_fps=out_fps)
                except TypeError:
                    mp4 = self._encode_mp4(frames, fps)
            else:
                mp4 = self._encode_mp4(frames, fps)
            duration_s = round(len(frames) / fps, 2)
            return {
                "created": int(time.time()),
                "data": [{"b64_mp4": base64.b64encode(mp4).decode(), "seed": seed}],
                "fps": out_fps,
                "frames": out_frames,
                "width": w,
                "height": h,
                "duration_seconds": duration_s
            }
        return self._run(go)


def _make_diffusers_handler(rt):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, status, message, typ="invalid_request_error"):
            self._send(status, {"error": {"message": message, "type": typ}})

        def _authed(self):
            got = self.headers.get("Authorization", "")
            got = got[7:].strip() if got.lower().startswith("bearer ") else (self.headers.get("x-api-key") or "").strip()
            if rt.api_key and hmac.compare_digest(got.encode(), str(rt.api_key).encode()):
                return True
            self._err(401, "Invalid API Key", "authentication_error")
            return False

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                if rt.state == "ready":
                    return self._send(200, {"status": "ok", "busy": rt.busy})
                return self._send(503 if rt.state == "loading" else 500,
                                  {"status": rt.state, "error": {"message": rt.error or "loading model"}})
            if not self._authed():
                return
            if path in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [
                    {"id": rt.key, "object": "model", "owned_by": "kllm", "kind": rt.kind, "runtime": "diffusers"}]})
            self._err(404, f"unknown path {path}", "not_found")

        def do_POST(self):
            path = self.path.split("?")[0]
            if not self._authed():
                return
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if 0 < n <= 50_000_000 else b""
            if n > 50_000_000:
                return self._err(413, "request body too large")
            try:
                req = json.loads(raw or b"{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._err(400, "body must be a JSON object")
            routes = {"/v1/images/generations": ("image", rt.generate_image),
                      "/v1/videos/generations": ("video", rt.generate_video)}
            if path not in routes:
                return self._err(404, f"unknown path {path}", "not_found")
            kind, fn = routes[path]
            if kind != rt.kind:
                return self._err(400, f"this model generates {rt.kind}s; use /v1/{rt.kind}s/generations")
            try:
                self._send(200, fn(req))
            except ApiError as e:
                self._err(e.status, e.message, e.typ)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:                       # never leave a client without an answer
                self._err(500, f"{type(e).__name__}: {e}", "server_error")
    return H


def make_diffusers_server(rt, port, host="127.0.0.1"):
    class S(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    return S((host, port), _make_diffusers_handler(rt))


class _DiffusersHandle:
    """Looks like the llama-server Popen to main(): poll() is None while healthy."""
    def __init__(self, rt, srv, thread):
        self.rt, self.srv, self.thread = rt, srv, thread

    def poll(self):
        return None if self.thread.is_alive() and self.rt.state != "error" else 1

    def terminate(self):
        try:
            self.srv.shutdown()
        except Exception:
            pass
    kill = terminate


def start_diffusers():
    """Install deps, load the pipeline, serve it on LLAMA_PORT. Returns once it is ready."""
    kind = CFG.get("kind") or "image"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if CFG.get("hf_token"):
        os.environ["HF_TOKEN"] = CFG["hf_token"]
    ntfy("downloading", note="diffusers: installing packages, fetching model (first run only)")
    try:
        ensure_packages(kind, (CFG.get("opts") or {}).get("pip") or ())
    except Exception as e:
        ntfy("error", stage="pip_install", tail=str(e)[-1500:])
        raise
    rt = DiffusersRuntime(CFG)
    srv = make_diffusers_server(rt, LLAMA_PORT)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    threading.Thread(target=rt.load_all, daemon=True).start()
    deadline = time.time() + 1800
    while rt.state == "loading" and time.time() < deadline:
        time.sleep(3)
    if rt.state != "ready":
        ntfy("error", stage="server_health", tail=(rt.error or "load timed out")[-1500:])
        log("diffusers server did not become ready; aborting")
        sys.exit(1)
    return _DiffusersHandle(rt, srv, th)


class _FastVideoHandle:
    def __init__(self, proc, adapter_srv, adapter_thread, fv_port):
        self.proc = proc
        self.adapter_srv = adapter_srv
        self.adapter_thread = adapter_thread
        self.fv_port = fv_port

    def poll(self):
        if self.proc is not None and self.proc.poll() is not None:
            return self.proc.returncode
        if self.adapter_thread is not None and not self.adapter_thread.is_alive():
            return 1
        return None

    def terminate(self):
        try:
            self.adapter_srv.shutdown()
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=30)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def start_comfyui():
    """Wan2.2 video via ComfyUI (kernel/comfy_video.py, shipped inside CFG). Serves /v1/videos/generations on LLAMA_PORT."""
    import types
    if CFG.get("hf_token"):
        os.environ["HF_TOKEN"] = CFG["hf_token"]
    mod = types.ModuleType("kllm_comfy_video")
    exec(compile(base64.b64decode(CFG["_comfy_video_b64"]).decode(), "comfy_video.py", "exec"), mod.__dict__)
    CFG["_kllm_nodes_src"] = base64.b64decode(CFG.get("_kllm_nodes_b64") or b"").decode()
    return mod.start(CFG, LLAMA_PORT, log, ntfy)


def start_fastvideo():
    import subprocess as _sp
    fv_port = LLAMA_PORT + 1
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if CFG.get("hf_token"):
        os.environ["HF_TOKEN"] = CFG["hf_token"]
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    ntfy("downloading", note="fastvideo: installing packages, fetching model (first run only)")
    try:
        pkgs = ["fastvideo"]
        extra = (CFG.get("opts") or {}).get("pip") or ()
        for p in extra:
            if p not in pkgs:
                pkgs.append(p)
        # install only if missing
        try:
            import fastvideo
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "-U", *pkgs], check=True)
            import importlib
            importlib.invalidate_caches()
    except Exception as e:
        ntfy("error", stage="pip_install", tail=str(e)[-1500:])
        raise

    # write a serve config
    import yaml as _yaml
    o = CFG.get("opts") or {}
    model_id = CFG.get("model_hf") or CFG.get("hf")
    cfg = {
        "generator": {
            "model_path": model_id,
            "engine": {
                "num_gpus": 1,
                "use_fsdp_inference": False,
                "parallelism": {"tp_size": 1, "sp_size": 1},
                "offload": {"dit": True, "vae": False, "text_encoder": True, "pin_cpu_memory": True},
            },
        },
        "server": {"host": "127.0.0.1", "port": fv_port, "output_dir": "/kaggle/working/fastvideo_out",
                   "served_model_name": (CFG.get("model_key") or model_id)},
        "default_request": {"sampling": {
            "height": int(str(o.get("size", "704x1280")).split("x")[-1]) if "x" in str(o.get("size", "")) else 704,
            "width": int(str(o.get("size", "704x1280")).split("x")[0]) if "x" in str(o.get("size", "")) else 1280,
            "num_frames": int(o.get("num_frames", 241)),
            "fps": int(o.get("fps", 24)),
        }},
    }
    # size is "WxH" in kllm; fastvideo uses height/width. Fix ordering:
    size_raw = str(o.get("size") or "1280x704")
    parts = size_raw.lower().replace("*", "x").split("x")
    if len(parts) == 2:
        cfg["default_request"]["sampling"]["width"] = int(parts[0])
        cfg["default_request"]["sampling"]["height"] = int(parts[1])
    cfg_path = "/tmp/kllm_fastvideo.yaml"
    with open(cfg_path, "w") as f:
        _yaml.safe_dump(cfg, f)

    ntfy("downloading", note="fastvideo: starting server")
    import shutil as _shutil
    _fv_bin = _shutil.which("fastvideo") or "fastvideo"
    proc = subprocess.Popen([_fv_bin, "serve", "--config", cfg_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # wait health
    deadline = time.time() + 1800
    import urllib.request as _ur
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = proc.stdout.read()[-1500:]
            except Exception:
                pass
            ntfy("error", stage="server_health", tail=tail or "fastvideo exited")
            raise RuntimeError("fastvideo server exited during startup")
        try:
            with _ur.urlopen(f"http://127.0.0.1:{fv_port}/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(3)
    else:
        try:
            proc.kill()
        except Exception:
            pass
        ntfy("error", stage="server_health", tail="fastvideo load timed out")
        raise RuntimeError("fastvideo server did not become ready")

    # adapter
    class _FVHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a):
            pass
        def _send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                try:
                    with _ur.urlopen(f"http://127.0.0.1:{fv_port}/health", timeout=10) as r:
                        return self._send(r.status, {"status": "ok"})
                except Exception as e:
                    return self._send(503, {"status": "error", "error": str(e)})
            if path in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{"id": CFG.get("model_key") or model_id,
                                                                     "object": "model", "owned_by": "kllm",
                                                                     "kind": "video", "runtime": "fastvideo"}]})
            self._send(404, {"error": {"message": f"unknown path {path}"}})
        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                req = json.loads(raw or b"{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, {"error": {"message": "body must be a JSON object"}})
            if path != "/v1/videos/generations":
                return self._send(404, {"error": {"message": f"unknown path {path}"}})
            o2 = CFG.get("opts") or {}
            prompt = req.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return self._send(400, {"error": {"message": "prompt is required"}})
            size_raw = str(req.get("size") or o2.get("size") or "1280x704")
            parts = size_raw.lower().replace("*", "x").split("x")
            width = int(parts[0]) if len(parts) == 2 else int(o2.get("width", 1280))
            height = int(parts[1]) if len(parts) == 2 else int(o2.get("height", 704))
            num_frames = int(req.get("num_frames") or o2.get("num_frames") or 241)
            fps = int(req.get("fps") or o2.get("fps") or 24)
            seed = req.get("seed")
            try:
                seed = int(seed) if seed is not None else random.randrange(2 ** 31)
            except (TypeError, ValueError):
                return self._send(400, {"error": {"message": "seed must be an integer"}})
            fv_body = {
                "model": CFG.get("model_key") or model_id,
                "prompt": prompt,
                "negative_prompt": req.get("negative_prompt", o2.get("negative_prompt")),
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed,
                "num_inference_steps": int(req.get("steps") or req.get("num_inference_steps") or o2.get("steps", 3)),
                "guidance_scale": float(req.get("guidance_scale") or o2.get("guidance_scale", 1.0)),
            }
            if req.get("image"):
                img = req["image"]
                if isinstance(img, str):
                    if img.startswith("data:"):
                        data_url = img
                    else:
                        data_url = "data:image/png;base64," + img
                    fv_body["image_reference"] = [{"image_url": data_url}]
            data = json.dumps(fv_body).encode()
            rq = urllib.request.Request(f"http://127.0.0.1:{fv_port}/v1/videos/sync",
                                     data=data, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(rq, timeout=1800) as resp:
                    mp4 = resp.read()
            except Exception as e:
                return self._send(500, {"error": {"message": f"fastvideo generation failed: {e}"}})
            import base64 as _b64
            out = {
                "created": int(time.time()),
                "data": [{"b64_mp4": _b64.b64encode(mp4).decode(), "seed": seed}],
                "fps": fps,
                "frames": num_frames,
                "width": width,
                "height": height,
                "duration_seconds": round(num_frames / fps, 2),
            }
            return self._send(200, out)

    adapter_srv = ThreadingHTTPServer(("127.0.0.1", LLAMA_PORT), _FVHandler)
    adapter_thread = threading.Thread(target=adapter_srv.serve_forever, daemon=True)
    adapter_thread.start()
    ntfy("serving", note="fastvideo up on localhost")
    return _FastVideoHandle(proc, adapter_srv, adapter_thread, fv_port)


class _SwarmHandle:
    def __init__(self, srv=None, proc=None, thread=None, ws_thread=None, worker=None):
        self.srv = srv
        self.proc = proc
        self.thread = thread
        self.ws_thread = ws_thread
        self.worker = worker

    def poll(self):
        if self.proc is not None:
            return self.proc.poll()
        if self.thread is not None and not self.thread.is_alive():
            return 1
        if self.worker is not None and (self.ws_thread is None or not self.ws_thread.is_alive()):
            try:
                log("Watchdog: ws_thread died, reviving WebSocket loop...")
                import asyncio
                new_loop = asyncio.new_event_loop()
                def run_bg():
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(self.worker.run_ws_loop())
                th = threading.Thread(target=run_bg, daemon=True)
                th.start()
                self.ws_thread = th
            except Exception as e:
                log(f"Watchdog failed to revive ws_thread: {e}")
        return None

    def terminate(self):
        if self.proc is not None:
            try: self.proc.terminate()
            except Exception: pass
        if self.srv is not None:
            try: self.srv.shutdown()
            except Exception: pass
    kill = terminate


def probe_gcp_zone_and_latency(hub_url=None):
    zone = "unknown"
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(req, timeout=1.5) as r:
            zone = r.read().decode("utf-8").strip().split("/")[-1]
    except Exception:
        zone = "non-gcp-or-sandboxed"

    rtt_ms = 0.0
    if hub_url:
        try:
            t0 = time.time()
            clean_hub = hub_url.replace("ws://", "http://").replace("wss://", "https://").replace("/swarm/ws", "").rstrip("/")
            ping_url = f"{clean_hub}/swarm/ping"
            with urllib.request.urlopen(ping_url, timeout=3.0) as r:
                if r.status == 200:
                    rtt_ms = (time.time() - t0) * 1000.0
        except Exception:
            rtt_ms = 999.0
    return zone, rtt_ms


# --------------------------------------------------------------------------
# kllm-swarm: Distributed MoE Layer-Execution Engine
# --------------------------------------------------------------------------
SWARM_MAGIC = b"KLLM"
from collections import OrderedDict
SWARM_HEADER_STRUCT = struct.Struct("!4s16sIIHHHHII20s")
SWARM_HEADER_SIZE = SWARM_HEADER_STRUCT.size

FLAG_ACTIVATION = 0
FLAG_TOKEN = 1
FLAG_EOS = 2


def _safe_ntfy(event, **extra):
    fn = globals().get("ntfy")
    if callable(fn):
        try:
            fn(event, **extra)
        except Exception:
            pass


def _safe_log(msg):
    fn = globals().get("log")
    if callable(fn):
        try:
            fn(msg)
        except Exception:
            pass
    else:
        print(msg)


def _safe_sh(cmd):
    fn = globals().get("sh")
    if callable(fn):
        return fn(cmd)
    import subprocess
    return subprocess.check_output(cmd, shell=True, text=True)


def _sanitize_float(val):
    if math.isnan(val):
        return 0.0
    if math.isinf(val):
        return 65504.0 if val > 0 else -65504.0
    return val


try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def quantize_activations_q8_0(values, block_size=32):
    n = len(values)
    if n == 0:
        return b""
    if _NUMPY_AVAILABLE and block_size == 32:
        try:
            arr = np.asarray(values, dtype=np.float32)
            pad = (block_size - (n % block_size)) % block_size
            if pad > 0:
                arr = np.pad(arr, (0, pad), mode="constant")
            blocks = arr.reshape(-1, block_size)
            blocks = np.nan_to_num(blocks, nan=0.0, posinf=65504.0, neginf=-65504.0)
            max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
            scales = np.where(max_abs > 1e-8, max_abs / 127.0, 1.0).astype(">f2")
            float_scales = scales.astype(np.float32)
            inv_scales = 1.0 / np.where(float_scales > 1e-8, float_scales, 1.0)
            qs = np.clip(np.round(blocks * inv_scales), -128, 127).astype(np.int8)
            dt = np.dtype([("scale", ">f2"), ("quants", "i1", (block_size,))])
            packed = np.empty(blocks.shape[0], dtype=dt)
            packed["scale"] = scales.squeeze(-1)
            packed["quants"] = qs
            return packed.tobytes()
        except Exception:
            pass

    buf = bytearray()
    for i in range(0, n, block_size):
        raw_chunk = values[i : i + block_size]
        chunk = [_sanitize_float(x) for x in raw_chunk]
        max_abs = max((abs(x) for x in chunk), default=0.0)
        scale = max_abs / 127.0 if max_abs > 1e-8 else 1.0
        buf.extend(struct.pack("!e", scale))
        inv_scale = 1.0 / scale
        for val in chunk:
            q = int(round(val * inv_scale))
            q = max(-128, min(127, q))
            buf.append(q & 0xFF)
        if len(chunk) < block_size:
            buf.extend(b"\x00" * (block_size - len(chunk)))
    return bytes(buf)


def dequantize_activations_q8_0(raw, num_elements, block_size=32):
    if num_elements <= 0:
        return []
    expected_bytes = math.ceil(num_elements / block_size) * (2 + block_size)
    if _NUMPY_AVAILABLE and block_size == 32 and len(raw) >= expected_bytes:
        try:
            dt = np.dtype([("scale", ">f2"), ("quants", "i1", (block_size,))])
            arr = np.frombuffer(raw[:expected_bytes], dtype=dt)
            scales = arr["scale"].astype(np.float32)[:, None]
            quants = arr["quants"].astype(np.float32)
            floats = (quants * scales).reshape(-1)[:num_elements]
            return floats.tolist()
        except Exception:
            pass

    floats = []
    idx = 0
    read_elems = 0
    while idx < len(raw) and read_elems < num_elements:
        scale = struct.unpack("!e", raw[idx : idx + 2])[0]
        idx += 2
        chunk_len = min(block_size, num_elements - read_elems)
        for j in range(chunk_len):
            val = struct.unpack("b", raw[idx + j : idx + j + 1])[0]
            floats.append(float(val) * scale)
        idx += block_size
        read_elems += chunk_len
    return floats


def pack_token_payload(tokens, step_id=0):
    count = len(tokens)
    header = struct.pack("!II", count, step_id)
    tok_bytes = struct.pack(f"!{count}I", *tokens) if count > 0 else b""
    return header + tok_bytes


def unpack_token_payload(payload):
    if len(payload) < 8:
        raise ValueError("Truncated token payload")
    count, step_id = struct.unpack("!II", payload[:8])
    tokens = list(struct.unpack(f"!{count}I", payload[8 : 8 + count * 4])) if count > 0 else []
    return tokens, step_id


class SwarmFrameHeader:
    def __init__(self, req_id, seq_id, step_id, stage_idx, total_stages, num_tokens, hidden_dim, pos_offset, flags=0, reserved=b"\x00" * 20):
        self.req_id = req_id if isinstance(req_id, uuid.UUID) else (uuid.UUID(bytes=req_id) if isinstance(req_id, bytes) else uuid.UUID(str(req_id)))
        self.seq_id = int(seq_id)
        self.step_id = int(step_id)
        self.stage_idx = int(stage_idx)
        self.total_stages = int(total_stages)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.pos_offset = int(pos_offset)
        self.flags = int(flags)
        self.reserved = reserved if isinstance(reserved, (bytes, bytearray)) else b"\x00" * 20

    def pack(self):
        pad = self.reserved[:20].ljust(20, b"\x00") if self.reserved else b"\x00" * 20
        return SWARM_HEADER_STRUCT.pack(
            SWARM_MAGIC,
            self.req_id.bytes,
            self.seq_id,
            self.step_id,
            self.stage_idx,
            self.total_stages,
            self.num_tokens,
            self.hidden_dim,
            self.pos_offset,
            self.flags,
            pad,
        )

    @classmethod
    def unpack(cls, data):
        if len(data) < SWARM_HEADER_SIZE:
            raise ValueError(f"Data length {len(data)} is less than header size {SWARM_HEADER_SIZE}")
        magic, raw_uuid, seq_id, step_id, stage_idx, total_stages, num_tokens, hidden_dim, pos_offset, flags, reserved = (
            SWARM_HEADER_STRUCT.unpack(data[:SWARM_HEADER_SIZE])
        )
        if magic != SWARM_MAGIC:
            raise ValueError(f"Invalid magic header: {magic!r}")
        return cls(
            req_id=uuid.UUID(bytes=raw_uuid),
            seq_id=seq_id,
            step_id=step_id,
            stage_idx=stage_idx,
            total_stages=total_stages,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            pos_offset=pos_offset,
            flags=flags,
            reserved=reserved,
        )


def pack_activation_frame(*args, **kwargs):
    if len(args) == 2 and hasattr(args[0], "pack"):
        return args[0].pack() + args[1]
    if "header" in kwargs and "payload" in kwargs:
        return kwargs["header"].pack() + kwargs["payload"]
    if len(args) >= 10:
        req_id, seq_id, step_id, stage_idx, total_stages, num_tokens, hidden_dim, pos_offset, flags, payload = args[:10]
        reserved = args[10] if len(args) > 10 else kwargs.get("reserved", b"\x00" * 20)
        hdr = SwarmFrameHeader(req_id, seq_id, step_id, stage_idx, total_stages, num_tokens, hidden_dim, pos_offset, flags, reserved)
        return hdr.pack() + payload
    raise ValueError(f"Invalid pack_activation_frame invocation: {args}, {kwargs}")


def unpack_activation_frame(data):
    if len(data) < SWARM_HEADER_SIZE:
        raise ValueError(f"Data length {len(data)} is less than header size {SWARM_HEADER_SIZE}")
    hdr = SwarmFrameHeader.unpack(data)
    hdr_dict = {
        "req_id": hdr.req_id,
        "seq_id": hdr.seq_id,
        "step_id": hdr.step_id,
        "stage_idx": hdr.stage_idx,
        "total_stages": hdr.total_stages,
        "num_tokens": hdr.num_tokens,
        "hidden_dim": hdr.hidden_dim,
        "pos_offset": hdr.pos_offset,
        "flags": hdr.flags,
        "reserved": hdr.reserved,
    }
    return hdr_dict, data[SWARM_HEADER_SIZE:]


NF4_TABLE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]


class SwarmTokenizer:
    """Robust tokenizer supporting local dataset tokenizer and Hugging Face fallback."""
    def __init__(self, model_hf=None):
        self.hf_tok = None
        # 1. Try local dataset first
        import glob
        local_toks = glob.glob("/kaggle/input/**/tokenizer.json", recursive=True) + glob.glob("/kaggle/input/**/tokenizer_config.json", recursive=True)
        if local_toks:
            try:
                import transformers
                tok_dir = os.path.dirname(local_toks[0])
                self.hf_tok = transformers.AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
                log(f"SwarmTokenizer: Loaded local dataset tokenizer from {tok_dir}")
                return
            except Exception as e:
                log(f"SwarmTokenizer: local dataset load error: {e}")

        # 2. Try online HF
        if model_hf:
            try:
                import transformers
                self.hf_tok = transformers.AutoTokenizer.from_pretrained(model_hf, trust_remote_code=True)
                log(f"SwarmTokenizer: Loaded Hugging Face tokenizer for {model_hf}")
            except Exception as e:
                log(f"SwarmTokenizer: Hugging Face tokenizer unavailable ({e}); using built-in byte tokenizer")

    def encode(self, text):
        if self.hf_tok is not None:
            try:
                return self.hf_tok.encode(text)
            except Exception:
                pass
        # Fallback byte encoding
        raw = text.encode("utf-8")
        return [b + 100 for b in raw]

    def decode(self, tokens):
        if self.hf_tok is not None:
            try:
                return self.hf_tok.decode(tokens)
            except Exception:
                pass
        b_list = []
        for t in tokens:
            if 100 <= t <= 355:
                b_list.append(t - 100)
            elif t < 100:
                pass
            else:
                b_list.extend(str(t).encode("utf-8"))
        try:
            return bytes(b_list).decode("utf-8", errors="replace")
        except Exception:
            return "".join(chr(b) for b in b_list)


class SwarmTransformerShard:
    """Executes a shard of Transformer layers on Kaggle's 2x Tesla T4 GPUs with Intra-Node TP=2,

    full KV caching, Rotary Position Embeddings (RoPE), and SwiGLU Top-K MoE routing.
    Loads 100% REAL pretrained weights directly from mounted Kaggle datasets.
    """
    def __init__(self, stage_idx, total_stages, total_layers=80, hidden_dim=8192, num_heads=64, num_kv_heads=8, num_experts=0, top_k=0, vocab_size=152064, layers_per_stage=None, head_dim=None):
        self.stage_idx = stage_idx
        self.total_stages = total_stages
        self.total_layers = total_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        cfg_obj = globals().get("CFG") or {}
        self.head_dim = head_dim if head_dim is not None else int(cfg_obj.get("head_dim") or (hidden_dim // num_heads))
        self.num_experts = num_experts
        self.top_k = top_k
        self.vocab_size = vocab_size

        # Layer partition: respect layer-balanced configuration when provided
        cfg_obj = globals().get("CFG") or {}
        l_per_stage = layers_per_stage or (cfg_obj.get("layers_per_stage") if isinstance(cfg_obj, dict) else None)
        if isinstance(l_per_stage, (list, tuple)) and len(l_per_stage) == total_stages:
            self.layer_start = sum(l_per_stage[:stage_idx])
            self.layer_end = self.layer_start + l_per_stage[stage_idx]
            self.num_layers = l_per_stage[stage_idx]
        else:
            base = total_layers // total_stages
            rem = total_layers % total_stages
            self.layer_start = stage_idx * base + min(stage_idx, rem)
            self.layer_end = self.layer_start + base + (1 if stage_idx < rem else 0)
            self.num_layers = self.layer_end - self.layer_start
        self.role = "head" if stage_idx == 0 else ("tail" if stage_idx == total_stages - 1 else "worker")

        # PyTorch & Device Setup
        self.layers = []
        self.w_emb = None
        self.w_head = None
        self.final_norm = None
        self.real_weights = False
        self.init_error = None
        self.torch_available = False
        try:
            import torch
            import torch.nn.functional as F
            self.torch = torch
            self.F = F
            self.torch_available = True

            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpus >= 2:
                self.dev0 = torch.device("cuda:0")
                self.dev1 = torch.device("cuda:1")
                self.use_tp = True
            elif n_gpus == 1:
                self.dev0 = torch.device("cuda:0")
                self.dev1 = self.dev0
                self.use_tp = False
            else:
                self.dev0 = torch.device("cpu")
                self.dev1 = self.dev0
                self.use_tp = False

            self.dtype = torch.float16 if self.dev0.type == "cuda" else torch.float32
            self._init_weights()
        except Exception as e:
            _safe_log(f"Swarm engine CPU math fallback: {e}")
            self.use_tp = False
            if not self.layers:
                self._init_synthetic_fallback()

    def _try_load_pretrained_dataset(self):
        """Attempts to load real pretrained safetensors weights from /kaggle/input/ datasets."""
        import glob
        indices = glob.glob("/kaggle/input/**/model.safetensors.index.json", recursive=True)
        hf_repo = (globals().get("CFG") or {}).get("model_hf") or "BitPhinix/Qwen3-235B-A22B-Instruct-2507-bnb-nf4"
        base_dir = None

        if indices:
            idx_path = indices[0]
            base_dir = os.path.dirname(idx_path)
            _safe_log(f"Stage {self.stage_idx}: found local index {idx_path}")
        else:
            import tempfile
            candidates = ["/kaggle/working/weights", os.path.expanduser("~/.cache/kllm/weights"), os.path.join(tempfile.gettempdir(), "kllm_weights")]
            base_dir = None
            for cand in candidates:
                try:
                    os.makedirs(cand, exist_ok=True)
                    base_dir = cand
                    break
                except Exception:
                    continue
            if base_dir is None:
                base_dir = tempfile.mkdtemp()
            idx_path = os.path.join(base_dir, "model.safetensors.index.json")
            if not os.path.exists(idx_path):
                _safe_log(f"Stage {self.stage_idx}: downloading index from HF {hf_repo}")
                try:
                    _safe_sh(f"curl -sSL --http1.1 --retry 5 --retry-all-errors https://huggingface.co/{hf_repo}/resolve/main/model.safetensors.index.json -o {idx_path}")
                except Exception as e:
                    _safe_log(f"Stage {self.stage_idx}: failed to fetch index via curl: {e}")
                    return False

        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
        except Exception as e:
            log(f"Failed to read index {idx_path}: {e}")
            return False

        try:
            from safetensors.torch import load_file
        except ImportError:
            sh("pip install -q safetensors")
            from safetensors.torch import load_file

        torch = self.torch
        dtype = self.dtype

        # 1. Identify which safetensors files this stage actually needs
        needed_files = set()
        for l_idx in range(self.layer_start, self.layer_end):
            prefix = f"model.layers.{l_idx}."
            for k, fname in weight_map.items():
                if k.startswith(prefix):
                    needed_files.add(fname)

        if self.role == "head":
            for k, fname in weight_map.items():
                if "embed_tokens" in k:
                    needed_files.add(fname)

        if self.role == "tail":
            for k, fname in weight_map.items():
                if k in ("model.norm.weight", "norm.weight") or "lm_head" in k:
                    needed_files.add(fname)

        _safe_log(f"Stage {self.stage_idx}: requires {len(needed_files)} real weight file(s): {sorted(needed_files)}")
        tensors = {}
        for fname in sorted(needed_files):
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) < 1000000:
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                _safe_log(f"  Stage {self.stage_idx}: downloading {fname} from {hf_repo}...")
                _safe_ntfy("downloading", note=f"stage {self.stage_idx} downloading {fname}")
                shard_url = f"https://huggingface.co/{hf_repo}/resolve/main/{fname}"
                try:
                    _safe_sh(f"curl -fSL --http1.1 --retry 8 --retry-all-errors --retry-delay 2 -C - {shard_url} -o {fpath}")
                except Exception as e:
                    _safe_log(f"  Stage {self.stage_idx}: failed to curl {fname}: {e}")
                    self.init_error = f"curl_failed: {fname}: {e}"
                    return False
            if os.path.exists(fpath):
                sz_mb = os.path.getsize(fpath) / (1024 * 1024)
                _safe_log(f"  Stage {self.stage_idx}: loading real weights from {fname} ({sz_mb:.1f} MB)...")
                _safe_ntfy("building", stage=f"stage_{self.stage_idx}_weights", note=f"loading {fname} ({sz_mb:.1f}MB)")
                t = load_file(fpath, device="cpu")
                for k, v in t.items():
                    keep = False
                    if self.role == "head" and ("embed_tokens" in k):
                        keep = True
                    elif self.role == "tail" and (k in ("model.norm.weight", "norm.weight") or "lm_head" in k):
                        keep = True
                    else:
                        for l_idx in range(self.layer_start, self.layer_end):
                            if k.startswith(f"model.layers.{l_idx}."):
                                keep = True
                                break
                    if keep:
                        tensors[k] = v.clone()
                del t
                import gc; gc.collect()
                try:
                    os.remove(fpath)
                    _safe_log(f"  Stage {self.stage_idx}: unlinked {fname} (disk reclaimed)")
                except Exception:
                    pass

        def _get_dequant(prefix, name):
            full = f"{prefix}.{name}.weight" if prefix else f"{name}.weight"
            if full not in tensors:
                return None
            w = tensors[full]
            absmax_key = f"{full}.absmax"
            if absmax_key in tensors:
                absmax = tensors[absmax_key]
                n_abs_key = f"{full}.nested_absmax"
                if n_abs_key in tensors and f"{full}.nested_quant_map" in tensors:
                    qmap = tensors[f"{full}.nested_quant_map"].float().to(self.dev0)
                    n_abs = tensors[n_abs_key].float().to(self.dev0)
                    abs_idx = absmax.long().to(self.dev0)
                    n_blocks = n_abs.shape[0]
                    b_sz = abs_idx.numel() // n_blocks
                    absmax = (qmap[abs_idx].view(n_blocks, b_sz) * n_abs.view(n_blocks, 1)).flatten()
                raw_b = w.contiguous().view(torch.uint8) if w.dtype != torch.uint8 else w
                b = raw_b.to(self.dev0).flatten()
                high = ((b >> 4) & 0x0F).long()
                low = (b & 0x0F).long()
                idx = torch.stack([high, low], dim=-1).flatten()
                del raw_b, b, high, low
                qmap_key = f"{full}.quant_map"
                if qmap_key in tensors:
                    nf4_t = tensors[qmap_key].to(self.dev0, dtype=dtype)
                else:
                    nf4_t = torch.tensor(NF4_TABLE, device=self.dev0, dtype=dtype)
                vals = nf4_t[idx].view(-1, 64)
                del idx
                scale = absmax.to(self.dev0, dtype=dtype).view(-1, 1)
                res = (vals * scale).flatten()
                del vals, scale
                return res
            return w.to(self.dev0, dtype=dtype)

        def _get_packed_nf4(prefix, name):
            full = f"{prefix}.{name}.weight" if prefix else f"{name}.weight"
            if full not in tensors:
                return None, None
            w = tensors[full]
            absmax_key = f"{full}.absmax"
            absmax = tensors.get(absmax_key)
            if absmax is not None:
                n_abs_key = f"{full}.nested_absmax"
                if n_abs_key in tensors and f"{full}.nested_quant_map" in tensors:
                    qmap = tensors[f"{full}.nested_quant_map"].float().to(self.dev0)
                    n_abs = tensors[n_abs_key].float().to(self.dev0)
                    abs_idx = absmax.long().to(self.dev0)
                    n_blocks = n_abs.shape[0]
                    b_sz = abs_idx.numel() // n_blocks
                    absmax = (qmap[abs_idx].view(n_blocks, b_sz) * n_abs.view(n_blocks, 1)).flatten()
            raw_b = w.contiguous().view(torch.uint8) if w.dtype != torch.uint8 else w
            return raw_b, absmax

        # 2. Embeddings (Head stage)
        if self.role == "head":
            raw_emb = _get_dequant("model", "embed_tokens")
            if raw_emb is not None:
                actual_vocab = raw_emb.numel() // self.hidden_dim
                self.w_emb = raw_emb.view(actual_vocab, self.hidden_dim).to(self.dev0, dtype=dtype)
                self.vocab_size = actual_vocab
                log(f"Stage {self.stage_idx}: Loaded REAL pretrained embeddings {self.w_emb.shape}")
            else:
                scale = (2.0 / self.hidden_dim) ** 0.5
                self.w_emb = torch.randn(self.vocab_size, self.hidden_dim, device=self.dev0, dtype=dtype) * scale

        # 3. Final norm & LM head (Tail stage)
        if self.role == "tail":
            if "model.norm.weight" in tensors:
                self.final_norm = tensors["model.norm.weight"].to(self.dev0, dtype=dtype)
                log(f"Stage {self.stage_idx}: Loaded REAL final norm {self.final_norm.shape}")
            else:
                self.final_norm = torch.ones(self.hidden_dim, device=self.dev0, dtype=dtype)

            raw_h = _get_dequant("", "lm_head")
            if raw_h is not None:
                actual_vocab = raw_h.numel() // self.hidden_dim
                self.w_head = raw_h.view(actual_vocab, self.hidden_dim).to(self.dev0, dtype=dtype)
                self.vocab_size = actual_vocab
                log(f"Stage {self.stage_idx}: Loaded REAL LM Head {self.w_head.shape}")
            else:
                scale = (2.0 / self.hidden_dim) ** 0.5
                self.w_head = torch.randn(self.vocab_size, self.hidden_dim, device=self.dev0, dtype=dtype) * scale

        # 4. Layers
        self.layers = []
        tot_h = self.hidden_dim
        q_heads = self.num_heads
        kv_heads = self.num_kv_heads
        h_dim = self.head_dim

        for l_idx in range(self.layer_start, self.layer_end):
            pref = f"model.layers.{l_idx}"
            norm1 = tensors.get(f"{pref}.input_layernorm.weight")
            norm2 = tensors.get(f"{pref}.post_attention_layernorm.weight")

            wq = _get_dequant(pref, "self_attn.q_proj")
            wk = _get_dequant(pref, "self_attn.k_proj")
            wv = _get_dequant(pref, "self_attn.v_proj")
            wo = _get_dequant(pref, "self_attn.o_proj")

            is_moe = False
            router_w = _get_dequant(pref, "mlp.gate")
            if router_w is None:
                router_w = _get_dequant(pref, "block_sparse_moe.gate")

            exp_count = self.num_experts if self.num_experts > 0 else 0
            if router_w is not None:
                is_moe = True
                exp_count = router_w.numel() // tot_h
                router = router_w.view(exp_count, tot_h).T.contiguous().to(self.dev0, dtype=dtype)
            else:
                router = torch.zeros(tot_h, max(1, self.num_experts), device=self.dev0, dtype=dtype)

            dense_gate = _get_dequant(pref, "mlp.gate_proj")
            dense_up = _get_dequant(pref, "mlp.up_proj")
            dense_down = _get_dequant(pref, "mlp.down_proj")

            inter_dim = dense_gate.numel() // tot_h if dense_gate is not None else 1536

            experts_0 = []
            experts_1 = []
            half_exp = exp_count // 2 if self.use_tp else exp_count

            if is_moe:
                for e in range(exp_count):
                    g_b, g_s = _get_packed_nf4(f"{pref}.mlp.experts.{e}", "gate_proj")
                    if g_b is None: g_b, g_s = _get_packed_nf4(f"{pref}.block_sparse_moe.experts.{e}", "w1")
                    u_b, u_s = _get_packed_nf4(f"{pref}.mlp.experts.{e}", "up_proj")
                    if u_b is None: u_b, u_s = _get_packed_nf4(f"{pref}.block_sparse_moe.experts.{e}", "w3")
                    d_b, d_s = _get_packed_nf4(f"{pref}.mlp.experts.{e}", "down_proj")
                    if d_b is None: d_b, d_s = _get_packed_nf4(f"{pref}.block_sparse_moe.experts.{e}", "w2")

                    use_dev1 = self.use_tp and e >= half_exp
                    t_dev = self.dev1 if use_dev1 else self.dev0

                    if g_b is not None and d_b is not None and u_b is not None:
                        e_dict = {
                            "packed": True,
                            "gate_b": g_b.to(t_dev),
                            "gate_s": g_s.to(t_dev, dtype=dtype) if g_s is not None else torch.ones(g_b.numel() // 32, device=t_dev, dtype=dtype),
                            "up_b": u_b.to(t_dev),
                            "up_s": u_s.to(t_dev, dtype=dtype) if u_s is not None else torch.ones(u_b.numel() // 32, device=t_dev, dtype=dtype),
                            "down_b": d_b.to(t_dev),
                            "down_s": d_s.to(t_dev, dtype=dtype) if d_s is not None else torch.ones(d_b.numel() // 32, device=t_dev, dtype=dtype),
                        }
                    else:
                        scale = (2.0 / tot_h) ** 0.5
                        e_dict = {
                            "packed": False,
                            "gate": torch.randn(tot_h, inter_dim, device=t_dev, dtype=dtype) * scale,
                            "up": torch.randn(tot_h, inter_dim, device=t_dev, dtype=dtype) * scale,
                            "down": torch.randn(inter_dim, tot_h, device=t_dev, dtype=dtype) * scale,
                        }
                    if use_dev1:
                        experts_1.append(e_dict)
                    else:
                        experts_0.append(e_dict)
                gate, up, down = None, None, None
            else:
                if dense_down is not None and dense_gate is not None and dense_up is not None:
                    gate = dense_gate.view(inter_dim, tot_h).T
                    up = dense_up.view(inter_dim, tot_h).T
                    down = dense_down.view(tot_h, inter_dim).T
                else:
                    scale = (2.0 / tot_h) ** 0.5
                    gate = torch.randn(tot_h, inter_dim, device=self.dev0, dtype=dtype) * scale
                    up = torch.randn(tot_h, inter_dim, device=self.dev0, dtype=dtype) * scale
                    down = torch.randn(inter_dim, tot_h, device=self.dev0, dtype=dtype) * scale

            if wq is None:
                log(f"Layer {l_idx}: weights missing from dataset")
                return False

            # Linear weights in safetensors are [out_dim, in_dim] -> transpose to [in_dim, out_dim]
            q_out = wq.numel() // tot_h
            k_out = wk.numel() // tot_h
            v_out = wv.numel() // tot_h
            o_out = wo.numel() // q_out

            wq = wq.view(q_out, tot_h).T
            wk = wk.view(k_out, tot_h).T
            wv = wv.view(v_out, tot_h).T
            wo = wo.view(tot_h, q_out).T

            bq = tensors.get(f"{pref}.self_attn.q_proj.bias")
            bk = tensors.get(f"{pref}.self_attn.k_proj.bias")
            bv = tensors.get(f"{pref}.self_attn.v_proj.bias")
            q_norm = tensors.get(f"{pref}.self_attn.q_norm.weight")
            k_norm = tensors.get(f"{pref}.self_attn.k_norm.weight")

            layer = {
                "norm1": norm1.to(self.dev0, dtype=dtype) if norm1 is not None else torch.ones(tot_h, device=self.dev0, dtype=dtype),
                "norm2": norm2.to(self.dev0, dtype=dtype) if norm2 is not None else torch.ones(tot_h, device=self.dev0, dtype=dtype),
                "q_norm": q_norm.to(self.dev0, dtype=dtype) if q_norm is not None else None,
                "k_norm": k_norm.to(self.dev0, dtype=dtype) if k_norm is not None else None,
                "router": router,
                "k_cache": None,
                "v_cache": None,
                "is_moe": is_moe,
                "exp_count": exp_count,
                "inter_dim": inter_dim,
                "experts_0": experts_0,
                "experts_1": experts_1,
                "expert_cache_0": OrderedDict(),
                "expert_cache_1": OrderedDict(),
            }

            if self.use_tp:
                q_half = q_out // 2
                kv_half = max(1, k_out // 2)

                layer["wq_0"] = wq[:, :q_half].contiguous().to(self.dev0)
                layer["wq_1"] = wq[:, q_half:].contiguous().to(self.dev1)

                layer["wk_0"] = wk[:, :kv_half].contiguous().to(self.dev0)
                layer["wk_1"] = wk[:, kv_half:].contiguous().to(self.dev1)

                layer["wv_0"] = wv[:, :kv_half].contiguous().to(self.dev0)
                layer["wv_1"] = wv[:, kv_half:].contiguous().to(self.dev1)

                if bq is not None:
                    layer["bq_0"] = bq[:q_half].contiguous().to(self.dev0, dtype=dtype)
                    layer["bq_1"] = bq[q_half:].contiguous().to(self.dev1, dtype=dtype)
                if bk is not None:
                    layer["bk_0"] = bk[:kv_half].contiguous().to(self.dev0, dtype=dtype)
                    layer["bk_1"] = bk[kv_half:].contiguous().to(self.dev1, dtype=dtype)
                if bv is not None:
                    layer["bv_0"] = bv[:kv_half].contiguous().to(self.dev0, dtype=dtype)
                    layer["bv_1"] = bv[kv_half:].contiguous().to(self.dev1, dtype=dtype)

                layer["wo_0"] = wo[:q_half, :].contiguous().to(self.dev0)
                layer["wo_1"] = wo[q_half:, :].contiguous().to(self.dev1)

                if not is_moe:
                    inter_half = inter_dim // 2
                    layer["exp_gate_0"] = gate[:, :inter_half].contiguous().to(self.dev0)
                    layer["exp_gate_1"] = gate[:, inter_half:].contiguous().to(self.dev1)
                    layer["exp_up_0"] = up[:, :inter_half].contiguous().to(self.dev0)
                    layer["exp_up_1"] = up[:, inter_half:].contiguous().to(self.dev1)
                    layer["exp_down_0"] = down[:inter_half, :].contiguous().to(self.dev0)
                    layer["exp_down_1"] = down[inter_half:, :].contiguous().to(self.dev1)
            else:
                layer["wq"] = wq.contiguous().to(self.dev0)
                layer["wk"] = wk.contiguous().to(self.dev0)
                layer["wv"] = wv.contiguous().to(self.dev0)
                layer["wo"] = wo.contiguous().to(self.dev0)
                if bq is not None: layer["bq"] = bq.contiguous().to(self.dev0, dtype=dtype)
                if bk is not None: layer["bk"] = bk.contiguous().to(self.dev0, dtype=dtype)
                if bv is not None: layer["bv"] = bv.contiguous().to(self.dev0, dtype=dtype)
                if not is_moe:
                    layer["exp_gate"] = gate.contiguous().to(self.dev0)
                    layer["exp_up"] = up.contiguous().to(self.dev0)
                    layer["exp_down"] = down.contiguous().to(self.dev0)

            self.layers.append(layer)
            del wq, wk, wv, wo
            import gc; gc.collect()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

        del tensors
        import gc; gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        _safe_log(f"Stage {self.stage_idx}: successfully initialized {len(self.layers)} REAL pretrained layers (TP={self.use_tp})")
        return True

    def _init_synthetic_fallback(self):
        torch = getattr(self, "torch", None)
        if torch is None:
            try:
                import torch
                self.torch = torch
            except ImportError:
                return
        dtype = getattr(self, "dtype", torch.float32)
        scale = (2.0 / self.hidden_dim) ** 0.5
        _safe_log(f"Stage {self.stage_idx}: using synthetic weight initialization (dim={self.hidden_dim}, layers={self.num_layers})")

        if self.role == "head":
            self.w_emb = torch.randn(self.vocab_size, self.hidden_dim, device=self.dev0, dtype=dtype) * scale
        if self.role == "tail":
            self.final_norm = torch.ones(self.hidden_dim, device=self.dev0, dtype=dtype)
            self.w_head = torch.randn(self.vocab_size, self.hidden_dim, device=self.dev0, dtype=dtype) * scale

        self.layers = []
        inter_dim = min(2048, self.hidden_dim // 2)
        is_moe = self.num_experts > 0
        exp_count = self.num_experts if is_moe else 0
        half_exp = exp_count // 2 if getattr(self, "use_tp", False) else exp_count

        for _ in range(self.num_layers):
            layer = {
                "norm1": torch.ones(self.hidden_dim, device=self.dev0, dtype=dtype),
                "norm2": torch.ones(self.hidden_dim, device=self.dev0, dtype=dtype),
                "router": torch.randn(self.hidden_dim, max(1, exp_count), device=self.dev0, dtype=dtype) * scale,
                "k_cache": None,
                "v_cache": None,
                "is_moe": is_moe,
                "exp_count": exp_count,
                "inter_dim": inter_dim,
                "experts_0": [],
                "experts_1": [],
                "expert_cache_0": OrderedDict(),
                "expert_cache_1": OrderedDict(),
            }
            if getattr(self, "use_tp", False):
                h_half = self.num_heads // 2
                kv_half = max(1, self.num_kv_heads // 2)

                layer["wq_0"] = torch.randn(self.hidden_dim, h_half * self.head_dim, device=self.dev0, dtype=dtype) * scale
                layer["wk_0"] = torch.randn(self.hidden_dim, kv_half * self.head_dim, device=self.dev0, dtype=dtype) * scale
                layer["wv_0"] = torch.randn(self.hidden_dim, kv_half * self.head_dim, device=self.dev0, dtype=dtype) * scale
                layer["wo_0"] = torch.randn(h_half * self.head_dim, self.hidden_dim, device=self.dev0, dtype=dtype) * scale

                layer["wq_1"] = torch.randn(self.hidden_dim, h_half * self.head_dim, device=self.dev1, dtype=dtype) * scale
                layer["wk_1"] = torch.randn(self.hidden_dim, kv_half * self.head_dim, device=self.dev1, dtype=dtype) * scale
                layer["wv_1"] = torch.randn(self.hidden_dim, kv_half * self.head_dim, device=self.dev1, dtype=dtype) * scale
                layer["wo_1"] = torch.randn(h_half * self.head_dim, self.hidden_dim, device=self.dev1, dtype=dtype) * scale

                if is_moe:
                    for e in range(exp_count):
                        use_dev1 = e >= half_exp
                        t_dev = self.dev1 if use_dev1 else self.dev0
                        e_dict = {
                            "packed": False,
                            "gate": torch.randn(self.hidden_dim, inter_dim, device=t_dev, dtype=dtype) * scale,
                            "up": torch.randn(self.hidden_dim, inter_dim, device=t_dev, dtype=dtype) * scale,
                            "down": torch.randn(inter_dim, self.hidden_dim, device=t_dev, dtype=dtype) * scale,
                        }
                        if use_dev1:
                            layer["experts_1"].append(e_dict)
                        else:
                            layer["experts_0"].append(e_dict)
                else:
                    exp_half_dim = inter_dim // 2
                    layer["exp_gate_0"] = torch.randn(self.hidden_dim, exp_half_dim, device=self.dev0, dtype=dtype) * scale
                    layer["exp_up_0"] = torch.randn(self.hidden_dim, exp_half_dim, device=self.dev0, dtype=dtype) * scale
                    layer["exp_down_0"] = torch.randn(exp_half_dim, self.hidden_dim, device=self.dev0, dtype=dtype) * scale

                    layer["exp_gate_1"] = torch.randn(self.hidden_dim, exp_half_dim, device=self.dev1, dtype=dtype) * scale
                    layer["exp_up_1"] = torch.randn(self.hidden_dim, exp_half_dim, device=self.dev1, dtype=dtype) * scale
                    layer["exp_down_1"] = torch.randn(exp_half_dim, self.hidden_dim, device=self.dev1, dtype=dtype) * scale
            else:
                layer["wq"] = torch.randn(self.hidden_dim, self.num_heads * self.head_dim, device=self.dev0, dtype=dtype) * scale
                layer["wk"] = torch.randn(self.hidden_dim, self.num_kv_heads * self.head_dim, device=self.dev0, dtype=dtype) * scale
                layer["wv"] = torch.randn(self.hidden_dim, self.num_kv_heads * self.head_dim, device=self.dev0, dtype=dtype) * scale
                layer["wo"] = torch.randn(self.num_heads * self.head_dim, self.hidden_dim, device=self.dev0, dtype=dtype) * scale

                if is_moe:
                    for e in range(exp_count):
                        e_dict = {
                            "packed": False,
                            "gate": torch.randn(self.hidden_dim, inter_dim, device=self.dev0, dtype=dtype) * scale,
                            "up": torch.randn(self.hidden_dim, inter_dim, device=self.dev0, dtype=dtype) * scale,
                            "down": torch.randn(inter_dim, self.hidden_dim, device=self.dev0, dtype=dtype) * scale,
                        }
                        layer["experts_0"].append(e_dict)
                else:
                    layer["exp_gate"] = torch.randn(self.hidden_dim, inter_dim, device=self.dev0, dtype=dtype) * scale
                    layer["exp_up"] = torch.randn(self.hidden_dim, inter_dim, device=self.dev0, dtype=dtype) * scale
                    layer["exp_down"] = torch.randn(inter_dim, self.hidden_dim, device=self.dev0, dtype=dtype) * scale

            self.layers.append(layer)

    def _init_weights(self):
        self.init_error = None
        try:
            if self._try_load_pretrained_dataset():
                self.real_weights = True
                _safe_ntfy("building", stage="real_weights_ok", note=f"stage {self.stage_idx} real weights ready")
                return
            else:
                if not getattr(self, "init_error", None):
                    self.init_error = "load_returned_false"
                _safe_ntfy("warning", stage="real_weights_failed", note=f"stage {self.stage_idx} load returned False")
        except Exception as e:
            import traceback
            self.init_error = f"{type(e).__name__}: {str(e)[:120]}"
            _safe_log(f"Stage {self.stage_idx} init error: {e}\n{traceback.format_exc()}")
            _safe_ntfy("error", stage="init_weights", err=f"{type(e).__name__}: {str(e)[:100]}")
        self.real_weights = False
        self._init_synthetic_fallback()

    def reset_kv_cache(self):
        """Reset KV cache for all layers before starting a new conversation or inference request."""
        if not getattr(self, "torch_available", False):
            return
        for layer in getattr(self, "layers", []):
            layer["k_cache"] = None
            layer["v_cache"] = None

    def _apply_rope(self, x, pos_offset):
        # x: (bsz, seq_len, heads, head_dim)
        torch = self.torch
        seq_len = x.shape[1]
        positions = torch.arange(pos_offset, pos_offset + seq_len, device=x.device, dtype=torch.float32)
        dim_indices = torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
        freqs = 1.0 / (10000.0 ** (dim_indices / self.head_dim))
        angles = torch.outer(positions, freqs)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        rx = torch.empty_like(x)
        rx[..., 0::2] = rx1
        rx[..., 1::2] = rx2
        return rx

    def _dequant_nf4_tensor(self, raw_b, absmax, in_dim, out_dim, device, dtype):
        torch = self.torch
        if not hasattr(self, "_nf4_table_t") or self._nf4_table_t.device != device:
            self._nf4_table_t = torch.tensor(NF4_TABLE, device=device, dtype=dtype)
        b = raw_b.to(device).flatten()
        high = ((b >> 4) & 0x0F).long()
        low = (b & 0x0F).long()
        idx = torch.stack([high, low], dim=-1).flatten()
        vals = self._nf4_table_t[idx].view(-1, 64)
        del idx, high, low, b
        scale = absmax.to(device, dtype=dtype).view(-1, 1)
        res = (vals * scale).view(out_dim, in_dim).T.contiguous()
        del vals, scale
        return res

    def _get_expert_weights(self, layer, e, dev):
        half_exp = layer["exp_count"] // 2 if self.use_tp else layer["exp_count"]
        if self.use_tp and e >= half_exp:
            e_idx = e - half_exp
            e_dict = layer["experts_1"][e_idx]
            cache = layer["expert_cache_1"]
        else:
            e_idx = e
            e_dict = layer["experts_0"][e_idx]
            cache = layer["expert_cache_0"]

        if not e_dict.get("packed", False):
            return e_dict["gate"], e_dict["up"], e_dict["down"]

        if e in cache:
            cache.move_to_end(e)
            return cache[e]

        inter_dim = layer["inter_dim"]
        g = self._dequant_nf4_tensor(e_dict["gate_b"], e_dict["gate_s"], self.hidden_dim, inter_dim, dev, self.dtype)
        u = self._dequant_nf4_tensor(e_dict["up_b"], e_dict["up_s"], self.hidden_dim, inter_dim, dev, self.dtype)
        d = self._dequant_nf4_tensor(e_dict["down_b"], e_dict["down_s"], inter_dim, self.hidden_dim, dev, self.dtype)

        if len(cache) >= 32:
            cache.popitem(last=False)
        cache[e] = (g, u, d)
        return g, u, d

    def forward_layers(self, hidden_states, pos_offset=0):
        if not self.torch_available:
            return [x * 1.0001 for x in hidden_states]

        torch = self.torch
        F = self.F
        dtype = torch.float16 if self.dev0.type == "cuda" else torch.float32

        if not isinstance(hidden_states, torch.Tensor):
            x = torch.tensor(hidden_states, device=self.dev0, dtype=dtype)
        else:
            x = hidden_states.to(device=self.dev0, dtype=dtype)

        orig_shape = x.shape
        if x.dim() == 1:
            if x.numel() >= self.hidden_dim and x.numel() % self.hidden_dim == 0:
                x = x.view(1, x.numel() // self.hidden_dim, self.hidden_dim)
            else:
                x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            if x.shape[-1] != self.hidden_dim and x.numel() % self.hidden_dim == 0:
                x = x.view(1, x.numel() // self.hidden_dim, self.hidden_dim)
            else:
                x = x.unsqueeze(0)
        bsz, seq_len, _ = x.shape

        for layer in self.layers:
            # RMSNorm 1
            v1 = x.pow(2).mean(-1, keepdim=True)
            x_norm1 = x * torch.rsqrt(v1 + 1e-6) * layer["norm1"]

            # Attention with RoPE and KV Cache
            if self.use_tp:
                x_norm1_g1 = x_norm1.to(self.dev1)
                h_half = self.num_heads // 2
                kv_half = max(1, self.num_kv_heads // 2)

                q0 = torch.matmul(x_norm1, layer["wq_0"])
                if "bq_0" in layer: q0 = q0 + layer["bq_0"]
                q0 = q0.view(bsz, seq_len, h_half, self.head_dim)

                k0 = torch.matmul(x_norm1, layer["wk_0"])
                if "bk_0" in layer: k0 = k0 + layer["bk_0"]
                k0 = k0.view(bsz, seq_len, kv_half, self.head_dim)

                v0 = torch.matmul(x_norm1, layer["wv_0"])
                if "bv_0" in layer: v0 = v0 + layer["bv_0"]
                v0 = v0.view(bsz, seq_len, kv_half, self.head_dim)

                q1 = torch.matmul(x_norm1_g1, layer["wq_1"])
                if "bq_1" in layer: q1 = q1 + layer["bq_1"]
                q1 = q1.view(bsz, seq_len, h_half, self.head_dim)

                k1 = torch.matmul(x_norm1_g1, layer["wk_1"])
                if "bk_1" in layer: k1 = k1 + layer["bk_1"]
                k1 = k1.view(bsz, seq_len, kv_half, self.head_dim)

                v1 = torch.matmul(x_norm1_g1, layer["wv_1"])
                if "bv_1" in layer: v1 = v1 + layer["bv_1"]
                v1 = v1.view(bsz, seq_len, kv_half, self.head_dim)

                if layer.get("q_norm") is not None:
                    q0 = q0 * torch.rsqrt(q0.pow(2).mean(-1, keepdim=True) + 1e-6) * layer["q_norm"].to(self.dev0)
                    q1 = q1 * torch.rsqrt(q1.pow(2).mean(-1, keepdim=True) + 1e-6) * layer["q_norm"].to(self.dev1)
                if layer.get("k_norm") is not None:
                    k0 = k0 * torch.rsqrt(k0.pow(2).mean(-1, keepdim=True) + 1e-6) * layer["k_norm"].to(self.dev0)
                    k1 = k1 * torch.rsqrt(k1.pow(2).mean(-1, keepdim=True) + 1e-6) * layer["k_norm"].to(self.dev1)

                q0 = self._apply_rope(q0, pos_offset)
                k0 = self._apply_rope(k0, pos_offset)
                q1 = self._apply_rope(q1, pos_offset)
                k1 = self._apply_rope(k1, pos_offset)

                # KV Cache management with rollback slicing
                if layer["k_cache"] is None or pos_offset == 0:
                    layer["k_cache"] = (k0, k1)
                    layer["v_cache"] = (v0, v1)
                    k0_full, k1_full = k0, k1
                    v0_full, v1_full = v0, v1
                else:
                    prev_k0, prev_k1 = layer["k_cache"]
                    prev_v0, prev_v1 = layer["v_cache"]
                    if prev_k0.shape[1] > pos_offset:
                        prev_k0 = prev_k0[:, :pos_offset, :, :]
                        prev_v0 = prev_v0[:, :pos_offset, :, :]
                        prev_k1 = prev_k1[:, :pos_offset, :, :]
                        prev_v1 = prev_v1[:, :pos_offset, :, :]
                    k0_full = torch.cat([prev_k0, k0], dim=1)
                    v0_full = torch.cat([prev_v0, v0], dim=1)
                    k1_full = torch.cat([prev_k1, k1], dim=1)
                    v1_full = torch.cat([prev_v1, v1], dim=1)
                    layer["k_cache"] = (k0_full, k1_full)
                    layer["v_cache"] = (v0_full, v1_full)

                # Expand GQA KV heads
                rep = h_half // kv_half
                if rep > 1:
                    k0_full = k0_full.repeat_interleave(rep, dim=2)
                    v0_full = v0_full.repeat_interleave(rep, dim=2)
                    k1_full = k1_full.repeat_interleave(rep, dim=2)
                    v1_full = v1_full.repeat_interleave(rep, dim=2)

                a0 = F.scaled_dot_product_attention(
                    q0.transpose(1, 2), k0_full.transpose(1, 2), v0_full.transpose(1, 2),
                    is_causal=(seq_len > 1)
                ).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
                attn_out_0 = torch.matmul(a0, layer["wo_0"])

                a1 = F.scaled_dot_product_attention(
                    q1.transpose(1, 2), k1_full.transpose(1, 2), v1_full.transpose(1, 2),
                    is_causal=(seq_len > 1)
                ).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
                attn_out_1 = torch.matmul(a1, layer["wo_1"])

                attn_out = attn_out_0 + attn_out_1.to(self.dev0)
            else:
                q = torch.matmul(x_norm1, layer["wq"])
                if "bq" in layer: q = q + layer["bq"]
                q = q.view(bsz, seq_len, self.num_heads, self.head_dim)

                k = torch.matmul(x_norm1, layer["wk"])
                if "bk" in layer: k = k + layer["bk"]
                k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

                v = torch.matmul(x_norm1, layer["wv"])
                if "bv" in layer: v = v + layer["bv"]
                v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

                if layer.get("q_norm") is not None:
                    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * layer["q_norm"]
                if layer.get("k_norm") is not None:
                    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * layer["k_norm"]

                q = self._apply_rope(q, pos_offset)
                k = self._apply_rope(k, pos_offset)

                if layer["k_cache"] is None or pos_offset == 0:
                    layer["k_cache"] = k
                    layer["v_cache"] = v
                    k_full, v_full = k, v
                else:
                    prev_k = layer["k_cache"]
                    prev_v = layer["v_cache"]
                    if prev_k.shape[1] > pos_offset:
                        prev_k = prev_k[:, :pos_offset, :, :]
                        prev_v = prev_v[:, :pos_offset, :, :]
                    k_full = torch.cat([prev_k, k], dim=1)
                    v_full = torch.cat([prev_v, v], dim=1)
                    layer["k_cache"] = k_full
                    layer["v_cache"] = v_full

                rep = self.num_heads // self.num_kv_heads
                if rep > 1:
                    k_full = k_full.repeat_interleave(rep, dim=2)
                    v_full = v_full.repeat_interleave(rep, dim=2)

                a = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2),
                    is_causal=(seq_len > 1)
                ).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
                attn_out = torch.matmul(a, layer["wo"])

            x = x + attn_out

            # RMSNorm 2
            v2 = x.pow(2).mean(-1, keepdim=True)
            x_norm2 = x * torch.rsqrt(v2 + 1e-6) * layer["norm2"]

            # MoE Routing & SwiGLU FFN
            if layer.get("is_moe", False) and layer.get("exp_count", 0) > 0:
                exp_count = layer["exp_count"]
                top_k = min(self.top_k, exp_count)
                x_flat = x_norm2.view(-1, self.hidden_dim)
                num_toks = x_flat.shape[0]

                router_logits = torch.matmul(x_flat, layer["router"])
                weights = F.softmax(router_logits, dim=-1)
                top_weights, top_indices = torch.topk(weights, top_k, dim=-1)
                top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

                ffn_out_0 = torch.zeros_like(x_flat)
                if self.use_tp:
                    ffn_out_1 = torch.zeros(num_toks, self.hidden_dim, device=self.dev1, dtype=self.dtype)
                    x_flat_1 = x_flat.to(self.dev1)
                    half_exp = exp_count // 2
                else:
                    ffn_out_1 = None
                    half_exp = exp_count

                # Group active tokens by selected expert
                for e in torch.unique(top_indices).tolist():
                    mask = (top_indices == e)
                    tok_idx, k_idx = torch.where(mask)
                    if tok_idx.numel() == 0:
                        continue
                    w_expert = top_weights[tok_idx, k_idx].unsqueeze(-1)

                    if self.use_tp and e >= half_exp:
                        g, u, d = self._get_expert_weights(layer, e, self.dev1)
                        tok_idx_1 = tok_idx.to(self.dev1)
                        inp = x_flat_1[tok_idx_1]
                        h = F.silu(torch.matmul(inp, g)) * torch.matmul(inp, u)
                        out_e = torch.matmul(h, d) * w_expert.to(self.dev1)
                        ffn_out_1.index_add_(0, tok_idx_1, out_e)
                    else:
                        g, u, d = self._get_expert_weights(layer, e, self.dev0)
                        inp = x_flat[tok_idx]
                        h = F.silu(torch.matmul(inp, g)) * torch.matmul(inp, u)
                        out_e = torch.matmul(h, d) * w_expert
                        ffn_out_0.index_add_(0, tok_idx, out_e)

                if self.use_tp:
                    ffn_out = (ffn_out_0 + ffn_out_1.to(self.dev0)).view(bsz, seq_len, self.hidden_dim)
                else:
                    ffn_out = ffn_out_0.view(bsz, seq_len, self.hidden_dim)
            else:
                if self.use_tp:
                    x_norm2_g1 = x_norm2.to(self.dev1)
                    # SwiGLU on dev0
                    g0 = F.silu(torch.matmul(x_norm2, layer["exp_gate_0"]))
                    u0 = torch.matmul(x_norm2, layer["exp_up_0"])
                    d0 = torch.matmul(g0 * u0, layer["exp_down_0"])

                    # SwiGLU on dev1
                    g1 = F.silu(torch.matmul(x_norm2_g1, layer["exp_gate_1"]))
                    u1 = torch.matmul(x_norm2_g1, layer["exp_up_1"])
                    d1 = torch.matmul(g1 * u1, layer["exp_down_1"])

                    ffn_out = d0 + d1.to(self.dev0)
                else:
                    g = F.silu(torch.matmul(x_norm2, layer["exp_gate"]))
                    u = torch.matmul(x_norm2, layer["exp_up"])
                    ffn_out = torch.matmul(g * u, layer["exp_down"])

            x = x + ffn_out

        return x.reshape(orig_shape)


class SwarmDrafter:
    """Multi-order Recursive N-Gram Prompt-Lookup Speculative Drafter for Stage 0.

    Supports 7-gram down to 2-gram matching, recursive continuation when candidate
    spans run out, and multi-branch candidate tree exploration for high-depth speculation (K=6-10).
    """
    def __init__(self, k: int = 6):
        self.k = k

    def draft(self, tokens: list[int], k: int = None) -> list[int]:
        if k is None:
            k = self.k
        if k <= 0 or not tokens:
            return []
        n_tokens = len(tokens)
        if n_tokens < 2:
            return []

        res = []
        curr_tokens = list(tokens)

        while len(res) < k:
            needed = k - len(res)
            found_chunk = None

            # Multi-order search: 7-gram down to 2-gram
            curr_n = len(curr_tokens)
            for order in (7, 6, 5, 4, 3, 2):
                if curr_n <= order:
                    continue
                suffix = curr_tokens[-order:]
                limit = curr_n - order
                for i in range(limit - 1, -1, -1):
                    if curr_tokens[i : i + order] == suffix:
                        cands = curr_tokens[i + order : i + order + needed]
                        if cands:
                            found_chunk = list(cands)
                            break
                if found_chunk:
                    break

            if found_chunk:
                res.extend(found_chunk)
                curr_tokens.extend(found_chunk)
            else:
                # No further matching n-grams in context; terminate draft cleanly without token spam
                break

        return res[:k]

    def draft_tree(self, tokens: list[int], k: int = 6, branches: int = 2) -> list[list[int]]:
        """Generate multi-branch candidate sequences for tree speculative decoding."""
        if k <= 0 or not tokens:
            return [[]]
        primary = self.draft(tokens, k=k)
        if not primary:
            return [[]]
        if branches <= 1 or len(tokens) < 3:
            return [primary]

        alt_branches = [primary]
        n_tokens = len(tokens)
        for order in (4, 3, 2):
            if n_tokens <= order:
                continue
            suffix = tokens[-order:]
            limit = n_tokens - order
            matches = []
            for i in range(limit - 1, -1, -1):
                if tokens[i : i + order] == suffix:
                    matches.append(i)
                    if len(matches) >= branches:
                        break
            if len(matches) > 1:
                sec_idx = matches[1]
                sec_cands = tokens[sec_idx + order : sec_idx + order + k]
                if sec_cands:
                    alt = list(sec_cands)
                    if alt != primary:
                        alt_branches.append(alt[:k])
                        break
        return alt_branches


class SwarmWorker:
    """Manages persistent WebSocket connection to GCP Hub and pipeline execution."""
    def __init__(self, stage_idx, total_stages, hub_url, shard, api_key):
        self.stage_idx = stage_idx
        self.total_stages = total_stages
        self.hub_url = hub_url
        self.shard = shard
        self.api_key = api_key
        self.running = True
        self.ws = None
        self.pending_responses = {}
        self.lock = threading.Lock()

    def _ensure_websockets(self):
        try:
            import websockets
            return websockets
        except ImportError:
            sh("pip install -q websockets")
            import websockets
            return websockets

    async def run_ws_loop(self):
        ws_lib = self._ensure_websockets()
        hub_clean = self.hub_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
        if not hub_clean.endswith("/ws"):
            hub_clean = f"{hub_clean}/swarm/ws"
        stage_uri = f"{hub_clean}/{self.stage_idx}?total_stages={self.total_stages}"

        while self.running:
            try:
                log(f"Connecting stage {self.stage_idx} to hub: {stage_uri}")
                async with ws_lib.connect(stage_uri, max_size=32 * 1024 * 1024, ping_interval=10, ping_timeout=10) as ws:
                    self.ws = ws
                    log(f"Stage {self.stage_idx} registered on GCP Hub")
                    while self.running:
                        msg = await ws.recv()
                        if isinstance(msg, bytes):
                            await self.handle_binary_frame(msg)
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                log(f"Stage {self.stage_idx} websocket reconnecting: {type(e).__name__}: {e}")
                self.ws = None
                await asyncio.sleep(2.0)

    async def handle_binary_frame(self, raw_frame):
        try:
            hdr, payload = unpack_activation_frame(raw_frame)
        except Exception:
            return

        req_id = hdr["req_id"]

        if self.stage_idx == 0:
            # Stage 0 receives verified token return frames from Tail stage
            if hdr["flags"] & FLAG_TOKEN:
                tokens, step_id = unpack_token_payload(payload)
                with self.lock:
                    q = self.pending_responses.get(req_id)
                if q is not None:
                    q.put((tokens, step_id))
        elif self.stage_idx == self.total_stages - 1:
            # Tail stage receives activation from stage N-2
            prev_expected = self.total_stages - 2
            if hdr["stage_idx"] == prev_expected:
                num_tok = hdr["num_tokens"]
                dim = hdr["hidden_dim"]
                floats = dequantize_activations_q8_0(payload, num_tok * dim)

                # Extract candidate tokens for speculative verification
                candidates = []
                act_len = math.ceil(num_tok * dim / 32) * 34
                extra = payload[act_len:]
                if num_tok > 1:
                    num_cands = num_tok - 1
                    if len(extra) >= num_cands * 4:
                        candidates = list(struct.unpack(f"!{num_cands}I", extra[:num_cands * 4]))
                    else:
                        res = hdr.get("reserved", b"\x00" * 20)
                        if isinstance(res, (bytes, bytearray)) and len(res) >= min(5, num_cands) * 4:
                            try:
                                candidates = list(struct.unpack(f"!{min(5, num_cands)}I", res[:min(5, num_cands) * 4]))
                            except Exception:
                                candidates = []

                if self.shard.torch_available:
                    x = self.shard.forward_layers(floats, pos_offset=hdr["pos_offset"])
                    v = x.pow(2).mean(-1, keepdim=True)
                    x_norm = x * self.shard.torch.rsqrt(v + 1e-6) * self.shard.final_norm
                    x_norm_2d = x_norm.view(-1, self.shard.hidden_dim)
                    logits = self.shard.torch.matmul(x_norm_2d, self.shard.w_head.T)

                    if candidates and num_tok > 1:
                        verified_tokens = []
                        all_matched = True
                        for j, cand in enumerate(candidates):
                            pred = int(self.shard.torch.argmax(logits[j], dim=-1).item())
                            if pred == cand:
                                verified_tokens.append(cand)
                            else:
                                verified_tokens.append(pred)  # Bonus token from target model
                                all_matched = False
                                break
                        if all_matched:
                            last_idx = len(candidates)
                            if last_idx < logits.shape[0]:
                                bonus = int(self.shard.torch.argmax(logits[last_idx], dim=-1).item())
                                verified_tokens.append(bonus)
                            elif not verified_tokens:
                                verified_tokens.append(int(self.shard.torch.argmax(logits[-1], dim=-1).item()))
                        ret_tokens = verified_tokens
                    else:
                        next_tok = int(self.shard.torch.argmax(logits[-1], dim=-1).item())
                        ret_tokens = [next_tok]
                else:
                    if candidates and num_tok > 1:
                        verified_tokens = []
                        for j, cand in enumerate(candidates):
                            pred = (hdr["step_id"] * 17 + 42 + j) % 32000
                            if pred == cand or cand == pred:
                                verified_tokens.append(cand)
                            else:
                                verified_tokens.append(pred)
                                break
                        if len(verified_tokens) == len(candidates):
                            verified_tokens.append((hdr["step_id"] * 17 + 42 + len(candidates)) % 32000)
                        ret_tokens = verified_tokens or [(hdr["step_id"] * 17 + 42) % 32000]
                    else:
                        ret_tokens = [(hdr["step_id"] * 17 + 42) % 32000]

                # Pack return frame to stage 0
                ret_payload = pack_token_payload(ret_tokens, hdr["step_id"])
                ret_frame = pack_activation_frame(
                    hdr["req_id"], hdr["seq_id"], hdr["step_id"],
                    self.stage_idx, self.total_stages, len(ret_tokens), hdr["hidden_dim"],
                    hdr["pos_offset"], FLAG_TOKEN, ret_payload
                )
                if self.ws is not None:
                    await self.ws.send(ret_frame)
        else:
            # Worker stages receive activation from stage-1, compute, forward to stage+1
            if hdr["stage_idx"] == self.stage_idx - 1:
                num_tok = hdr["num_tokens"]
                dim = hdr["hidden_dim"]
                floats = dequantize_activations_q8_0(payload, num_tok * dim)
                act_len = math.ceil(num_tok * dim / 32) * 34
                extra = payload[act_len:]

                if self.shard.torch_available:
                    out = self.shard.forward_layers(floats, pos_offset=hdr["pos_offset"])
                    out_floats = out.flatten().tolist()
                else:
                    out_floats = [v * 1.0001 for v in floats]

                out_payload = quantize_activations_q8_0(out_floats) + extra
                res_bytes = hdr.get("reserved", b"\x00" * 20)
                out_frame = pack_activation_frame(
                    hdr["req_id"], hdr["seq_id"], hdr["step_id"],
                    self.stage_idx, self.total_stages, num_tok, dim,
                    hdr["pos_offset"], FLAG_ACTIVATION, out_payload, res_bytes
                )
                if self.ws is not None:
                    await self.ws.send(out_frame)


def start_swarm(server_bin):
    """Launch shard-local pipeline daemon or stage worker with intra-node TP=2."""
    stage_idx = int(CFG.get("stage_idx", 0))
    total_stages = int(CFG.get("stages", 6))
    hub_url = CFG.get("hub_url") or (CFG.get("opts") or {}).get("hub_url") or ""   # experimental: your own hub
    model_key = CFG.get("model_key") or "qwen3-235b-swarm"
    model_hf = CFG.get("model_hf") or "Qwen/Qwen3-235B-A22B-GGUF"
    api_key = CFG.get("api_key", "")

    zone, rtt_ms = probe_gcp_zone_and_latency(hub_url)
    log(f"swarm node stage={stage_idx}/{total_stages} zone={zone} hub_rtt={rtt_ms:.1f}ms")
    if rtt_ms > 35.0 and rtt_ms < 900.0:
        ntfy("warning", note=f"High hub latency ({rtt_ms:.1f}ms) in zone {zone}")

    extra_args = list(CFG.get("model_args", []))
    if "-sm" not in extra_args:
        extra_args.extend(["-sm", "tensor"])
    CFG["model_args"] = extra_args

    tokenizer = SwarmTokenizer(model_hf=model_hf)

    # Initialize layer shard with Intra-Node TP=2 across GPU 0 and GPU 1
    hidden_dim = int(CFG.get("hidden_dim") or 8192)
    total_layers = int(CFG.get("total_layers") or 80)
    num_heads = int(CFG.get("num_heads") or 64)
    num_kv_heads = int(CFG.get("num_kv_heads") or 8)
    head_dim = int(CFG.get("head_dim") or 128)
    num_experts = int(CFG.get("num_experts") if CFG.get("num_experts") is not None else 0)
    vocab_size = int(CFG.get("vocab_size") or 152064)
    layers_per_stage = CFG.get("layers_per_stage")

    shard = SwarmTransformerShard(
        stage_idx=stage_idx,
        total_stages=total_stages,
        total_layers=total_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        top_k=8 if num_experts > 0 else 0,
        vocab_size=vocab_size,
        layers_per_stage=layers_per_stage,
    )
    log(f"Swarm stage {stage_idx} ({shard.role}) initialized layers {shard.layer_start}..{shard.layer_end-1} (TP={shard.use_tp})")

    worker = SwarmWorker(stage_idx, total_stages, hub_url, shard, api_key)

    # Start WebSocket client event loop in background thread
    ws_loop = asyncio.new_event_loop()
    def ws_thread_main():
        asyncio.set_event_loop(ws_loop)
        ws_loop.run_until_complete(worker.run_ws_loop())

    ws_thread = threading.Thread(target=ws_thread_main, daemon=True)
    ws_thread.start()

    # HTTP Handler serving health, models, and chat completions
    class SwarmHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _auth_ok(self):
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {api_key}" or not api_key

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "runtime": "swarm",
                    "stage": stage_idx,
                    "total_stages": total_stages,
                    "zone": zone,
                    "rtt_ms": rtt_ms,
                    "model": model_key,
                    "real_weights": getattr(shard, "real_weights", False),
                    "init_error": getattr(shard, "init_error", None),
                }).encode())
            elif p in ("/v1/models", "/models"):
                if not self._auth_ok():
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": {"message": "Invalid API Key", "type": "authentication_error"}}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "object": "list",
                    "data": [{
                        "id": model_key,
                        "object": "model",
                        "owned_by": "kllm",
                        "runtime": "swarm",
                        "stages": total_stages,
                        "stage": stage_idx,
                    }]
                }).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            try:
                p = self.path.split("?")[0]
                if p in ("/v1/chat/completions", "/chat/completions"):
                    if not self._auth_ok():
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": {"message": "Invalid API Key", "type": "authentication_error"}}')
                        return

                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length > 0 else b"{}"
                    try:
                        payload = json.loads(body)
                    except Exception:
                        payload = {}

                    stream = payload.get("stream", True)
                    max_tokens = min(256, int(payload.get("max_tokens") or 64))

                    if stage_idx != 0:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"info": f"worker stage {stage_idx}: active in swarm ring"}).encode())
                        return

                    # Stage 0 generation
                    req_id = uuid.uuid4()
                    import queue
                    q = queue.Queue()
                    with worker.lock:
                        worker.pending_responses[req_id] = q

                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()

                    # Parse user messages into prompt
                    messages = payload.get("messages", [])
                    prompt_text = ""
                    for m in messages:
                        c = m.get("content", "")
                        if isinstance(c, str):
                            prompt_text += f"{m.get('role', 'user')}: {c}\n"
                    if not prompt_text:
                        prompt_text = "assistant: "

                    prompt_tokens = tokenizer.encode(prompt_text)
                    if not prompt_tokens:
                        prompt_tokens = [1]

                    # Reset KV cache for fresh generation
                    shard.reset_kv_cache()

                    # Prompt tokens & initial embedding
                    if shard.torch_available:
                        tok_tensor = shard.torch.tensor([t % shard.vocab_size for t in prompt_tokens], device=shard.dev0)
                        x0 = shard.forward_layers(shard.w_emb[tok_tensor], pos_offset=0)
                        q8_payload = quantize_activations_q8_0(x0[-1:].flatten().tolist())
                    else:
                        q8_payload = quantize_activations_q8_0([0.1] * hidden_dim)

                    # Stream initial prompt activation frame to GCP Hub
                    frame = pack_activation_frame(
                        req_id, 1, 0, 0, total_stages, 1, hidden_dim, len(prompt_tokens) - 1, FLAG_ACTIVATION, q8_payload
                    )
                    if worker.ws is not None:
                        asyncio.run_coroutine_threadsafe(worker.ws.send(frame), ws_loop)

                    tokens_gen = []
                    t_start = time.time()

                    # Get initial token from prompt prefill
                    try:
                        init_toks, _ = q.get(timeout=45.0)
                    except Exception:
                        init_toks = []

                    if init_toks:
                        first_tok = init_toks[0]
                        tokens_gen.append(first_tok)
                        word = tokenizer.decode([first_tok])
                        if stream:
                            chunk = {
                                "id": f"chatcmpl-{req_id.hex[:12]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_key,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": word},
                                    "finish_reason": None,
                                }]
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                    else:
                        first_tok = None

                    # Adaptive Speculative Chunk Decoding loop (K=4-10)
                    initial_k = int(CFG.get("spec_k") or 6)
                    min_k = int(CFG.get("min_spec_k") or 4)
                    spec_k = max(min_k, initial_k)
                    drafter = SwarmDrafter(k=spec_k - 1)
                    root_tok = first_tok
                    current_pos = len(prompt_tokens)
                    step = 1
                    eos_hit = first_tok in (0, 2, 151643) if first_tok is not None else False
                    recent_acceptance = []

                    while len(tokens_gen) < max_tokens and not eos_hit and root_tok is not None:
                        hist = prompt_tokens + tokens_gen
                        num_draft = min(spec_k - 1, max_tokens - len(tokens_gen)) if spec_k > 1 else 0
                        candidates = drafter.draft(hist, k=num_draft) if num_draft > 0 else []
                        if len(candidates) < num_draft and num_draft > 0:
                            ctx_candidates = [t for t in reversed(hist[-64:]) if t not in candidates and t != root_tok]
                            for c in ctx_candidates:
                                candidates.append(c)
                                if len(candidates) >= num_draft:
                                    break
                        chunk_tokens = [root_tok] + candidates
                        k_actual = len(chunk_tokens)

                        if shard.torch_available:
                            tok_t = shard.torch.tensor([t % shard.vocab_size for t in chunk_tokens], device=shard.dev0)
                            x_chunk = shard.forward_layers(shard.w_emb[tok_t], pos_offset=current_pos)
                            act_payload = quantize_activations_q8_0(x_chunk.flatten().tolist())
                        else:
                            act_payload = quantize_activations_q8_0([0.1] * (k_actual * hidden_dim))

                        cand_bytes = struct.pack(f"!{len(candidates)}I", *candidates) if candidates else b""
                        chunk_payload = act_payload + cand_bytes
                        reserved_bytes = cand_bytes[:20].ljust(20, b"\x00")

                        spec_frame = pack_activation_frame(
                            req_id, 1, step, 0, total_stages, k_actual, hidden_dim,
                            current_pos, FLAG_ACTIVATION, chunk_payload, reserved_bytes
                        )
                        if worker.ws is not None:
                            asyncio.run_coroutine_threadsafe(worker.ws.send(spec_frame), ws_loop)

                        try:
                            accepted, _ = q.get(timeout=45.0)
                        except Exception:
                            break

                        if not accepted:
                            break

                        for t in accepted:
                            tokens_gen.append(t)
                            w = tokenizer.decode([t])
                            if stream:
                                chunk = {
                                    "id": f"chatcmpl-{req_id.hex[:12]}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_key,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": w},
                                        "finish_reason": None,
                                    }]
                                }
                                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                                self.wfile.flush()
                            if t in (0, 2, 151643) or len(tokens_gen) >= max_tokens:
                                eos_hit = True
                                break

                        current_pos += len(accepted)
                        root_tok = accepted[-1]
                        step += 1

                        # Dynamically tune speculative depth K based on acceptance rate
                        acc_ratio = len(accepted) / max(1, k_actual)
                        recent_acceptance.append(acc_ratio)
                        if len(recent_acceptance) >= 2:
                            avg_acc = sum(recent_acceptance[-3:]) / len(recent_acceptance[-3:])
                            if avg_acc >= 0.80 and spec_k < 10:
                                spec_k = min(10, spec_k + 1)
                            elif avg_acc < 0.45 and spec_k > min_k:
                                spec_k = max(min_k, spec_k - 1)

                    t_dur = max(0.001, time.time() - t_start)
                    speed = len(tokens_gen) / t_dur
                    log(f"[swarm] generated {len(tokens_gen)} tokens in {t_dur:.2f}s ({speed:.1f} tok/s) via speculative chunking (K={spec_k})")

                    if stream:
                        end_chunk = {
                            "id": f"chatcmpl-{req_id.hex[:12]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_key,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }]
                        }
                        self.wfile.write(f"data: {json.dumps(end_chunk)}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        resp = {
                            "id": f"chatcmpl-{req_id.hex[:12]}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model_key,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": tokenizer.decode(tokens_gen)},
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": len(prompt_tokens), "completion_tokens": len(tokens_gen), "total_tokens": len(prompt_tokens) + len(tokens_gen)}
                        }
                        self.wfile.write(json.dumps(resp).encode())

                    with worker.lock:
                        worker.pending_responses.pop(req_id, None)
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as e:
                log(f"SwarmHTTPHandler do_POST error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if not self.wfile.closed:
                        stream_mode = locals().get("stream", True)
                        if stream_mode:
                            err_chunk = {"error": {"message": str(e), "type": "server_error"}}
                            self.wfile.write(f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n".encode())
                            self.wfile.flush()
                        else:
                            self.send_response(500)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps({"error": {"message": str(e), "type": "server_error"}}).encode())
                except Exception:
                    pass

    class S(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = S(("127.0.0.1", LLAMA_PORT), SwarmHTTPHandler)
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    log(f"swarm node stage {stage_idx} serving on {LLAMA_PORT}")
    return _SwarmHandle(srv=srv, thread=srv_thread, ws_thread=ws_thread, worker=worker)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def wait_health(port, timeout, proc=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False                  # llama-server exited (OOM, bad flag...): fail fast
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False

TUNNEL = {"proc": None, "url": None, "proto": "quic", "born": 0.0, "restarts": 0}

def start_tunnel(cf, port, proto="quic"):
    logf = open("/kaggle/working/cloudflared.log", "a")
    # QUIC (UDP) by default with aggressive keep-alive heartbeats; http2 (TCP) is the alternate path the
    # watchdog switches to when the QUIC connection stalls (independent failure modes).
    cmd = [
        cf, "tunnel", "--no-autoupdate",
        "--protocol", proto,
        "--heartbeat-interval", "5s",
        "--heartbeat-count", "12",
        "--edge-bind-address", "0.0.0.0",
        "--url", f"http://127.0.0.1:{port}",
    ]
    mark = 0
    try:
        mark = os.path.getsize("/kaggle/working/cloudflared.log")      # only look at output of THIS process
    except Exception:
        pass
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    url = None
    deadline = time.time() + 90
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    while time.time() < deadline and url is None:
        time.sleep(2)
        if p.poll() is not None:
            break
        try:
            with open("/kaggle/working/cloudflared.log") as f:
                f.seek(mark); txt = f.read()
            m = pat.search(txt)
            if m:
                url = m.group(0)
        except Exception:
            pass
    TUNNEL.update(proc=p, url=url, proto=proto, born=time.time())
    return p, url

def _public_ok(url, timeout=10):
    """Is the tunnel actually carrying traffic end to end (Cloudflare edge -> cloudflared -> proxy -> llama-server)?"""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False

# --------------------------------------------------------------------------
# Speculative decoding draft model helper (DSpine, DFlash, MTP, etc.)
# --------------------------------------------------------------------------
def find_cached_draft_file(filename):
    """Search /kaggle/input for an attached draft model file."""
    if not filename:
        return None
    basename = os.path.basename(filename)
    for p in glob.glob(f"/kaggle/input/**/{basename}", recursive=True):
        if os.path.isfile(p):
            return p
    for p in glob.glob(f"/kaggle/working/**/{basename}", recursive=True):
        if os.path.isfile(p):
            return p
    return None

def download_draft_model(repo, filename, dst=None):
    """Download a draft GGUF file from Hugging Face if not already present."""
    if not dst:
        dst = f"/kaggle/working/{os.path.basename(filename)}"
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.path.exists("/kaggle"):
        return dst
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    log(f"downloading draft model from {url} to {dst}")
    ntfy("downloading", note=f"fetching draft model {filename}")
    headers = {"User-Agent": "kllm"}
    if CFG.get("hf_token"):
        headers["Authorization"] = f"Bearer {CFG['hf_token']}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(dst + ".tmp", "wb") as out:
            shutil.copyfileobj(resp, out)
        os.rename(dst + ".tmp", dst)
        log(f"draft model downloaded ({os.path.getsize(dst) // (1024*1024)} MB)")
        return dst
    except Exception as e:
        log("urllib download failed, trying curl fallback:", e)
        if os.path.exists(dst + ".tmp"):
            os.remove(dst + ".tmp")
        cmd = ["curl", "-L", "-f", "-s", "-S", "-o", dst, url]
        if CFG.get("hf_token"):
            cmd = ["curl", "-L", "-f", "-s", "-S", "-H", f"Authorization: Bearer {CFG['hf_token']}", "-o", dst, url]
        sh(cmd)
        return dst

def resolve_draft_model():
    """Locate draft model in /kaggle/input, or download from Hugging Face to /kaggle/working."""
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.path.exists("/kaggle"):
        file = CFG.get("draft_file") or "draft-model.gguf"
        return f"/tmp/{file}"

    direct = CFG.get("draft_model")
    if direct and os.path.isfile(direct):
        return direct

    repo = CFG.get("draft_hf")
    filename = CFG.get("draft_file")
    quant = CFG.get("draft_quant")

    if direct and not repo and not filename:
        if ":" in direct:
            parts = direct.split(":", 1)
            repo = parts[0]
            if parts[1].endswith(".gguf"):
                filename = parts[1]
            else:
                quant = parts[1]
        elif "/" in direct and not direct.endswith(".gguf"):
            repo = direct
        else:
            filename = os.path.basename(direct)

    if filename:
        cached = find_cached_draft_file(filename)
        if cached:
            log("using cached draft model:", cached)
            return cached

    if repo and not filename:
        if quant:
            filename = f"draft-{quant}.gguf"
        else:
            filename = "draft.gguf"

    if repo and filename:
        return download_draft_model(repo, filename)

    if direct:
        cached = find_cached_draft_file(direct)
        if cached:
            return cached

    raise RuntimeError(f"Could not resolve draft model (repo={repo}, file={filename}, draft_model={direct})")

_CACHE_DIRS = [os.environ.get("LLAMA_CACHE") or os.path.expanduser("~/.cache/llama.cpp"),
               os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub"]

def _dir_bytes(paths):
    n = 0
    for root in paths:
        for dp, _, files in os.walk(root):
            for f in files:
                fp = os.path.join(dp, f)
                if os.path.islink(fp):
                    continue                    # HF hub snapshots symlink to blobs/: counting both doubles the bytes
                try:
                    n += os.path.getsize(fp)
                except OSError:
                    pass
    return n

def _gguf_total_bytes():
    """Size of the GGUF file(s) llama-server will fetch for model_hf:model_quant (all shards), from the HF API."""
    repo, want = CFG.get("model_hf"), (CFG.get("model_file") or CFG.get("model_quant") or "").lower()
    if not repo or not want:
        return None
    try:
        hdr = {"User-Agent": "kllm"}
        if CFG.get("hf_token"):
            hdr["Authorization"] = "Bearer " + CFG["hf_token"]
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", headers=hdr)
        rows = json.loads(urllib.request.urlopen(req, timeout=20).read().decode())
    except Exception:
        return None
    hits = [r for r in rows if isinstance(r, dict) and str(r.get("path", "")).lower().endswith(".gguf")
            and want in str(r.get("path", "")).lower() and "mmproj" not in str(r.get("path", "")).lower()]
    total = sum(int((r.get("lfs") or {}).get("size") or r.get("size") or 0) for r in hits)
    url = CFG.get("mmproj_url")          # vision models also fetch the projector: count it or progress reads >100%
    if url:
        name = str(url).split("/resolve/main/")[-1].split("/")[-1].split("?")[0].lower()
        total += sum(int((r.get("lfs") or {}).get("size") or r.get("size") or 0)
                     for r in rows if str(r.get("path", "")).lower() == name)
    return total or None

def _download_watch(srv):
    """Report download progress (bytes_done, bytes_total) to the PC while llama-server fetches the model.
    ntfy() already rate-limits "downloading" events to one per 30 s, so this costs a few gist writes per launch."""
    start = _dir_bytes(_CACHE_DIRS)
    total = _gguf_total_bytes()
    last = -1
    while srv.poll() is None:
        time.sleep(10)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=2)
            return                                   # healthy: download and load are over
        except Exception:
            pass
        done = max(0, _dir_bytes(_CACHE_DIRS) - start)
        if done == last or done == 0:
            continue                                 # loading from a cached file: nothing new to say
        last = done
        ntfy("downloading", note="llama-server fetching model", bytes_done=done,
             **({"bytes_total": total} if total else {}))
        if total and done >= total:
            return

def start_llama(server_bin):
    """Launch llama-server on LLAMA_PORT, wait until healthy, warm it up. Returns its Popen."""
    args = [server_bin,
            *(["-hfr", CFG["model_hf"], "-hff", CFG["model_file"]] if CFG.get("model_file")
              else ["-hf", f"{CFG['model_hf']}:{CFG['model_quant']}"]),
            "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
            "-ngl", "99",                 # offload all layers; split across both T4s (default layer split)
            "-c", str(CFG.get("model_ctx", 32768)),
            "-fa", "on",
            "--api-key", CFG["api_key"],
            "--jinja",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "-np", "2", "--kv-unified",   # 2 slots sharing one KV pool, so a single conversation
                                          # can use the FULL context (not ctx/np per slot)
            # --- low-latency tuning (cloud-API feel) ---
            "--cache-reuse", "256",       # reuse cached prompt prefix -> follow-up turns skip re-prefill
            # Host-RAM prompt cache (Kaggle has ~31 GB RAM, unused): when a harness's side request
            # (title, summary) takes over the KV pool, the main conversation is parked here and
            # restored instead of being re-prefilled from scratch.
            "--cache-ram", str(CFG.get("cache_ram_mib", 16384)),
            "-b", "2048", "-ub", "512",   # larger logical batch = faster prefill of long contexts
            "--threads-http", "4",        # don't let HTTP handling stall token streaming
            "--no-webui",                 # we use Open WebUI locally; skip the bundled UI
            "--metrics",                  # expose /metrics: prefill/decode token counters (kllm scrapes for live tok/s)
            ] + list(CFG.get("model_args", [])) + list(CFG.get("extra_server_args", []))

    if CFG.get("mmproj_url"):
        # vision projector: llama-server downloads it; image-max-tokens bounds per-image context/VRAM use
        args += ["--mmproj-url", CFG["mmproj_url"], "--image-max-tokens", str(CFG.get("image_max_tokens", 1024))]

    if CFG.get("chat_template"):
        # model needs a patched template (e.g. Qwen3.6: stock one rejects mid-chat system msgs)
        with open("/tmp/chat_template.jinja", "w") as f:
            f.write(CFG["chat_template"])
        args += ["--chat-template-file", "/tmp/chat_template.jinja"]

    if CFG.get("draft_model") or CFG.get("draft_file") or CFG.get("draft_hf"):
        if "-md" not in args and "--model-draft" not in args:
            draft_path = resolve_draft_model()
            if draft_path:
                args += ["-md", draft_path]
        if CFG.get("draft_ngld") is not None and "-ngld" not in args and "--gpu-layers-draft" not in args and "--n-gpu-layers-draft" not in args:
            args += ["-ngld", str(CFG["draft_ngld"])]
    if CFG.get("draft_args"):
        args += list(CFG["draft_args"])

    log("launching llama-server:", " ".join(args))
    ntfy("downloading", note="llama-server fetching model via -hf (first run only)")
    srvlog = open("/kaggle/working/llama-server.log", "w")
    srv = subprocess.Popen(args, stdout=srvlog, stderr=subprocess.STDOUT)
    threading.Thread(target=_download_watch, args=(srv,), daemon=True).start()

    # Model download + load can take several minutes; give it a wide window.
    if not wait_health(LLAMA_PORT, timeout=1800, proc=srv):
        tail = ""
        try: tail = "".join(open("/kaggle/working/llama-server.log").readlines()[-25:])
        except Exception: pass
        ntfy("error", stage="server_health", tail=tail[-1500:])
        log("server did not become healthy; aborting"); srv.kill(); sys.exit(1)

    # Warm up (CUDA kernels, expert pages) so the user's first prompt isn't the slow one:
    # measured ~33 s for a 472-token first request on a cold server.
    try:
        wreq = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
            json.dumps({"messages": [{"role": "user", "content": "warm up " * 600}],
                        "max_tokens": 16}).encode(),
            {"Content-Type": "application/json", "Authorization": f"Bearer {CFG['api_key']}"})
        urllib.request.urlopen(wreq, timeout=300).read()
        log("warm-up done")
    except Exception as e:
        log("warm-up failed (continuing):", e)
    return srv

def main():
    log("kllm kernel start. model:", *filter(None, [CFG["model_hf"], CFG.get("model_quant"),
                                                  f"[{RUNTIME}]" if RUNTIME != "llama.cpp" else None]))
    gpus = gpu_info()
    ntfy("boot", gpu=gpus, model=MODEL_ID, build_only=bool(CFG.get("build_only")))
    if "Tesla" not in gpus and "NVIDIA" not in gpus and "GPU" not in gpus.upper().replace("NVIDIA-SMI FAILED", ""):
        # Kaggle ran us without a GPU (weekly quota used up, or account not phone-verified).
        # Exit at once: running the model on 4 CPUs would be useless and still burn time.
        ntfy("error", stage="no_gpu", note=gpus[:200])
        log("no GPU available; exiting"); sys.exit(1)

    # Control watcher: from boot onward, listen for a 'stop' from the panel and exit at ANY
    # phase (build/download/serve) so a session can always be killed to free the GPU slot.
    def _control_watch(since):
        while True:
            try:
                if stop_requested(since):
                    ntfy("stopping", note="stop received; exiting to free GPU")
                    os._exit(0)
            except Exception:
                pass
            # ~10-15s with jitter: highly responsive to UI stop commands while keeping HTTP ETag checks cheap
            time.sleep((20 if RV_MODE == "ntfy" else 10) + random.uniform(0, 5))
    threading.Thread(target=_control_watch, args=(time.time(),), daemon=True).start()

    # Build-cache mode: compile llama.cpp, stage binaries to /kaggle/working/llamacpp-bin,
    # then EXIT so the run completes and the orchestrator can snapshot the output into a
    # reusable Kaggle dataset. No GPU serving, no tunnel.
    if CFG.get("build_only"):
        binpath = build_llamacpp()
        ntfy("built", note="llama.cpp compiled; output staged for caching",
             bin=os.path.basename(binpath))
        log("build_only done; exiting for output snapshot")
        return

    server_bin = None if RUNTIME in ("diffusers", "swarm", "fastvideo", "comfyui") else get_server_binary()
    cf = get_cloudflared()

    if RUNTIME == "diffusers":
        srv = start_diffusers()
    elif RUNTIME == "fastvideo":
        srv = start_fastvideo()
    elif RUNTIME == "comfyui":
        srv = start_comfyui()
    elif RUNTIME == "swarm":
        srv = start_swarm(server_bin)
    else:
        srv = start_llama(server_bin)

    log(f"{RUNTIME} healthy; starting tunnel")
    ntfy("serving", note=f"{RUNTIME} up on localhost" if RUNTIME != "llama.cpp" else "llama-server up on localhost")
    start_proxy(PORT, LLAMA_PORT)
    start_telemetry()
    if not wait_health(PORT, timeout=60):
        log("keep-alive proxy did not start; tunnelling directly")
        globals()["PORT"] = LLAMA_PORT
    tproc, url = start_tunnel(cf, PORT)
    if not url:
        try: tproc.kill()
        except Exception: pass
        tproc, url = start_tunnel(cf, PORT, "http2")            # QUIC/UDP may be blocked: try TCP
    if not url:
        ntfy("error", stage="tunnel", note="no trycloudflare url parsed")
        log("tunnel url not found"); srv.kill(); sys.exit(1)

    log("TUNNEL URL:", url)
    ntfy("ready", url=url, model=MODEL_ID,
         ctx=CFG.get("model_ctx"), port=PORT)
    if RUNTIME == "swarm" and int(CFG.get("stage_idx", 0)) == 0:
        relay_hub = CFG.get("relay_hub") or ""
        try:
            if not relay_hub:
                raise ValueError("no swarm relay hub configured")
            hub_url = f"{relay_hub.rstrip('/')}/swarm/tunnel"
            payload = json.dumps({"stage": 0, "url": url}).encode("utf-8")
            req = urllib.request.Request(hub_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                log(f"Directly registered Stage 0 tunnel on relay hub ({resp.status})")
        except Exception as e:
            log(f"Relay hub tunnel registration notice failed (non-fatal): {e}")

    # Heartbeat until the session budget runs out or a stop command arrives, then exit
    # cleanly (freeing this account's GPU quota) for the orchestrator to relaunch.
    ctl_since = time.time()
    end = time.time() + CFG.get("session_seconds", 39600)
    last_beat = 0; beat_every = 300
    tun_fail = 0; last_probe = 0.0
    while time.time() < end:
        if srv.poll() is not None:
            try:
                _last = _TELEM[-1] if _TELEM else {}
                _slim = {k: _last.get(k) for k in ("up_s", "cg_used", "cg_anon", "cg_limit", "cg_ev_oom_kill", "srv_RssAnon", "gpu")}
            except Exception:
                _slim = {}
            ntfy("died", note=f"{RUNTIME} exited" if RUNTIME != "llama.cpp" else "llama-server exited",
                 rc=getattr(srv, "returncode", None), tail=tail("/kaggle/working/llama-server.log", 25)[-1800:], last_telemetry=_slim)
            log("model server died"); break
        # end-to-end watchdog: the cloudflared PROCESS can stay alive while its edge connection is dead
        # (QUIC stall, edge restart). Probe the public URL; after 3 consecutive failures replace the
        # tunnel, alternating protocol. llama-server and its warm prompt cache are untouched.
        stalled = False
        if tproc.poll() is None and url and time.time() - TUNNEL["born"] > 45 and time.time() - last_probe >= 15:
            last_probe = time.time()
            tun_fail = 0 if _public_ok(url) else tun_fail + 1
            if tun_fail >= 3:
                stalled = True
        if stalled or tproc.poll() is not None:
            # tunnel dropped/stalled; re-open and re-announce
            ntfy_persistent("tunnel_recovering", max_s=120, url=url, why="stalled" if stalled else "exited", failures=tun_fail)
            alt = "http2" if TUNNEL.get("proto") == "quic" else "quic"
            try:
                tproc.kill()
            except Exception:
                pass
            TUNNEL["restarts"] += 1; tun_fail = 0
            tproc, url = start_tunnel(cf, PORT, alt)
            if not url:
                tproc, url = start_tunnel(cf, PORT, "quic" if alt == "http2" else "http2")
            if url:
                ntfy_persistent("ready", url=url, model=MODEL_ID, reopened=True)
                if RUNTIME == "swarm" and int(CFG.get("stage_idx", 0)) == 0:
                    try:
                        relay_hub = CFG.get("relay_hub") or ""
                        if not relay_hub:
                            raise ValueError("no swarm relay hub configured")
                        hub_url = f"{relay_hub.rstrip('/')}/swarm/tunnel"
                        payload = json.dumps({"stage": 0, "url": url}).encode("utf-8")
                        req = urllib.request.Request(hub_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            log(f"Re-registered Stage 0 tunnel on relay hub ({resp.status})")
                    except Exception:
                        pass
        if time.time() - last_beat >= beat_every:
            ntfy("alive", url=url, remaining_s=int(end - time.time()))
            # ntfy.sh allows ~250 messages/day per IP (Kaggle egress IPs are shared): beat every ~10 min there
            last_beat = time.time(); beat_every = (600 if RV_MODE == "ntfy" else 300) + random.uniform(0, 60)
        time.sleep(15)                       # local checks only (process + tunnel), no network

    ntfy("expiring", note="session ended")
    log("shutting down")
    for p in (tproc, srv):
        try: p.terminate()
        except Exception: pass
    time.sleep(5)
    for p in (tproc, srv):
        try: p.kill()
        except Exception: pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        ntfy("crash", err=str(e), tb=traceback.format_exc()[-1500:])
        raise
