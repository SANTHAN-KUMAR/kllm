"""Optional public tunnel (cloudflared quick tunnel) so other devices can reach the gateway.

Off by default. Enable with `remote_enabled: true` in config.yaml (then `kllm serve` keeps the
tunnel up) or run `kllm remote`. cloudflared is found on PATH / `cloudflared_bin`, else
downloaded once into <root>/bin from the official GitHub release — only when the tunnel is
actually requested. The tunnel exposes the relay (or LiteLLM when enabled); everything except
the model API stays local-only and the model API needs `gateway_api_key` (see gateway/auth.py).
"""
import json, os, platform, re, shutil, stat, subprocess, sys, tarfile, threading, time, urllib.request, pathlib
from . import common

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"


def asset_name(system=None, machine=None):
    """Release asset for this platform (None if unsupported)."""
    system = (system or platform.system()).lower()
    m = (machine or platform.machine()).lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "amd64" if m in ("x86_64", "amd64") else \
           "arm" if m.startswith("armv") else "386" if m in ("i386", "i686", "x86") else None
    if arch is None:
        return None
    if system == "linux":
        return f"cloudflared-linux-{arch}"
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz" if arch in ("amd64", "arm64") else None
    if system == "windows":
        return f"cloudflared-windows-{arch}.exe" if arch in ("amd64", "386") else None
    return None


def find_cloudflared(cfg=None):
    cfg = cfg or {}
    exe = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    for c in (cfg.get("cloudflared_bin"), common.ROOT / "bin" / exe, shutil.which("cloudflared")):
        if c and pathlib.Path(str(c)).expanduser().exists():
            return str(pathlib.Path(str(c)).expanduser())
    return None


def ensure_cloudflared(cfg=None, log=print):
    """Path to cloudflared, downloading it into <root>/bin on first use."""
    found = find_cloudflared(cfg)
    if found:
        return found
    asset = asset_name()
    if not asset:
        raise RuntimeError(f"no cloudflared build for {platform.system()}/{platform.machine()}; "
                           "install it yourself and set cloudflared_bin: in config.yaml")
    exe = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    dest = common.ROOT / "bin" / exe
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading cloudflared ({asset}) -> {dest}")
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(RELEASE + asset, headers={"User-Agent": "kllm"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if asset.endswith(".tgz"):
        with tarfile.open(tmp) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("cloudflared"))
            with tf.extractfile(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        tmp.unlink()
    else:
        os.replace(tmp, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(dest)


def target_port(cfg):
    """What the tunnel exposes: LiteLLM if the user enabled it, else the relay itself."""
    return cfg.get("litellm_port", 4000) if cfg.get("litellm_enabled") else cfg.get("relay_port", 8080)


def _remote_file():
    return common.ROOT / "state" / "remote.json"


def run_forever(cfg_fn=common.load_config, log=print, stop=None, port=None):
    """Keep a tunnel up (restart on exit), recording its URL in state/remote.json.
    Supports both quick tunnels (default) and named tunnels (when tunnel_name is set in config)."""
    stop = stop or threading.Event()
    while not stop.is_set():
        cfg = cfg_fn()
        try:
            exe = ensure_cloudflared(cfg, log)
        except Exception as e:
            log(f"remote tunnel unavailable: {e}")
            stop.wait(300); continue
        port_ = port or target_port(cfg)
        tunnel_name = cfg.get("tunnel_name")
        tunnel_creds = cfg.get("tunnel_credentials_file")

        # A named tunnel is only reachable once a DNS name is routed to it (tunnel_hostname).
        # Without one, `<name>.cfargotunnel.com` does not resolve publicly, so the gateway would
        # look "up" while nothing can reach it: fall back to a quick tunnel instead.
        if tunnel_name and tunnel_creds and os.path.isfile(os.path.expanduser(tunnel_creds)) \
                and cfg.get("tunnel_hostname"):
            # Named tunnel: stable hostname, persistent connection
            creds = os.path.expanduser(tunnel_creds)
            cmd = [
                exe, "tunnel", "--no-autoupdate",
                "--protocol", "http2",
                "--heartbeat-interval", "10s",
                "--heartbeat-count", "5",
                "--credentials-file", creds,
                "--url", f"http://127.0.0.1:{port_}",
                "run", tunnel_name,
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            url = cfg.get("tunnel_hostname", f"https://{tunnel_name}.cfargotunnel.com")
            _remote_file().write_text(json.dumps({"url": url, "ts": int(time.time())}), encoding="utf-8")
            log(f"named tunnel: {url}  (key required: gateway_api_key)")
            pump_thread = None
        else:
            # Quick tunnel: ephemeral URL
            cmd = [
                exe, "tunnel", "--no-autoupdate",
                "--protocol", "http2",
                "--heartbeat-interval", "10s",
                "--heartbeat-count", "5",
                "--edge-bind-address", "0.0.0.0",
                "--url", f"http://127.0.0.1:{port_}",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            url = None
            def pump():
                nonlocal url
                for line in proc.stdout:
                    m = URL_RE.search(line)
                    if m and url is None:
                        url = m.group(0)
                        _remote_file().write_text(json.dumps({"url": url, "ts": int(time.time())}), encoding="utf-8")
                        log(f"remote tunnel: {url}  (key required: gateway_api_key)")
            pump_thread = threading.Thread(target=pump, daemon=True)
            pump_thread.start()

        try:
            while proc.poll() is None and not stop.is_set():
                stop.wait(2)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(5)
                except subprocess.TimeoutExpired: proc.kill()
            try: _remote_file().unlink()
            except FileNotFoundError: pass
        stop.wait(3)


class Controller:
    """Start/stop the tunnel at runtime (`kllm serve` and the panel's toggle share one instance)."""
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = None
        self._thread = None

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive() and self._stop and not self._stop.is_set())

    def start(self, log=print, cfg_fn=None):
        with self._lock:
            if self.running:
                return False
            self._stop = threading.Event()
            kw = {"log": log, "stop": self._stop}
            if cfg_fn:
                kw["cfg_fn"] = cfg_fn
            self._thread = threading.Thread(target=run_forever, kwargs=kw, name="tunnel", daemon=True)
            self._thread.start()
            return True

    def stop(self):
        with self._lock:
            if self._stop:
                self._stop.set()
            return True


CTL = Controller()
