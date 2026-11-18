"""Gateway API key for the relay — what LiteLLM's master_key used to provide.

Rule: requests from this machine (127.0.0.1 / ::1) need no key. Anything else must present
`gateway_api_key` (config) as `Authorization: Bearer <key>` or `x-api-key: <key>` (Anthropic
SDKs), and may only use the model API (/v1/*, /health) — the panel and the /__relay and
/admin endpoints stay local-only. If no real key is configured, non-local requests are
refused (secure by default).

A tunnel (cloudflared, nginx...) connects from 127.0.0.1, so a request that carries proxy
headers (X-Forwarded-For, CF-Connecting-IP, ...) is NOT treated as local.
Pure ASGI middleware: no buffering, streaming responses pass through untouched.
"""
import hmac, json

LOCAL = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}
PROXY_HEADERS = (b"x-forwarded-for", b"x-real-ip", b"forwarded", b"cf-connecting-ip", b"cf-ray",
                 b"x-forwarded-host", b"true-client-ip")
PLACEHOLDER_KEYS = {"", "sk-change-me", "sk-kllm-local", "changeme"}
PUBLIC_PREFIXES = ("/v1/", "/health")


def real_key(cfg):
    k = (cfg or {}).get("gateway_api_key")
    return None if k is None or str(k).strip() in PLACEHOLDER_KEYS else str(k).strip()


def is_local(client_host, header_names):
    return client_host in LOCAL and not any(h in header_names for h in PROXY_HEADERS)


def presented_key(headers):
    """headers: iterable of (lower-case bytes name, bytes value)."""
    for name, value in headers:
        if name == b"authorization":
            v = value.decode("latin-1").strip()
            return v[7:].strip() if v.lower().startswith("bearer ") else v
        if name == b"x-api-key":
            return value.decode("latin-1").strip()
    return ""


def check(cfg, client_host, headers, path):
    """-> None if allowed, else (status, message)."""
    names = {n for n, _ in headers}
    if is_local(client_host, names):
        return None
    key = real_key(cfg)
    if key is None:
        return 403, "kllm: remote access is disabled until gateway_api_key is set in config.yaml"
    if not (path in ("/health", "/v1") or path.startswith(PUBLIC_PREFIXES)):
        return 403, "kllm: this endpoint is only available from the local machine"
    got = presented_key(headers)
    if not hmac.compare_digest(got.encode(), key.encode()):
        return 401, "kllm: missing or invalid API key (send Authorization: Bearer <gateway_api_key>)"
    return None


class GatewayKeyMiddleware:
    """`cfg` is a zero-arg callable returning the current config dict (hot-reloaded)."""
    def __init__(self, app, cfg):
        self.app, self.cfg = app, cfg

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        client = (scope.get("client") or ("", 0))[0]
        denied = check(self.cfg(), client, scope.get("headers") or [], scope.get("path", ""))
        if denied is None:
            return await self.app(scope, receive, send)
        status, msg = denied
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        body = json.dumps({"error": {"message": msg, "type": "authentication_error"}}).encode()
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
            (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})


# ---- local control surface: CSRF and DNS-rebinding guard --------------------------------------------------
# /admin, /mcp and /__bridge can start processes (the shell harness runs arbitrary commands), so a web page
# open in the user's browser must never reach them. A browser always sends Origin on a cross-site POST, and
# cannot send Content-Type: application/json (or a custom header) cross-site without a CORS preflight that
# kllm never grants. CLI and MCP clients send JSON and no Origin, so they keep working.
GUARDED_PREFIXES = ("/admin", "/mcp", "/__bridge", "/chat-ui")
LOCAL_NAMES = {"localhost", "127.0.0.1", "::1", "testserver"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _hostname(value):
    v = (value or "").strip().lower()
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0]
    if v.startswith("[") and "]" in v:
        return v[1:v.index("]")]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def guard_local(method, path, headers, allowed_hosts=()):
    """-> None if allowed, else (status, message). headers: iterable of (lower-case bytes, bytes)."""
    if not path.startswith(GUARDED_PREFIXES):
        return None
    h = {n: v.decode("latin-1") for n, v in headers}
    allowed = LOCAL_NAMES | {str(x).lower() for x in allowed_hosts or ()}
    if _hostname(h.get(b"host", "")) not in allowed:
        return 403, "kllm: unexpected Host header for a local-only endpoint"
    origin = h.get(b"origin")
    if origin and (origin == "null" or _hostname(origin) not in allowed):
        return 403, "kllm: cross-site requests to the control API are refused"
    if method not in SAFE_METHODS:
        ctype = h.get(b"content-type", "").split(";", 1)[0].strip().lower()
        has_body = h.get(b"content-length", "0").strip() not in ("", "0") or b"transfer-encoding" in h
        if has_body and not h.get(b"x-kllm-panel") and ctype != "application/json":
            return 403, "kllm: control requests need Content-Type: application/json or the X-Kllm-Panel header"
    return None


class LocalGuardMiddleware:
    """`cfg` is a zero-arg callable returning the current config dict (hot-reloaded)."""
    def __init__(self, app, cfg):
        self.app, self.cfg = app, cfg

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        try:
            extra = (self.cfg() or {}).get("admin_allowed_hosts") or ()
        except Exception:
            extra = ()
        denied = guard_local(scope.get("method", "GET"), scope.get("path", ""), scope.get("headers") or [], extra)
        if denied is None:
            return await self.app(scope, receive, send)
        status, msg = denied
        body = json.dumps({"ok": False, "msg": msg, "error": {"message": msg, "type": "forbidden"}}).encode()
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
