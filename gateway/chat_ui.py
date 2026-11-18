"""Chat UI: llama.cpp's own web UI (MIT, vendored in ui/vendor/llama-ui), served at /chat-ui/ on the relay.

The UI runs in llama-server "router" mode, so it shows a model picker. kllm answers the few calls it makes:
  GET  /chat-ui/props               router props (no model) or the props of the backend serving ?model=
  GET  /chat-ui/v1/models           kllm chat models with status loaded / loading / unloaded and vision flags
  GET  /chat-ui/models/sse          status changes as backends come and go
  POST /chat-ui/models/load         accepted and ignored: a model starts on its first message (autoscale), never on a pick
  POST /chat-ui/v1/chat/completions the relay's normal chat route (routing, failover, prompt cache affinity)
  GET  /chat-ui/slots               [] (kllm does not expose per-slot state)
Everything else the UI might ask for (stream resume, tools, MCP proxy) answers 404, which the UI treats as absent.
"""
import asyncio, json, pathlib, time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import media, vision

DIR = (pathlib.Path(__file__).parent / "ui" / "vendor" / "llama-ui").resolve()
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}


def _gz(rel):
    """A vendored file, served still gzipped (every browser accepts gzip)."""
    f = (DIR / (rel + ".gz")).resolve()
    if DIR not in f.parents or not f.is_file():
        return JSONResponse({"error": {"code": 404, "message": "not found", "type": "not_found_error"}}, status_code=404)
    ext = pathlib.Path(rel).suffix
    cache = "no-cache" if ext == ".html" else "public, max-age=31536000, immutable"    # bundle names carry a hash
    return Response(f.read_bytes(), media_type=TYPES.get(ext, "application/octet-stream"),
                    headers={"Content-Encoding": "gzip", "Cache-Control": cache, "Vary": "Accept-Encoding"})


def _status(R, key):
    """loaded | loading | unloaded for one model key, from the relay's live state."""
    s = R.state()
    health = s.get("health") or {}
    for b in s.get("backends") or []:
        if b.get("url") and R._canonical_model(b.get("model_key")) == key and (health.get(b["account"]) or {}).get("ok", True):
            return "loaded"
    if any(R._canonical_model((p or {}).get("model_key")) == key for p in (s.get("pending") or {}).values()):
        return "loading"
    return "unloaded"


def chat_models(R):
    """[(key, model dict)] for chat models a user can pick (no media, no aliases)."""
    return [(k, m) for k, m in R._registry().items()
            if isinstance(m, dict) and media.kind(m) == "chat" and not m.get("alias_of") and not m.get("hidden")]


def _vision(key, m):
    tags = [str(t).lower() for t in (m.get("tags") or [])]
    return bool(m.get("mmproj")) or "vision" in tags or vision.is_vision_model(key)


def models_payload(R):
    now = int(time.time())
    data = []
    for key, m in chat_models(R):
        row = {"id": key, "aliases": [], "tags": m.get("tags") or [], "object": "model", "owned_by": "kllm", "created": now,
               "status": {"value": _status(R, key), "args": []},
               "architecture": {"input_modalities": ["text", "image"] if _vision(key, m) else ["text"], "output_modalities": ["text"]},
               "source": "preset", "can_remove": False}
        if m.get("ctx"):
            row["context_length"] = int(m["ctx"])
        data.append(row)
    return {"object": "list", "data": data}


def router_props(R):
    return {"role": "router", "max_instances": 0, "models_autoload": True,
            "model_alias": "kllm", "model_path": "none",
            "default_generation_settings": {"params": {}, "n_ctx": 0},
            "ui_settings": {}, "build_info": "kllm relay", "cors_proxy_enabled": False}


def fallback_props(key, m):
    """Props for a model nothing serves yet: enough for the UI to show its context size and image button."""
    return {"role": "model", "model_alias": key, "model_path": key,
            "default_generation_settings": {"params": {}, "n_ctx": int(m.get("ctx") or 0)},
            "modalities": {"vision": _vision(key, m), "audio": False}, "build_info": "kllm relay"}


def make_router(R):
    r = APIRouter()

    @r.get("/chat-ui")
    async def root_redirect():
        return Response(status_code=307, headers={"Location": "/chat-ui/"})

    @r.get("/chat-ui/")
    async def index():
        return _gz("index.html")

    @r.get("/chat-ui/_app/{path:path}")
    async def assets(path: str):
        return _gz("_app/" + path)

    @r.get("/chat-ui/props")
    async def props(model: str = ""):
        if not model:
            return router_props(R)
        key = R._canonical_model(model)
        m = R._registry().get(key)
        if not isinstance(m, dict):
            return JSONResponse({"error": {"code": 404, "message": f"model '{model}' not found", "type": "not_found_error"}}, status_code=404)
        for label, url, api_key, mk in R._pool():
            if R._canonical_model(mk) != key:
                continue
            try:
                async with httpx.AsyncClient(timeout=8) as c:
                    res = await c.get(url + "/props", headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
                if res.status_code == 200:
                    p = res.json()
                    p["model_alias"] = key        # the UI matches messages to models by this name
                    p["role"] = "model"
                    return p
            except (httpx.HTTPError, ValueError):
                continue
        return fallback_props(key, m)

    @r.get("/chat-ui/v1/models")
    async def models():
        return models_payload(R)

    @r.get("/chat-ui/models/sse")
    async def models_sse(request: Request):
        async def gen():
            seen = {}
            yield b": kllm\n\n"
            while not await request.is_disconnected():
                for key, _ in chat_models(R):
                    st = _status(R, key)
                    if seen.get(key) not in (None, st):
                        yield ("data: " + json.dumps({"model": key, "event": "status_change", "data": {"status": st}}) + "\n\n").encode()
                    seen[key] = st
                for _ in range(10):                      # look again every 3 s, but notice a closed tab sooner
                    if await request.is_disconnected():
                        return
                    await asyncio.sleep(0.3)
                yield b": ping\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @r.post("/chat-ui/models/load")
    async def load(request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        key = R._canonical_model(str(body.get("model") or ""))
        if key not in dict(chat_models(R)):
            return JSONResponse({"error": {"code": 404, "message": f"model '{body.get('model')}' not found", "type": "not_found_error"}}, status_code=404)
        # picking a model in the UI must not spend a GPU: nothing launches here. The first message for it goes through
        # the relay like any other request, and autoscale starts a kernel then (and the relay holds the request until
        # it serves). 2026-11-10: browsing three models in the picker launched three kernels.
        return {"success": True}

    @r.post("/chat-ui/models/unload")
    async def unload():
        # stopping kernels is Fleet's job: the chat UI never shuts a backend down
        return {"success": True}

    @r.get("/chat-ui/slots")
    async def slots():
        return []

    @r.get("/chat-ui/tools")
    async def tools():
        # llama-server's built-in file and shell tools are not offered: the relay is not a sandbox. MCP servers can
        # still be added in the UI's settings (kllm's own is at /mcp on this origin).
        return []

    @r.get("/chat-ui/sw.js")
    async def service_worker():
        # the UI registers an offline worker; a no-op one keeps the console clean and caches nothing stale
        return Response("self.addEventListener('install', () => self.skipWaiting());\n", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})

    @r.api_route("/chat-ui/v1/chat/completions", methods=["POST"])
    async def completions(request: Request):
        return await R.proxy("v1/chat/completions", request)

    @r.api_route("/chat-ui/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def absent(path: str):
        return JSONResponse({"error": {"code": 404, "message": f"kllm: /{path} is not available here", "type": "not_found_error"}}, status_code=404)

    return r
