"""Panel API for "Add a model": inspect a HuggingFace repo, then add it to models.yaml.
Kept out of relay.py on purpose; relay.include_router()s it before its catch-all proxy route.
The existing /admin/api/add-model (manual form) stays in relay.py."""
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from . import common, inspect_model as im, media


def _params(body):
    try:
        cfg = common.load_config()
    except common.ConfigMissing:
        cfg = {}
    ctx = body.get("ctx")
    return dict(source=(body.get("source") or "").strip(), file=(body.get("file") or "").strip() or None,
                ctx=int(ctx) if str(ctx or "").strip().isdigit() else None,
                name=(body.get("name") or "").strip() or None,
                token=cfg.get("hf_token") or None, taken=set(common.load_models())), cfg


def make_router(refresh_litellm=lambda: None):
    r = APIRouter()

    @r.post("/admin/api/inspect")
    async def inspect(request: Request):
        params, _ = _params(await request.json())
        try:
            res = await asyncio.to_thread(im.inspect_source, **params)
        except im.InspectError as e:
            return JSONResponse({"ok": False, "msg": str(e)}, status_code=400)
        out = {k: res[k] for k in ("runtime", "kind", "repo", "key", "entry", "files", "chosen",
                                   "weights_gb", "vram_gb", "ctx", "warnings", "fits")}
        # every chat model kllm ships is abliterated; flag one that is not so the panel can say so
        out["abliterated"] = res["kind"] != "chat" or any(
            "abliterat" in str(x).lower() for x in (res["repo"], res.get("chosen"), res["key"]))
        return {"ok": True, **out}

    @r.post("/admin/api/add-from-hf")
    async def add_from_hf(request: Request):
        body = await request.json()
        params, _cfg = _params(body)
        try:
            res = await asyncio.to_thread(im.inspect_source, **params)
            if not res["fits"] and not body.get("force"):
                raise im.InspectError("; ".join(res["warnings"]) or "does not fit 2xT4")
            if body.get("default") and media.kind(res["entry"]) != "chat":
                raise im.InspectError("only chat models can be the default model")
            im.write_entry(res["key"], res["entry"], common.MODELS_FILE)
        except im.InspectError as e:
            return JSONResponse({"ok": False, "msg": str(e)}, status_code=400)
        if body.get("default"):
            common.set_config_value("default_model", res["key"])
        msg = f"added '{res['key']}' ({res['kind']}, {res['runtime']}). Launch it from the model library."
        try:
            refresh_litellm()
        except Exception as e:
            msg += f" (LiteLLM refresh failed: {e})"
        return {"ok": True, "msg": msg, "key": res["key"]}

    return r
