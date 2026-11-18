"""The vendored llama.cpp chat UI on /chat-ui/: kllm answers its router-mode calls from the relay's own state."""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway import chat_ui

REG = {"qwen": {"ctx": 262144, "tags": ["vision"]}, "small": {"ctx": 32768}, "img": {"runtime": "diffusers", "kind": "image"},
       "alias": {"alias_of": "qwen"}}


def fake_relay(spawned, backends=(), pending=None):
    st = {"backends": list(backends), "pending": pending or {}, "health": {}}
    return SimpleNamespace(state=lambda: st, _registry=lambda: REG, _canonical_model=lambda k: (REG.get(k) or {}).get("alias_of", k),
                           _pool=lambda: [], _spawn=lambda args: spawned.append(args))


def client(r):
    app = FastAPI()
    app.include_router(chat_ui.make_router(r))
    return TestClient(app)


def test_models_list_chat_models_with_live_status():
    r = fake_relay([], backends=[{"account": "kg1", "url": "https://x", "model_key": "qwen"}], pending={"kg2": {"model_key": "small"}})
    data = {m["id"]: m for m in client(r).get("/chat-ui/v1/models").json()["data"]}
    assert set(data) == {"qwen", "small"}                                   # no image models, no aliases
    assert data["qwen"]["status"]["value"] == "loaded" and data["small"]["status"]["value"] == "loading"
    assert data["qwen"]["architecture"]["input_modalities"] == ["text", "image"] and data["qwen"]["context_length"] == 262144


def test_picking_a_model_never_launches_a_kernel():
    spawned = []
    c = client(fake_relay(spawned))
    assert c.post("/chat-ui/models/load", json={"model": "small"}).json() == {"success": True}
    assert spawned == []


def test_props_and_ui_files():
    c = client(fake_relay([]))
    assert c.get("/chat-ui/props").json()["role"] == "router"
    p = c.get("/chat-ui/props", params={"model": "qwen"}).json()          # nothing serves it: props from models.yaml
    assert p["modalities"]["vision"] and p["default_generation_settings"]["n_ctx"] == 262144
    r = c.get("/chat-ui/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert c.get("/chat-ui/tools").json() == [] and c.get("/chat-ui/nope").status_code == 404
