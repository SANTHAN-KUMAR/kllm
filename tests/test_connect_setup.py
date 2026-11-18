"""Connect setup and undo write configs safely: comments kept, other entries kept, Undo restores the exact file."""
import json, os

import pytest

from gateway import connect


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(connect.common, "which", lambda name: f"/usr/bin/{name}" if name in ("opencode", "agy", "gemini", "codex") else None)
    monkeypatch.setattr(connect, "_mcp_stdio", lambda: {"command": "/py", "args": ["-m", "gateway.mcp_server"], "cwd": "/kllm", "env": {}})
    connect._CACHE["rows"] = None
    return tmp_path


def test_opencode_provider_and_mcp_keep_comments_and_undo(home):
    cfg = home / ".config" / "opencode" / "opencode.jsonc"
    cfg.parent.mkdir(parents=True)
    original = '{\n  // my theme\n  "theme": "dark",\n  "provider": {\n    "other": {"name": "x"}\n  }\n}\n'
    cfg.write_text(original, encoding="utf-8")
    connect.setup("opencode", "models", "http://127.0.0.1:8080")
    connect.setup("opencode", "mcp", "http://127.0.0.1:8080")
    text = cfg.read_text(encoding="utf-8")
    data = json.loads(connect._strip_jsonc(text))
    assert "// my theme" in text and data["theme"] == "dark"
    assert set(data["provider"]) == {"other", "kllm"} and data["provider"]["kllm"]["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert data["mcp"]["kllm"]["url"] == "http://127.0.0.1:8080/mcp"
    rows = {r["id"]: r for r in connect.detect(force=True)}
    assert [o["state"] for o in rows["opencode"]["options"]] == ["on", "on"]
    connect.undo("opencode", "mcp")
    connect.undo("opencode", "models")
    assert cfg.read_text(encoding="utf-8") == original


def test_antigravity_mcp_merges_and_undo_restores(home):
    p = home / ".gemini" / "antigravity-cli" / "mcp.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    connect.setup("agy", "mcp")
    servers = json.loads(p.read_text(encoding="utf-8"))["mcpServers"]
    assert set(servers) == {"other", "kllm"} and servers["kllm"]["command"] == "/py"
    with pytest.raises(connect.SetupError):
        connect.setup("agy", "models")
    connect.undo("agy", "mcp")
    assert p.read_text(encoding="utf-8") == before


def test_gemini_settings_created_then_removed_on_undo(home):
    p = home / ".gemini" / "settings.json"
    connect.setup("gemini", "mcp", "http://127.0.0.1:9090")
    assert json.loads(p.read_text(encoding="utf-8"))["mcpServers"]["kllm"] == {"httpUrl": "http://127.0.0.1:9090/mcp"}
    connect.undo("gemini", "mcp")
    assert not p.exists()


def test_codex_appends_provider_profile_and_mcp(home):
    p = home / ".codex" / "config.toml"
    p.parent.mkdir(parents=True)
    p.write_text('model = "o4-mini"\n', encoding="utf-8")
    connect.setup("codex", "models", "http://127.0.0.1:8080")
    connect.setup("codex", "mcp")
    t = p.read_text(encoding="utf-8")
    assert t.startswith('model = "o4-mini"') and "[model_providers.kllm]" in t and "[profiles.kllm]" in t and "[mcp_servers.kllm]" in t
    assert 'base_url = "http://127.0.0.1:8080/v1"' in t
    connect.undo("codex", "mcp"); connect.undo("codex", "models")
    assert p.read_text(encoding="utf-8") == 'model = "o4-mini"\n'


def test_broken_json_is_refused_not_overwritten(home):
    p = home / ".gemini" / "settings.json"
    p.parent.mkdir(parents=True)
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(connect.SetupError):
        connect.setup("gemini", "mcp")
    assert p.read_text(encoding="utf-8") == "{ not json"


def test_claude_launcher_undo_leaves_a_foreign_launcher_alone(home):
    b = home / ".local" / "bin"
    b.mkdir(parents=True)
    dst = connect._launcher(b)
    if os.name == "nt":
        dst.write_text("@echo off\r\nsomething-else.exe %*\r\n", encoding="utf-8")   # a launcher someone else wrote
    else:
        other = home / "my-launcher"
        other.write_text("#!/bin/sh\n", encoding="utf-8")
        os.symlink(other, dst)
    assert connect._ours(dst) is False
    before = dst.read_text(encoding="utf-8")
    with pytest.raises(connect.SetupError):
        connect.undo("claude-code", "models")
    assert dst.exists() and dst.read_text(encoding="utf-8") == before


def test_setup_twice_changes_nothing_the_second_time(home):
    cfg = home / ".config" / "opencode" / "opencode.jsonc"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{\n  "provider": {}\n}\n', encoding="utf-8")
    connect.setup("opencode", "models"); connect.setup("opencode", "mcp")
    once = cfg.read_text(encoding="utf-8")
    assert "already" in connect.setup("opencode", "models") and "already" in connect.setup("opencode", "mcp")
    assert cfg.read_text(encoding="utf-8") == once
    for tool in ("agy", "gemini"):
        connect.setup(tool, "mcp")
        assert "already" in connect.setup(tool, "mcp")
    p = home / ".codex" / "config.toml"
    connect.setup("codex", "models"); connect.setup("codex", "mcp")
    t = p.read_text(encoding="utf-8")
    assert "already" in connect.setup("codex", "models") and "already" in connect.setup("codex", "mcp") and p.read_text(encoding="utf-8") == t


def test_two_changes_in_one_millisecond_undo_back_to_the_original(home, monkeypatch):
    p = home / ".codex" / "config.toml"
    p.parent.mkdir(parents=True)
    p.write_text('model = "o4-mini"\n', encoding="utf-8")
    monkeypatch.setattr(connect.time, "time", lambda: 1_700_000_000.0)
    connect.setup("codex", "models", "http://127.0.0.1:8080")
    connect.setup("codex", "mcp")
    connect.undo("codex", "mcp"); connect.undo("codex", "models")
    assert p.read_text(encoding="utf-8") == 'model = "o4-mini"\n'
