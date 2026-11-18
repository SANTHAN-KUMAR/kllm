"""Paths, single-instance lock, tunnel helpers, claude-kllm launcher logic."""
import json, os, pathlib
from gateway import common, tunnel, claude_launcher as cl


def test_default_root_order(tmp_path, monkeypatch):
    monkeypatch.setenv("KLLM_ROOT", str(tmp_path / "x"))
    assert common.default_root() == tmp_path / "x"
    monkeypatch.delenv("KLLM_ROOT")
    monkeypatch.setattr(common, "is_checkout", lambda: True)
    assert common.default_root() == common.PKG_ROOT
    monkeypatch.setattr(common, "is_checkout", lambda: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    if os.name != "nt":
        assert common.default_root() == tmp_path / "xdg" / "kllm"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setattr(common.pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    if os.name != "nt":
        assert common.default_root() == tmp_path / ".config" / "kllm"


def test_no_machine_specific_paths_in_code():
    bad = ("/run/media/", "/home/", "~/kllm", "$HOME/kllm", ".local/bin/kg", "C:\\Users\\")
    for f in (common.PKG_DIR).glob("*.py"):
        txt = f.read_text(encoding="utf-8")
        for b in bad:
            assert b not in txt.replace("~/.local/bin/kgN", ""), (f.name, b)


def test_data_path_prefers_user_override(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    assert common.data_path("kernel").name == "kernel" and common.data_path("kernel") != tmp_path / "kernel"
    (tmp_path / "kernel").mkdir()
    assert common.data_path("kernel") == tmp_path / "kernel"


def test_single_instance_lock_is_exclusive():
    h = common.single_instance("unit-test-lock")
    assert h is not None
    assert common.single_instance("unit-test-lock") is None
    os.close(h)
    h2 = common.single_instance("unit-test-lock")
    assert h2 is not None
    os.close(h2)


def test_missing_config_message(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CONFIG_FILE", tmp_path / "nope.yaml")
    try:
        common.load_config(); assert False
    except common.ConfigMissing as e:
        assert "kllm init" in str(e)


def test_cloudflared_asset_names():
    assert tunnel.asset_name("Linux", "x86_64") == "cloudflared-linux-amd64"
    assert tunnel.asset_name("Linux", "aarch64") == "cloudflared-linux-arm64"
    assert tunnel.asset_name("Darwin", "arm64") == "cloudflared-darwin-arm64.tgz"
    assert tunnel.asset_name("Windows", "AMD64") == "cloudflared-windows-amd64.exe"
    assert tunnel.asset_name("FreeBSD", "x86_64") is None


def test_tunnel_url_and_target_port(tmp_path, monkeypatch):
    assert tunnel.URL_RE.search("INF |  https://quick-wolf-12.trycloudflare.com  |").group(0) == \
        "https://quick-wolf-12.trycloudflare.com"
    assert tunnel.target_port({"relay_port": 8080}) == 8080
    assert tunnel.target_port({"relay_port": 8080, "litellm_enabled": True, "litellm_port": 4000}) == 4000
    exe = tmp_path / "cf"; exe.write_text("x", encoding="utf-8")
    assert tunnel.find_cloudflared({"cloudflared_bin": str(exe)}) == str(exe)


def test_ensure_cloudflared_does_not_download_when_present(tmp_path, monkeypatch):
    exe = tmp_path / "cf"; exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(tunnel.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")))
    assert tunnel.ensure_cloudflared({"cloudflared_bin": str(exe)}) == str(exe)


def test_claude_launcher_model_and_env():
    st = {"default_model": "big", "pool": [{"model_key": "big", "ctx": 262144}, {"model_key": "small", "ctx": 98304}]}
    assert cl.pick_model(st) == ("big", 262144)
    assert cl.pick_model(st, "small") == ("small", 98304)
    assert cl.pick_model({"default_model": "big", "pool": [{"model_key": "small", "ctx": 98304}]}) == ("kllm", 98304)
    assert cl.pick_model({"pool": []}) == ("kllm", 32768)
    env = cl.build_env("http://127.0.0.1:8080", "big", 262144, {"ANTHROPIC_API_KEY": "real", "PATH": "/bin"})
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080" and "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144" and env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"
    # long-context models get room for effort-mapped thinking + the answer; small windows stay at 16K
    assert cl.build_env("r", "m", 98304, {})["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"
    assert cl.build_env("r", "m", 32768, {})["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16384"
    assert cl.build_env("r", "m", 1, {"KLLM_API_KEY": "sk-remote"})["ANTHROPIC_AUTH_TOKEN"] == "sk-remote"


def test_claude_launcher_args():
    a = cl.build_args("m", ["-p", "hi"])
    assert a[:2] == ["--model", "m"] and "--strict-mcp-config" in a and a[-2:] == ["-p", "hi"]
    assert json.loads(a[a.index("--settings") + 1])["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert "--tools" not in cl.build_args("m", [], lean=False)


def test_claude_launcher_parse_model_arg():
    m, rest = cl.parse_model_arg(["--model", "x-qwen36-512k", "-p", "hi"])
    assert m == "x-qwen36-512k" and rest == ["-p", "hi"]

    m, rest = cl.parse_model_arg(["--model=qwen3-coder-30b-abliterated", "--verbose"])
    assert m == "qwen3-coder-30b-abliterated" and rest == ["--verbose"]

    m, rest = cl.parse_model_arg(["-p", "hi", "--dangerously-skip-permissions"])
    assert m is None and rest == ["-p", "hi", "--dangerously-skip-permissions"]

    # When --model is specified, pick_model resolves its context correctly
    st = {
        "default_model": "qwen3.6-35b-abliterated",
        "pool": [
            {"model_key": "qwen3.6-35b-abliterated", "ctx": 262144},
            {"model_key": "x-qwen36-512k", "ctx": 524288},
        ]
    }
    model, ctx = cl.pick_model(st, m)
    assert model == "qwen3.6-35b-abliterated" and ctx == 262144

    m, _ = cl.parse_model_arg(["--model", "x-qwen36-512k"])
    model, ctx = cl.pick_model(st, m)
    assert model == "x-qwen36-512k" and ctx == 524288

