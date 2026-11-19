"""Update checks: PyPI is the source, results are cached, nothing blocks, upgrades use the installing tool."""
import io, json, time

import pytest

from gateway import cli, common, updates


def fake_pypi(monkeypatch, releases, calls=None):
    body = json.dumps({"info": {"version": max(releases)},
                       "releases": {v: [{"yanked": y}] for v, y in releases.items()}}).encode()

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def urlopen(req, timeout=0):
        if calls is not None:
            calls.append(req.full_url)
        return R(body)
    monkeypatch.setattr(updates.urllib.request, "urlopen", urlopen)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(updates, "current_version", lambda: "0.3.1")
    monkeypatch.delenv("KLLM_NO_UPDATE_CHECK", raising=False)
    return tmp_path


def test_version_order():
    assert updates.newer("0.3.2", "0.3.1") and updates.newer("0.10.0", "0.9.9") and updates.newer("1.0", "0.99.9")
    assert not updates.newer("0.3.1", "0.3.1") and not updates.newer("0.3.2rc1", "0.3.2")


def test_latest_skips_prereleases_and_yanked(root, monkeypatch):
    fake_pypi(monkeypatch, {"0.3.1": False, "0.3.2": False, "0.4.0rc1": False, "0.3.3": True})
    assert updates.fetch_latest() == "0.3.2"


def test_status_caches_for_a_day(root, monkeypatch):
    calls = []
    fake_pypi(monkeypatch, {"0.3.1": False, "0.3.2": False}, calls)
    s = updates.status()
    assert s["update_available"] and s["latest"] == "0.3.2" and s["current"] == "0.3.1"
    updates.status()
    assert len(calls) == 1                                        # second call served from the cache
    assert updates.notice() == "kllm 0.3.2 is available (you have 0.3.1). Update: kllm update"


def test_offline_is_quiet(root, monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(updates.urllib.request, "urlopen", boom)
    s = updates.status()
    assert s["latest"] is None and s["update_available"] is False and updates.notice() is None


def test_opt_out(root, monkeypatch):
    monkeypatch.setenv("KLLM_NO_UPDATE_CHECK", "1")
    assert not updates.enabled({}) and updates.notice({}) is None
    monkeypatch.delenv("KLLM_NO_UPDATE_CHECK")
    assert not updates.enabled({"update_check": False}) and updates.enabled({})


@pytest.mark.parametrize("exe,method,first", [
    ("/home/a/.local/share/uv/tools/kllm-gateway/bin/python", "uv", "uv"),
    ("C:/Users/a/AppData/Roaming/uv/tools/kllm-gateway/Scripts/python.exe", "uv", "uv"),
    ("/home/a/.local/pipx/venvs/kllm-gateway/bin/python", "pipx", "pipx"),
    ("/usr/bin/python3", "pip", None),
])
def test_upgrade_command_matches_the_installer(monkeypatch, tmp_path, exe, method, first):
    monkeypatch.setattr(common, "PKG_ROOT", tmp_path / "site-packages")
    monkeypatch.setattr(updates.sys, "executable", exe)
    monkeypatch.setattr(updates.sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setattr(updates, "find_tool", lambda name: f"/opt/bin/{name}")       # the installing tool is present
    u = updates.upgrade_command()
    last = u["steps"][-1]
    assert u["method"] == method and "kllm-gateway" in last
    if first:
        assert last[0] == f"/opt/bin/{first}"
    else:
        assert last[1:4] == ["-m", "pip", "install"]


def test_missing_uv_falls_back_to_the_environments_pip(monkeypatch, tmp_path):
    env = tmp_path / "tools" / "kllm-gateway"; env.mkdir(parents=True)
    (env / "uv-receipt.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(common, "PKG_ROOT", env / "Lib" / "site-packages")
    monkeypatch.setattr(updates.sys, "prefix", str(env))
    monkeypatch.setattr(updates.sys, "executable", str(env / "python"))
    monkeypatch.setattr(updates, "find_tool", lambda name: None)
    u = updates.upgrade_command()                       # the fake interpreter has no pip: ensurepip comes first
    assert "not found" in u["method"]
    assert u["steps"][0][1:3] == ["-m", "ensurepip"] and u["steps"][-1][1:4] == ["-m", "pip", "install"]


def test_find_tool_looks_where_installers_put_it(monkeypatch, tmp_path):
    monkeypatch.setattr(updates.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("USERPROFILE", str(tmp_path))
    exe = "uv.exe" if updates.os.name == "nt" else "uv"
    (tmp_path / ".local" / "bin").mkdir(parents=True)
    (tmp_path / ".local" / "bin" / exe).write_text("", encoding="utf-8")
    assert updates.find_tool("uv") == str(tmp_path / ".local" / "bin" / exe)
    assert updates.find_tool("pipx") is None


def test_checkout_upgrades_with_git(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(common, "PKG_ROOT", tmp_path)
    assert updates.upgrade_command()["cmd"][:1] == ["git"] and "pull" in updates.upgrade_command()["cmd"]


def test_cli_update_check_reports_without_upgrading(root, monkeypatch, capsys):
    fake_pypi(monkeypatch, {"0.3.1": False, "0.3.2": False})
    monkeypatch.setattr(updates, "run_upgrade", lambda **k: (_ for _ in ()).throw(AssertionError("upgraded")))
    import argparse
    cli.cmd_update(argparse.Namespace(check=True))
    out = capsys.readouterr().out
    assert "0.3.2 is available" in out and "kllm update" in out


def test_cli_says_when_already_latest(root, monkeypatch, capsys):
    fake_pypi(monkeypatch, {"0.3.1": False})
    import argparse
    cli.cmd_update(argparse.Namespace(check=False))
    assert "is the latest version" in capsys.readouterr().out


@pytest.mark.parametrize("marker,method", [("uv-receipt.toml", "uv"), ("pipx_metadata.json", "pipx")])
def test_installer_marker_files_win_over_paths(monkeypatch, tmp_path, marker, method):
    env = tmp_path / "custom-tool-dir" / "kllm-gateway"
    env.mkdir(parents=True)
    (env / marker).write_text("", encoding="utf-8")
    monkeypatch.setattr(common, "PKG_ROOT", env / "Lib" / "site-packages")
    monkeypatch.setattr(updates.sys, "prefix", str(env))
    monkeypatch.setattr(updates.sys, "executable", str(env / "Scripts" / "python.exe"))
    assert updates._install_method() == method
