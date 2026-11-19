"""kllm uninstall: shows a plan first, needs a yes, never deletes a checkout or the user's own Kaggle files,
keeps going when one step fails, and removes the package with the tool that installed it."""
import argparse, pathlib

import pytest

from gateway import cli, common, uninstall, updates


def test_dedicated_root_is_removed_whole(tmp_path):
    root = tmp_path / "kllm"
    (root / "state").mkdir(parents=True); (root / "config.yaml").write_text("x", encoding="utf-8")
    assert uninstall.files_to_remove(root) == [root]


def test_a_checkout_root_loses_only_runtime_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (tmp_path / "gateway").mkdir(); (tmp_path / "state").mkdir(); (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
    got = {p.name for p in uninstall.files_to_remove(tmp_path)}
    assert got == {"state", "config.yaml"} and tmp_path not in uninstall.files_to_remove(tmp_path)


def test_users_own_kaggle_folder_is_never_in_the_plan(tmp_path, monkeypatch):
    home = tmp_path / "home"; (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(common, "ROOT", tmp_path / "kllm")
    (tmp_path / "kllm").mkdir()
    monkeypatch.setattr(uninstall, "relay_running", lambda: False)
    monkeypatch.setattr(uninstall, "tool_undos", lambda: [])
    monkeypatch.setattr(common, "load_config", lambda: {"accounts": []})
    monkeypatch.setattr(common, "read_state", lambda: {"backends": []})
    paths = [str(d) for _, k, d in uninstall.plan() if k == "path"]
    assert not any(".kaggle" in p and "kaggle_home" not in p for p in paths)


def test_keep_config_keeps_files_and_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path / "kllm"); (tmp_path / "kllm").mkdir()
    monkeypatch.setattr(uninstall, "relay_running", lambda: True)
    monkeypatch.setattr(uninstall, "tool_undos", lambda: [("opencode", "models", "opencode: Run on kllm's models")])
    monkeypatch.setattr(uninstall, "credential_files", lambda cfg: [tmp_path / ".kaggle_home2"])
    monkeypatch.setattr(common, "load_config", lambda: {"accounts": [{"name": "kg2"}]})
    monkeypatch.setattr(common, "read_state", lambda: {"backends": [{"account": "kg2"}]})
    kinds = [k for _, k, _ in uninstall.plan(keep_config=True)]
    assert kinds[:2] == ["gpus", "relay"] and "tool" in kinds and "path" not in kinds
    assert "path" in [k for _, k, _ in uninstall.plan(keep_config=False)]


@pytest.mark.parametrize("method,first", [("uv", "uv"), ("pipx", "pipx"), ("pip", None), ("checkout", "none")])
def test_package_removed_with_the_installing_tool(monkeypatch, method, first):
    monkeypatch.setattr(updates, "upgrade_command", lambda: {"method": method, "cmd": [], "display": []})
    m, cmd = uninstall.package_command()
    if first == "none":
        assert cmd is None
    elif first is None:
        assert cmd[1:4] == ["-m", "pip", "uninstall"] and "-y" in cmd
    else:
        assert cmd[0] == first and "uninstall" in cmd and "kllm-gateway" in cmd


def test_one_failing_step_does_not_stop_the_rest(tmp_path, monkeypatch):
    victim = tmp_path / "state"; victim.mkdir()
    monkeypatch.setattr(uninstall, "stop_relay", lambda: False)
    out = []
    failed = uninstall.run([("Stop the relay", "relay", None), ("Delete state", "path", victim)], log=out.append)
    assert failed == 1 and not victim.exists() and "FAIL" in out[0] and "done" in out[1]


def test_windows_package_removal_runs_after_exit(monkeypatch):
    calls = []
    monkeypatch.setattr(uninstall.os, "name", "nt")
    monkeypatch.setattr(uninstall.subprocess, "Popen", lambda *a, **k: calls.append(a[0]))
    uninstall._remove_package(["uv", "tool", "uninstall", "kllm-gateway"], log=lambda *a: None)
    assert calls and "uninstall" in calls[0][-1]                # a detached helper, not a direct call


def _args(**kw):
    return argparse.Namespace(**{"keep_config": False, "dry_run": False, "yes": False, **kw})


def test_cli_needs_an_explicit_yes(monkeypatch, capsys):
    monkeypatch.setattr(uninstall, "plan", lambda keep_config=False: [("Delete everything", "path", "/nope")])
    monkeypatch.setattr(uninstall, "run", lambda steps: (_ for _ in ()).throw(AssertionError("ran without a yes")))
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    cli.cmd_uninstall(_args())
    assert "nothing was removed" in capsys.readouterr().out


def test_cli_dry_run_only_lists(monkeypatch, capsys):
    monkeypatch.setattr(uninstall, "plan", lambda keep_config=False: [("Delete X", "path", "/x")])
    monkeypatch.setattr(uninstall, "run", lambda steps: (_ for _ in ()).throw(AssertionError("ran on a dry run")))
    cli.cmd_uninstall(_args(dry_run=True))
    out = capsys.readouterr().out
    assert "Delete X" in out and "Kept: your own Kaggle credentials" in out
