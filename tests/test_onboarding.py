"""First run without any Kaggle credentials never dead-ends: kllm writes a starter config and opens the panel's
welcome page, where the user pastes a token (or drops kaggle.json)."""
import pathlib, sys

import yaml

from gateway import cli, common, rendezvous, setup


def _no_creds(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    for v in ("KAGGLE_CONFIG_DIR", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(v, raising=False)


def test_starter_config_without_credentials(tmp_path, monkeypatch):
    _no_creds(monkeypatch, tmp_path)
    root = tmp_path / "root"
    out = []
    assert setup.ensure_config(root, log=out.append, allow_empty=True) is True
    c = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert c["accounts"] == [] and rendezvous._real(c["rendezvous_secret"]) and c["gateway_api_key"].startswith("sk-kllm-")
    assert "setup page" in out[0]
    assert (root / "models.yaml").exists()


def test_bare_kllm_opens_the_welcome_page(tmp_path, monkeypatch, capsys):
    _no_creds(monkeypatch, tmp_path)
    root = tmp_path / "root"; root.mkdir()
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "CONFIG_FILE", root / "config.yaml")
    started, opened = [], []
    monkeypatch.setattr(common, "ensure_serve", lambda **kw: started.append(kw) or True)
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(sys, "argv", ["kllm"])
    cli.main([])
    assert started and opened == ["http://127.0.0.1:8080/admin#welcome"]
    assert "paste your Kaggle token" in capsys.readouterr().out


def test_up_without_accounts_goes_to_setup_not_a_traceback(tmp_path, monkeypatch, capsys):
    _no_creds(monkeypatch, tmp_path)
    root = tmp_path / "root"; root.mkdir()
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "CONFIG_FILE", root / "config.yaml")
    monkeypatch.setattr(common, "ensure_serve", lambda **kw: True)
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    from gateway import orchestrator
    monkeypatch.setattr(orchestrator, "launch", lambda **kw: (_ for _ in ()).throw(AssertionError("launched")))
    cli.main(["up"])
    assert opened and opened[0].endswith("#welcome")
    assert "no Kaggle account yet" in capsys.readouterr().out
