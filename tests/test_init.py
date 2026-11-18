"""kllm init: config.yaml from config.example.yaml, accounts, gist via API, gateway key."""
import json, os, stat
import pytest, yaml
from gateway import cli, common, setup, auth


def kjson(tmp_path, name, user):
    p = tmp_path / name
    p.write_text(json.dumps({"username": user, "key": "k-" + user}), encoding="utf-8")
    return str(p)


def run(tmp_path, **kw):
    kw.setdefault("network", False)
    return setup.run_init(root=tmp_path / "root", log=lambda *a: None, **kw)


def test_non_interactive_creates_config_gist_and_key(tmp_path, fake_github):
    a1, a2 = kjson(tmp_path, "one.json", "alice"), kjson(tmp_path, "two (1).json", "bob")
    res = run(tmp_path, accounts=[a1, f"kgx={a2}"], github_token="ghp_realtoken")
    cfg = yaml.safe_load((tmp_path / "root" / "config.yaml").read_text(encoding="utf-8"))
    assert [(a["name"], a["user"], a["kaggle_json"]) for a in cfg["accounts"]] == \
        [("kg1", "alice", a1), ("kgx", "bob", a2)]
    assert cfg["gist_id"] == res["gist_id"] and cfg["gist_id"].startswith("new")
    assert cfg["gist_user"] == "octo" and cfg["gist_token"] == "ghp_realtoken"
    assert auth.real_key(cfg) and cfg["gateway_api_key"] == res["gateway_api_key"]
    assert len(cfg["gateway_api_key"]) > 20 and "kaggle_bin" not in cfg
    # example comments/keys survive
    assert cfg["autoscale"]["max_backends"] == 6 and cfg["litellm_enabled"] is False
    assert fake_github.gists[cfg["gist_id"]]["public"] is False                 # secret gist
    assert (tmp_path / "root" / "models.yaml").exists()
    if os.name != "nt":
        # Only verify 0o600 if the underlying filesystem supports POSIX chmod (NTFS/FAT mounts ignore chmod)
        try:
            cfg_p = tmp_path / "root" / "config.yaml"
            cfg_p.chmod(0o600)
            if stat.S_IMODE(cfg_p.stat().st_mode) == 0o600:
                assert stat.S_IMODE(cfg_p.stat().st_mode) == 0o600
        except Exception:
            pass


def test_without_token_uses_ntfy_with_a_generated_secret(tmp_path):
    from gateway import rendezvous
    out = []
    setup.run_init(root=tmp_path / "r", accounts=[kjson(tmp_path, "k.json", "alice")], network=False, log=out.append)
    cfg = yaml.safe_load((tmp_path / "r" / "config.yaml").read_text(encoding="utf-8"))
    assert rendezvous.mode(cfg) == "ntfy" and "gist_token" not in cfg
    assert rendezvous._real(cfg["rendezvous_secret"]) and len(cfg["rendezvous_secret"]) >= 40
    assert "GitHub" not in "\n".join(out)                                    # nothing GitHub is asked for


def test_existing_gist_id_and_explicit_key(tmp_path, fake_github):
    res = run(tmp_path, accounts=[kjson(tmp_path, "k.json", "alice")], github_token="tok", gist_id="g1",
              api_key="sk-mine-12345")
    cfg = yaml.safe_load((tmp_path / "root" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["gist_id"] == "g1" and cfg["gist_user"] == "octo" and cfg["gateway_api_key"] == "sk-mine-12345"
    assert not any(r[0] == "POST" for r in fake_github.requests)               # no gist created


def test_refuses_to_overwrite_without_force_and_keeps_backup(tmp_path):
    acct = kjson(tmp_path, "k.json", "alice")
    first = run(tmp_path, accounts=[acct])
    with pytest.raises(FileExistsError):
        run(tmp_path, accounts=[acct])
    second = run(tmp_path, accounts=[acct], force=True)
    assert (tmp_path / "root" / "config.yaml.bak").exists()
    assert second["gateway_api_key"] == first["gateway_api_key"]               # existing real key is kept


def test_requires_an_account_and_readable_kaggle_json(tmp_path):
    with pytest.raises(ValueError, match="No Kaggle credentials"):
        run(tmp_path)
    bad = tmp_path / "x.json"; bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="username"):
        run(tmp_path, accounts=[str(bad)])


def test_interactive_prompts_only_for_kaggle(tmp_path, fake_github):
    answers = iter([kjson(tmp_path, "k.json", "alice"), ""])
    asked_secret = []
    res = setup.run_init(root=tmp_path / "r", interactive=True, network=False, log=lambda *a: None,
                         ask=lambda prompt: next(answers), ask_secret=lambda prompt: asked_secret.append(prompt) or "x")
    cfg = yaml.safe_load((tmp_path / "r" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["accounts"][0]["user"] == "alice" and not asked_secret and not res["gist_id"]


def test_init_never_touches_kaggle(tmp_path, monkeypatch):
    from gateway import kaggle
    monkeypatch.setattr(kaggle, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("kaggle called")))
    run(tmp_path, accounts=[kjson(tmp_path, "k.json", "alice")])


def test_prerequisites_report(monkeypatch):
    rows = {r[0]: r for r in setup.prerequisites(network=False)}
    assert rows["python >= 3.10"][1] is True
    assert "kaggle CLI" in rows and "claude (optional)" in rows and "network: ntfy.sh" not in rows
    monkeypatch.setattr(setup, "_reachable", lambda url, timeout=5: url.startswith("https://ntfy.sh"))
    rows = {r[0]: r[1] for r in setup.prerequisites(network=True)}
    assert rows["network: ntfy.sh"] is True and rows["network: kaggle.com"] is False


def test_cli_init_non_interactive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(common, "ROOT", tmp_path / "cliroot")
    cli.main(["init", "--non-interactive", "--no-network", "--account", kjson(tmp_path, "k.json", "alice"),
              "--api-key", "sk-cli-key-123"])
    cfg = yaml.safe_load((tmp_path / "cliroot" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["gateway_api_key"] == "sk-cli-key-123" and cfg["accounts"][0]["name"] == "kg1"
    with pytest.raises(SystemExit):                                  # exists, no --force
        cli.main(["init", "--non-interactive", "--no-network", "--account", kjson(tmp_path, "k.json", "alice")])
