"""Account add (mocked Kaggle check), disable/enable, remove with credential-file safety."""
import json, os, stat, subprocess, types
import pytest, yaml
from gateway import accounts, common

TOKEN = "KGAT_" + "ab12" * 8


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts.pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    test_cfg = tmp_path / "config.yaml"
    test_state = tmp_path / "state.json"
    test_cfg.write_text("default_model: m\naccounts:\n  - {name: kg1, user: alice}  # first\nkernel_slug: s\n", encoding="utf-8")
    test_state.write_text('{"backends": [], "pending": {}}', encoding="utf-8")
    monkeypatch.setattr(common, "CONFIG_FILE", test_cfg)
    monkeypatch.setattr(common, "STATE_FILE", test_state)
    yield tmp_path


def fake_run(user="bob", fail_list=False, calls=None, envs=None):
    """Stands in for the Python probe add_account runs with the account's credentials in its environment."""
    def run(cmd, **kw):
        if calls is not None: calls.append(cmd)
        if envs is not None: envs.append(kw.get("env") or {})
        return types.SimpleNamespace(stdout=f"username: {user}\n", stderr="403 Forbidden" if fail_list else "",
                                     returncode=1 if fail_list else 0)
    return run


def test_add_account_creates_creds_wrapper_and_config_line(env):
    calls, envs = [], []
    r = accounts.add_account(TOKEN, run=fake_run(calls=calls, envs=envs))
    assert r["name"] == "kg2" and r["user"] == "bob"
    tok = env / ".kaggle_home2" / ".kaggle" / "access_token"
    assert tok.read_text(encoding="utf-8").strip() == TOKEN and (os.name == "nt" or stat.S_IMODE(tok.stat().st_mode) in (0o600, 0o777))
    w = env / ".local" / "bin" / "kg2"
    if os.name != "nt":                                          # convenience wrapper for the kaggle CLI by hand
        assert w.exists() and os.access(w, os.X_OK) and ".kaggle_home2" in w.read_text(encoding="utf-8")
    else:
        assert not w.exists() and r["wrapper"] is None           # a #!/bin/sh script cannot run on Windows
    cfg = yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))
    assert cfg["accounts"] == [{"name": "kg1", "user": "alice"},
                               {"name": "kg2", "user": "bob", "kaggle_home": str(env / ".kaggle_home2")}]
    assert "# first" in common.CONFIG_FILE.read_text(encoding="utf-8") and cfg["kernel_slug"] == "s"
    assert all(TOKEN not in " ".join(c) for c in calls)          # never on a command line
    assert envs[0].get("KAGGLE_API_TOKEN") == TOKEN              # the check runs with the account's own token


def test_add_account_from_kaggle_json_contents(env):
    calls = []
    pasted = json.dumps({"username": "carol", "key": "0123456789abcdef0123456789abcdef"})
    r = accounts.add_account(pasted, run=fake_run(user="carol", calls=calls))
    f = env / ".kaggle_home2" / ".kaggle" / "kaggle.json"
    assert r["user"] == "carol" and json.loads(f.read_text(encoding="utf-8"))["key"].startswith("0123")
    cfg = yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))
    assert cfg["accounts"][-1] == {"name": "kg2", "user": "carol", "kaggle_config_dir": str(env / ".kaggle_home2" / ".kaggle")}
    assert all("0123456789abcdef" not in " ".join(c) for c in calls)
    with pytest.raises(accounts.AccountError, match="username"):
        accounts.add_account('{"username": "x"}', run=fake_run())


@pytest.mark.parametrize("token", ["", "abc", "KGAT_xyz", "KGAT_" + "g" * 32, "not json {"])
def test_add_account_rejects_malformed_token(env, token):
    with pytest.raises(accounts.AccountError, match="KGAT_"):
        accounts.add_account(token, run=fake_run())
    assert not (env / ".kaggle_home2").exists()


def test_add_account_rejected_by_kaggle_leaves_nothing(env):
    with pytest.raises(accounts.AccountError, match="rejected") as e:
        accounts.add_account(TOKEN, run=fake_run(fail_list=True))
    assert TOKEN not in str(e.value)
    assert not (env / ".local" / "bin" / "kg2").exists()
    assert not (env / ".kaggle_home2" / ".kaggle" / "access_token").exists()
    assert len(yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["accounts"]) == 1


def test_add_account_duplicates(env):
    with pytest.raises(accounts.AccountError, match="already configured"):
        accounts.add_account(TOKEN, run=fake_run(user="alice"))
    with pytest.raises(accounts.AccountError, match="already exists"):
        accounts.add_account(TOKEN, name="kg1", run=fake_run())
    with pytest.raises(accounts.AccountError, match="account name"):
        accounts.add_account(TOKEN, name="../evil", run=fake_run())


def test_disable_enable(env):
    accounts.set_disabled("kg1", True)
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["accounts"][0]["disabled"] is True
    assert "# first" in common.CONFIG_FILE.read_text(encoding="utf-8")
    accounts.set_disabled("kg1", False)
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["accounts"] == [{"name": "kg1", "user": "alice"}]


def test_remove_keeps_credentials_unless_explicitly_deleted(env):
    accounts.add_account(TOKEN, run=fake_run())
    home, tok, wrapper = accounts.credential_paths("kg2")
    assert accounts.remove_account("kg2") == []                  # config only
    assert tok.exists() and (os.name == "nt" or wrapper.exists())
    assert [a["name"] for a in yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["accounts"]] == ["kg1"]
    accounts.add_account(TOKEN, name="kg2", run=fake_run())
    gone = accounts.remove_account("kg2", delete_credentials=True)
    want = {str(home)} | ({str(wrapper)} if os.name != "nt" else set())
    assert set(gone) == want and not home.exists() and not wrapper.exists()


def test_remove_refuses_running_account_and_never_touches_foreign_files(env):
    common.write_state({"backends": [{"account": "kg1"}], "pending": {}})
    with pytest.raises(accounts.AccountError, match="stop it first"):
        accounts.remove_account("kg1")
    common.write_state({"backends": [], "pending": {}})
    foreign = env / ".local" / "bin" / "kg1"; foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
    (env / ".kaggle_home1").mkdir()
    assert [p.name for p in accounts.removable_credentials("kg1")] == [".kaggle_home1"]   # not a foreign wrapper
    accounts.remove_account("kg1", delete_credentials=True)
    assert foreign.exists()


def test_add_account_rejects_disallowed_prefix(env):
    cfg = {"accounts": [{"name": "kg1", "user": "alice"}], "account_allow_prefixes": ["kg"]}
    with pytest.raises(accounts.AccountError, match="not permitted by account_allow_prefixes"):
        accounts.add_account(TOKEN, name="lk4", cfg=cfg, run=fake_run())

