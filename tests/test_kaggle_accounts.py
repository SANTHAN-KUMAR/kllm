"""Per-account kaggle credentials: kaggle_config_dir / kaggle_json, legacy kgN wrapper fallback."""
import json, os, stat, sys
import pytest
from gateway import kaggle


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle.pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    return tmp_path


def make_json(path, user="alice", key="k123"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"username": user, "key": key}), encoding="utf-8")
    return path


def test_config_dir_sets_kaggle_config_dir(home):
    d = home / "acct1"; make_json(d / "kaggle.json")
    acct = {"name": "kg1", "kaggle_config_dir": str(d)}
    assert kaggle.resolve_credentials(acct) == ("config_dir", str(d))
    env = kaggle.account_env(acct, {"KAGGLE_USERNAME": "stale", "KAGGLE_KEY": "stale", "X": "1"})
    assert env["KAGGLE_CONFIG_DIR"] == str(d) and "KAGGLE_USERNAME" not in env and env["X"] == "1"


def test_config_dir_without_json_is_an_error(home):
    with pytest.raises(kaggle.NoCredentials, match="no kaggle.json"):
        kaggle.resolve_credentials({"name": "kg1", "kaggle_config_dir": str(home)})


def test_kaggle_json_any_filename_becomes_env_vars(home):
    f = make_json(home / "Downloads" / "kaggle (2).json", user="bob", key="sekret")
    acct = {"name": "kg2", "kaggle_json": str(f)}
    env = kaggle.account_env(acct, {})
    assert env["KAGGLE_USERNAME"] == "bob" and env["KAGGLE_KEY"] == "sekret"
    # HOME must be isolated so a global ~/.kaggle/access_token cannot override these creds
    assert env["HOME"].endswith(".kaggle_empty_home") and os.path.isdir(env["HOME"])
    cmd, _ = kaggle.build_command(acct, ["kernels", "status", "bob/x"], {})
    assert "sekret" not in " ".join(cmd) and cmd[-3:] == ["kernels", "status", "bob/x"]


def test_wrapper_account_quota_refuses_instead_of_using_global_token(home, monkeypatch):
    monkeypatch.setattr(kaggle, "resolve_credentials", lambda a: ("wrapper", "/x/kg21"))
    r = kaggle.fetch_account_quota({"name": "kg21"}, {})
    assert r["ok"] is False and "wrapper-only" in r["error"]


def test_kaggle_json_tilde_and_bad_file(home, monkeypatch):
    monkeypatch.setenv("HOME", str(home))
    make_json(home / ".kaggle" / "kaggle.json")
    assert kaggle.resolve_credentials({"name": "a", "kaggle_json": "~/.kaggle/kaggle.json"})[0] == "json"
    (home / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(kaggle.NoCredentials, match="cannot read"):
        kaggle.account_env({"name": "a", "kaggle_json": str(home / "bad.json")}, {})
    with pytest.raises(kaggle.NoCredentials, match="not found"):
        kaggle.resolve_credentials({"name": "a", "kaggle_json": str(home / "nope.json")})


def test_config_dir_wins_over_json_and_wrapper(home):
    d = home / "d"; make_json(d / "kaggle.json")
    w = home / ".local" / "bin" / "kg1"; w.parent.mkdir(parents=True); w.write_text("#!/bin/sh\n", encoding="utf-8")
    acct = {"name": "kg1", "kaggle_config_dir": str(d), "kaggle_json": str(make_json(home / "j.json"))}
    assert kaggle.resolve_credentials(acct)[0] == "config_dir"


def test_legacy_wrapper_is_the_fallback(home):
    w = home / ".local" / "bin" / "kg3"; w.parent.mkdir(parents=True); w.write_text("#!/bin/sh\n", encoding="utf-8")
    assert kaggle.resolve_credentials({"name": "kg3", "user": "u"}) == ("wrapper", str(w))
    cmd, _ = kaggle.build_command({"name": "kg3"}, ["kernels", "list"], {})
    assert cmd == [str(w), "kernels", "list"]


def test_wrapper_found_on_path(home, monkeypatch):
    b = home / "pathbin"; b.mkdir(); w = b / "kg4"; w.write_text("#!/bin/sh\n", encoding="utf-8"); w.chmod(0o755)
    monkeypatch.setenv("PATH", str(b))
    assert kaggle.resolve_credentials({"name": "kg4"})[0] == "wrapper"


def test_no_credentials_message_says_what_to_do(home):
    with pytest.raises(kaggle.NoCredentials, match="kaggle_json"):
        kaggle.resolve_credentials({"name": "kg9"})


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell script as a fake kaggle CLI")
def test_run_executes_cli_with_account_env(home, monkeypatch):
    from gateway import common
    fake = home / "fake-kaggle"
    fake.write_text('#!/bin/sh\necho "user=$KAGGLE_USERNAME key=$KAGGLE_KEY args=$*"\n', encoding="utf-8"); fake.chmod(0o755)
    j = make_json(home / "k.json", user="carol", key="kk")
    monkeypatch.setattr(common, "load_config", lambda: {
        "kaggle_bin": str(fake), "accounts": [{"name": "kg1", "user": "carol", "kaggle_json": str(j)}]})
    rc, out = kaggle.run("kg1", "kernels", "status", "carol/kllm-server")
    assert rc == 0 and "user=carol key=kk args=kernels status carol/kllm-server" in out
    rc, out = kaggle.run("kg7", "kernels", "list")       # unknown account, no wrapper
    assert rc == 127 and "no credentials" in out
    with pytest.raises(RuntimeError):
        kaggle.run("kg7", "x", check=True)


def test_kaggle_command_prefers_configured_then_sibling_script(home, tmp_path, monkeypatch):
    exe = tmp_path / "mykaggle"; exe.write_text("x", encoding="utf-8")
    assert kaggle.kaggle_command({"kaggle_bin": str(exe)}) == [str(exe)]
    assert kaggle.kaggle_command({"kaggle_bin": "~/.venvs/kaggle/bin/kaggle"})[0] != "~/.venvs/kaggle/bin/kaggle"
    cmd = kaggle.kaggle_command({})
    assert cmd[0]   # something runnable is always returned (sibling script, PATH, or python -c)


def test_token_account_env_clears_conflicting_kaggle_vars(home):
    d = home / "h1" / ".kaggle"
    d.mkdir(parents=True)
    tok = "KGAT_" + "ab12" * 8
    (d / "access_token").write_text(tok + "\n", encoding="utf-8")
    env = kaggle.account_env({"name": "kg1", "kaggle_home": str(home / "h1")},
                             {"KAGGLE_CONFIG_DIR": "/stale", "KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k", "KEEP": "1"})
    assert env["KAGGLE_API_TOKEN"] == tok and env["KEEP"] == "1"
    for k in ("KAGGLE_CONFIG_DIR", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        assert k not in env
