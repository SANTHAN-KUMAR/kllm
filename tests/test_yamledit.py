"""Comment-preserving YAML edits: accounts in config.yaml, model entries in models.yaml."""
import pathlib
import pytest, yaml
from gateway import yamledit as y

HERE = pathlib.Path(__file__).resolve().parent.parent
CFG = (HERE / "config.example.yaml").read_text(encoding="utf-8")
MODELS = (HERE / "models.yaml").read_text(encoding="utf-8")


def accts(text):
    return (yaml.safe_load(text) or {}).get("accounts")


def test_disable_flow_account_keeps_comments_and_other_lines():
    new = y.edit_account(CFG, "kg2", set_fields={"disabled": True})
    assert accts(new)[1]["disabled"] is True and "disabled" not in accts(new)[0]
    assert "# ---- Kaggle accounts (rotation pool)" in new
    old_lines, new_lines = CFG.splitlines(), new.splitlines()
    assert len(old_lines) == len(new_lines)
    assert sum(a != b for a, b in zip(old_lines, new_lines)) == 1       # exactly one line changed
    again = y.edit_account(new, "kg2", drop_fields=("disabled",))
    assert accts(again) == accts(CFG)


def test_block_style_accounts_and_trailing_comment():
    cfg = ("a: 1\naccounts:\n  - name: kg1   # main\n    user: u1\n    kaggle_json: /x.json\n"
           "  - {name: kg2, user: u2}   # second\nz: 2\n")
    new = y.edit_account(cfg, "kg1", set_fields={"disabled": True})
    assert accts(new)[0] == {"name": "kg1", "user": "u1", "kaggle_json": "/x.json", "disabled": True}
    assert yaml.safe_load(new)["z"] == 2 and "# main" in new
    new = y.edit_account(new, "kg2", set_fields={"disabled": True})
    assert "# second" in new and accts(new)[1]["disabled"] is True
    gone = y.remove_account(new, "kg1")
    assert [a["name"] for a in accts(gone)] == ["kg2"] and "z: 2" in gone


def test_remove_and_append_account():
    new = y.remove_account(CFG, "kg1")
    assert [a["name"] for a in accts(new)] == ["kg2"]
    new = y.append_account(CFG, {"name": "kg3", "user": "third"})
    assert [a["name"] for a in accts(new)] == ["kg1", "kg2", "kg3"]
    assert new.replace("  - {name: kg3, user: third}\n", "") == CFG
    empty = y.append_account("default_model: x\n", {"name": "kg1", "user": "u"})
    assert accts(empty) == [{"name": "kg1", "user": "u"}]
    with pytest.raises(y.EditError):
        y.edit_account(CFG, "nope", set_fields={"disabled": True})
    with pytest.raises(y.EditError):
        y.remove_account(CFG, "nope")


def test_edit_model_changes_only_what_changed():
    new = y.update_model(MODELS, "qwen3-coder-30b-abliterated",
                         {"ctx": 65536, "desc": 'say "hi" — ok', "args": ["--temp", "0.5", "-sm", "tensor"]})
    m = yaml.safe_load(new)["models"]
    assert m["qwen3-coder-30b-abliterated"]["ctx"] == 65536
    assert m["qwen3-coder-30b-abliterated"]["desc"] == 'say "hi" — ok'
    assert m["qwen3-coder-30b-abliterated"]["args"] == ["--temp", "0.5", "-sm", "tensor"]
    orig = yaml.safe_load(MODELS)["models"]
    assert {k: v for k, v in m.items() if k != "qwen3-coder-30b-abliterated"} == \
           {k: v for k, v in orig.items() if k != "qwen3-coder-30b-abliterated"}
    assert "# full attention: 96 KiB KV/token" in new            # the ctx comment survived
    section_comments = [l for l in MODELS.splitlines() if l.strip().startswith("# ----")]
    assert section_comments and all(c in new for c in section_comments)   # every section comment survived
    assert new.count("# hybrid model") == MODELS.count("# hybrid model")


def test_edit_model_add_remove_fields_and_noop():
    assert y.update_model(MODELS, "validate-3b", {"ctx": 8192}) == MODELS      # unchanged -> untouched
    new = y.update_model(MODELS, "validate-3b", {"template": "templates/x.jinja", "vram_gb": 4.5, "tags": ["a", "b"]})
    e = yaml.safe_load(new)["models"]["validate-3b"]
    assert e["template"] == "templates/x.jinja" and e["vram_gb"] == 4.5 and e["tags"] == ["a", "b"]
    back = y.update_model(new, "validate-3b", {"template": None})
    assert "template" not in yaml.safe_load(back)["models"]["validate-3b"]
    img = y.update_model(MODELS, "sdxl-turbo", {"opts": {"size": "768x768", "steps": 2}})
    assert yaml.safe_load(img)["models"]["sdxl-turbo"]["opts"] == {"size": "768x768", "steps": 2}
    with pytest.raises(y.EditError):
        y.update_model(MODELS, "nope", {"ctx": 1})


def test_long_args_round_trip_with_awkward_strings():
    args = ["--chat-template-kwargs", '{"enable_thinking": false}', "--temp", "0.6", "-x: y", "#hash", "it's"] * 4
    new = y.update_model(MODELS, "validate-3b", {"args": args})
    assert yaml.safe_load(new)["models"]["validate-3b"]["args"] == args
    val_lines = [l for l in new.split("validate-3b:")[1].split("desc:")[0].splitlines() if l.strip()]
    assert max(len(l) for l in val_lines) < 130


def test_delete_model_removes_block_only():
    new = y.delete_model(MODELS, "x-qwen36-512k")
    m = yaml.safe_load(new)["models"]
    assert "x-qwen36-512k" not in m and "sdxl-turbo" in m and "validate-3b" in m
    assert len(m) == len(yaml.safe_load(MODELS)["models"]) - 1
    assert "# ---- image / video (diffusers)" in new
    with pytest.raises(y.EditError):
        y.delete_model(MODELS, "nope")
    last = y.delete_model(MODELS, "ltx-video")
    assert "ltx-video" not in yaml.safe_load(last)["models"]


def test_atomic_write_validates_and_keeps_backup(tmp_path):
    f = tmp_path / "c.yaml"; f.write_text("a: 1\n", encoding="utf-8")
    y.atomic_write(f, "a: 2\n")
    assert f.read_text(encoding="utf-8") == "a: 2\n" and (tmp_path / "c.yaml.bak").read_text(encoding="utf-8") == "a: 1\n"
    with pytest.raises(y.EditError):
        y.atomic_write(f, "a: [1\n")
    assert f.read_text(encoding="utf-8") == "a: 2\n"
