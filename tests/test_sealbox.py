"""Sealed ntfy rendezvous: PC and kernel interoperate, forged/altered/foreign messages are dropped, the PC polls
incrementally, stop commands reach the kernel, and first-run config needs nothing but Kaggle credentials."""
import json, pathlib, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kernel_loader import load_kernel
from gateway import rendezvous, sealbox, setup

SECRET = "s3cret-" + "x" * 40


class FakeNtfy:
    """POST /<topic> stores a message; GET /<topic>/json?poll=1&since=<id|unix|12h> lists newer ones."""
    def __init__(self):
        self.msgs = {}           # topic -> [ {id, time, event, message} ]
        self.gets = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                topic = self.path.strip("/")
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                lst = outer.msgs.setdefault(topic, [])
                lst.append({"id": f"m{sum(map(len, outer.msgs.values()))}", "time": int(time.time()),
                            "event": "message", "message": body})
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                topic = u.path.strip("/").split("/")[0]
                since = urllib.parse.parse_qs(u.query).get("since", ["all"])[0]
                outer.gets.append((topic, since))
                lst = outer.msgs.get(topic, [])
                ids = [m["id"] for m in lst]
                if since in ids:
                    lst = lst[ids.index(since) + 1:]
                elif since.isdigit():
                    lst = [m for m in lst if m["time"] >= int(since)]
                body = "".join(json.dumps(m) + "\n" for m in lst).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def ntfy(monkeypatch):
    f = FakeNtfy()
    monkeypatch.setenv("KLLM_NTFY_SERVER", f.url)
    monkeypatch.setattr(rendezvous, "NTFY_MIN_POLL_S", 0.0)
    rendezvous._NTFY.clear()
    yield f
    f.server.shutdown()


def cfg():
    return {"rendezvous": "ntfy", "rendezvous_secret": SECRET}


def kernel(server_url):
    mod = load_kernel("validate-3b", cfg={"rendezvous": "ntfy", "rendezvous_secret": SECRET, "gist_id": ""})
    mod.NTFY_SERVER = server_url
    return mod


# ---- crypto --------------------------------------------------------------------------------------------------------
def test_roundtrip_tamper_and_wrong_secret():
    box = sealbox.seal(SECRET, {"event": "ready", "url": "https://a.trycloudflare.com", "ts": 1})
    assert "trycloudflare" not in box and box.startswith("v1.")
    assert sealbox.unseal(SECRET, box)["url"] == "https://a.trycloudflare.com"
    assert sealbox.unseal("other-secret", box) is None
    flipped = box[:20] + ("A" if box[20] != "A" else "B") + box[21:]
    assert sealbox.unseal(SECRET, flipped) is None
    assert sealbox.unseal(SECRET, "v1.short") is None and sealbox.unseal(SECRET, '{"event":"ready"}') is None
    assert sealbox.seal(SECRET, {"a": 1}) != sealbox.seal(SECRET, {"a": 1})          # fresh nonce every time


def test_kernel_copy_interoperates_with_gateway(tmp_path):
    k = kernel("http://unused")
    assert k._sb_topics() == sealbox.topics(SECRET)
    assert sealbox.unseal(SECRET, k._sb_seal({"x": 1})) == {"x": 1}
    assert k._sb_unseal(sealbox.seal(SECRET, {"y": 2})) == {"y": 2}
    assert k.RV_MODE == "ntfy"


def test_topics_are_unguessable_and_distinct():
    s, c = sealbox.topics(SECRET)
    assert s != c and len(s) >= 40 and SECRET not in s
    assert sealbox.topics(sealbox.new_secret()) != (s, c)


# ---- end to end through a fake ntfy -------------------------------------------------------------------------------
def test_kernel_status_reaches_pc_and_forgeries_are_dropped(ntfy):
    k = kernel(ntfy.url)
    k.CFG["account"] = "kg1"
    assert k._ntfy_publish({"event": "ready", "account": "kg1", "url": "https://good.trycloudflare.com", "ts": int(time.time())})
    status, _ = sealbox.topics(SECRET)
    # an attacker who learned the topic posts a plain and a wrongly-sealed fake URL
    ntfy.msgs[status].append({"id": "evil1", "time": int(time.time()), "event": "message",
                              "message": json.dumps({"event": "ready", "account": "kg1", "url": "https://evil.example",
                                                     "ts": int(time.time()) + 99})})
    ntfy.msgs[status].append({"id": "evil2", "time": int(time.time()), "event": "message",
                              "message": sealbox.seal("guess", {"event": "ready", "account": "kg1",
                                                                "url": "https://evil.example", "ts": int(time.time()) + 99})})
    ok, evs = rendezvous.latest_events_checked(cfg())
    assert ok and [e["url"] for e in evs] == ["https://good.trycloudflare.com"]


def test_pc_polls_incrementally_and_keeps_newest_per_account(ntfy):
    status, _ = sealbox.topics(SECRET)
    now = int(time.time())
    for acct, ev, ts in (("kg1", "boot", now - 50), ("kg2", "ready", now - 40), ("kg1", "ready", now - 10)):
        ntfy.msgs.setdefault(status, []).append({"id": f"i{ts}", "time": ts, "event": "message",
                                                 "message": sealbox.seal(SECRET, {"event": ev, "account": acct, "ts": ts})})
    evs = rendezvous.latest_events(cfg())
    assert [(e["account"], e["event"]) for e in evs] == [("kg1", "ready"), ("kg2", "ready")]
    rendezvous.latest_events(cfg())
    assert ntfy.gets[0][1] == "12h" and ntfy.gets[-1][1] == f"i{now - 10}"           # second poll: only newer


def test_stop_command_reaches_the_kernel_and_old_stops_are_ignored(ntfy):
    k = kernel(ntfy.url)
    k.CFG["account"] = "kg3"
    started = time.time()
    assert k.stop_requested(started) is False
    assert rendezvous.publish_control(cfg(), "kg9", "stop")                          # another account's stop
    assert k.stop_requested(started) is False
    assert rendezvous.publish_control(cfg(), "kg3", "stop")
    assert k.stop_requested(started) is True
    assert k.stop_requested(time.time() + 3600) is False                              # a later session ignores it


def test_unreachable_ntfy_is_reported_as_unknown(monkeypatch):
    monkeypatch.setenv("KLLM_NTFY_SERVER", "http://127.0.0.1:9")
    monkeypatch.setattr(rendezvous, "NTFY_MIN_POLL_S", 0.0)
    rendezvous._NTFY.clear()
    assert rendezvous.latest_events_checked(cfg()) == (False, [])


def test_gist_token_never_reaches_the_kernel_in_ntfy_mode():
    k = load_kernel("validate-3b", cfg={"rendezvous": "ntfy", "rendezvous_secret": SECRET,
                                        "gist_id": "g", "gist_token": "ghp_must_not_leak"})
    assert k.CFG["gist_token"] == "" and k.CFG["gist_id"] == "" and k.CFG["rendezvous_secret"] == SECRET


# ---- zero-config first run -----------------------------------------------------------------------------------------
def test_first_run_creates_config_from_kaggle_json(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text(json.dumps({"username": "alice", "key": "k"}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    for v in ("KAGGLE_CONFIG_DIR", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    root = tmp_path / "root"
    assert setup.ensure_config(root, log=lambda *a: None) is True
    c = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert c["accounts"][0]["user"] == "alice" and rendezvous.mode(c) == "ntfy"
    assert rendezvous._real(c["rendezvous_secret"]) and c["gateway_api_key"].startswith("sk-kllm-")


def test_first_run_from_env_credentials_stores_a_private_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "bob"); monkeypatch.setenv("KAGGLE_KEY", "kk")
    acct = setup.detect_account(tmp_path / "root", log=lambda *a: None)
    assert acct["user"] == "bob" and json.loads(pathlib.Path(acct["kaggle_json"]).read_text(encoding="utf-8"))["key"] == "kk"


def test_no_credentials_explains_the_one_step(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    for v in ("KAGGLE_CONFIG_DIR", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    out = []
    assert setup.ensure_config(tmp_path / "root", log=out.append) is False
    assert "kaggle.json" in out[0] and not (tmp_path / "root" / "config.yaml").exists()


def test_old_config_gets_a_secret_but_keeps_its_gist(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    (root / "config.yaml").write_text("# mine\nrendezvous: gist\ngist_id: abc123\naccounts: []\n", encoding="utf-8")
    assert setup.ensure_config(root, log=lambda *a: None)
    c = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert rendezvous.mode(c) == "gist" and rendezvous._real(c["rendezvous_secret"])
    assert (root / "config.yaml").read_text(encoding="utf-8").startswith("# mine")
