"""Test isolation: kllm state/logs/config go to a throwaway KLLM_ROOT, never the checkout."""
import json, os, pathlib, shutil, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="kllm-test-root-"))
os.environ["KLLM_ROOT"] = str(_ROOT)
shutil.copy(HERE / "models.yaml", _ROOT / "models.yaml")

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """No test may touch the real home directory (Path.home() reads USERPROFILE on Windows, HOME elsewhere).
    Tests that set HOME themselves get USERPROFILE mirrored, so they stay isolated on Windows too."""
    h = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    real_setenv = monkeypatch.setenv

    def setenv(name, value, prepend=None):
        real_setenv(name, value, prepend)
        if name == "HOME":
            real_setenv("USERPROFILE", value, prepend)
    monkeypatch.setenv = setenv
    # a real `kllm serve` on this machine (port 8080) must not change what the CLI does inside a test
    from gateway import common
    monkeypatch.setattr(common, "relay_up", lambda *a, **k: False)
    # no test may reach PyPI for update checks
    monkeypatch.setenv("KLLM_NO_UPDATE_CHECK", "1")
    yield


class FakeGitHub:
    """Just enough of the GitHub gist REST API: GET/PATCH /gists/<id>, POST /gists, GET /user."""
    def __init__(self):
        self.gists = {"g1": {"id": "g1", "files": {}, "owner": {"login": "octo"}}}
        self.requests = []          # (method, path, auth header, json body)
        self.fail = None            # set to an HTTP status to fail every call
        self.etag_n = 1
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, obj=None, headers=None):
                body = json.dumps(obj).encode() if obj is not None else b""
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            def _handle(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                outer.requests.append((method, self.path, self.headers.get("Authorization"), body))
                if outer.fail:
                    return self._send(outer.fail, {"message": "boom"})
                if not (self.headers.get("Authorization") or "").startswith("Bearer "):
                    return self._send(401, {"message": "Requires authentication"})
                if method == "GET" and self.path == "/user":
                    return self._send(200, {"login": "octo"})
                if method == "POST" and self.path == "/gists":
                    gid = f"new{len(outer.gists)}"
                    outer.gists[gid] = {"id": gid, "owner": {"login": "octo"}, "public": body["public"],
                                        "files": {k: {"content": v["content"]} for k, v in body["files"].items()}}
                    return self._send(201, outer.gists[gid])
                gid = self.path.rsplit("/", 1)[-1]
                g = outer.gists.get(gid)
                if g is None:
                    return self._send(404, {"message": "Not Found"})
                if method == "PATCH":
                    for k, v in body["files"].items():
                        g["files"][k] = {"content": v["content"]}
                    outer.etag_n += 1
                    return self._send(200, g, {"ETag": f'"{outer.etag_n}"'})
                tag = f'"{outer.etag_n}"'
                if self.headers.get("If-None-Match") == tag:
                    return self._send(304, None, {"ETag": tag})
                return self._send(200, g, {"ETag": tag})
            def do_GET(self): self._handle("GET")
            def do_PATCH(self): self._handle("PATCH")
            def do_POST(self): self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def set_status(self, account, **ev):
        self.gists["g1"]["files"][f"s-{account}.json"] = {"content": json.dumps({"account": account, **ev})}
        self.etag_n += 1


@pytest.fixture
def fake_github(monkeypatch):
    from gateway import rendezvous
    gh = FakeGitHub()
    monkeypatch.setenv("KLLM_GITHUB_API", gh.url)
    rendezvous._etag.clear()
    yield gh
    gh.server.shutdown()
