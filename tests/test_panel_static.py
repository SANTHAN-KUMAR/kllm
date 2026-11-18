"""The panel (gateway/ui): no external requests, no browser dialogs, every endpoint it calls exists, lint clean,
served locally with no path traversal."""
import pathlib, re, sys
from fastapi.testclient import TestClient
from gateway import relay

ROOT = pathlib.Path(__file__).resolve().parent.parent
UI = ROOT / "gateway" / "ui"
FILES = [p for p in UI.rglob("*") if p.suffix in (".js", ".html", ".css")]
SRC = {p: p.read_text(encoding="utf-8") for p in FILES}
JS = "\n".join(t for p, t in SRC.items() if p.suffix in (".js", ".html"))


def test_no_external_resources_or_dialogs():
    for p, t in SRC.items():
        assert not re.search(r"""(src|href)=["']https?://|@import\s+url\(["']?https?:|url\(["']?https?:""", t), p
        assert not re.search(r"fetch\(\s*['\"`]https?://", t), p
        assert not re.search(r"\b(?:window\.)?(?:alert|confirm|prompt)\(", t), p
    assert "visibilityState" in JS or "document.hidden" in JS
    assert "prefers-color-scheme" in SRC[UI / "css" / "tokens.css"]
    assert (UI / "css" / "tokens.css").read_text(encoding="utf-8") == (ROOT / "docs" / "redesign" / "tokens.css").read_text(encoding="utf-8")
    assert all((UI / "fonts" / f).stat().st_size > 10000 for f in ("geist.woff2", "geist-mono.woff2", "inter-tight.woff2"))


def _route_matches(path, routes):
    for r in routes:
        if r == path or (path.endswith("/") and r.startswith(path)):
            return True
        if re.fullmatch(re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(r)), path.rstrip("/")):
            return True
    return False


def test_every_api_path_in_the_panel_is_routed():
    paths = set(re.findall(r"['\"`](/admin/api/[a-z0-9/_.-]+)", JS)) | set(re.findall(r"['\"`](/__relay/[a-z]+)", JS))
    assert len(paths) > 60
    routes = set(relay.app.openapi()["paths"])
    missing = sorted(p for p in paths if not _route_matches(p, routes))
    assert not missing, missing


def test_every_page_module_exists():
    app = SRC[UI / "js" / "app.js"]
    mods = set(re.findall(r':\s*"((?:agents/)?[a-z]+)"', app.split("export function parse")[0]))
    mods = {m for m in mods if (UI / "js" / "pages" / f"{m}.js").exists() or "/" in m or m in ("home",)}
    for m in mods:
        assert (UI / "js" / "pages" / f"{m}.js").exists(), m


def test_ui_lint_is_clean(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import ui_lint
    assert ui_lint.lint([str(UI)]) == 0, capsys.readouterr().out


def test_panel_is_served_locally():
    from gateway import common
    common.CONFIG_FILE.write_text("relay_host: 127.0.0.1\nrelay_port: 18199\ndefault_model: x\naccounts: []\n", encoding="utf-8")
    relay._cfg["mtime"] = 0
    try:
        c = TestClient(relay.app, client=("127.0.0.1", 5))
        r = c.get("/admin")
        assert r.status_code == 200 and "kllm control panel" in r.text and r.headers["cache-control"] == "no-store"
        assert c.get("/chat").status_code == 200
        js = c.get("/admin/ui/js/app.js")
        assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
        assert c.get("/admin/ui/fonts/geist.woff2").headers["content-type"] == "font/woff2"
        for bad in ("/admin/ui/../relay.py", "/admin/ui/%2e%2e/relay.py", "/admin/ui/js/../../relay.py", "/admin/ui/nope.js"):
            assert c.get(bad).status_code == 404, bad
        remote = TestClient(relay.app, client=("203.0.113.9", 5))
        assert remote.get("/admin").status_code == 403
    finally:
        common.CONFIG_FILE.unlink(missing_ok=True)
        relay._cfg["mtime"] = 0
