"""Past image and video outputs, for the Image and Video pages (GET /admin/api/media).

The relay hands every successful /v1/images/generations and /v1/videos/generations response to save() after it
has streamed it to the client. Files go to <root>/media/<kind>/, one JSON line per output in media/index.jsonl.
The folder is capped (config `media_max_mb`, default 2048) and the oldest outputs are removed first.
"""
import base64, json, threading, time, uuid
from . import common

MAX_CAPTURE = 300 * 1024 * 1024        # never buffer more than this from one response
_LOCK = threading.Lock()


def media_dir():
    return common.ROOT / "media"


def _index():
    return media_dir() / "index.jsonl"


def _ext(raw, kind):
    if kind == "video":
        return "mp4"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return "bin"


def _decode(item, kind):
    """-> raw bytes of one output item, or None."""
    for k in ("b64_mp4", "b64_json"):
        if isinstance(item.get(k), str):
            try:
                return base64.b64decode(item[k], validate=False)
            except ValueError:
                return None
    u = item.get("url")
    if isinstance(u, str) and u.startswith("data:") and "," in u:
        try:
            return base64.b64decode(u.split(",", 1)[1], validate=False)
        except ValueError:
            return None
    return None


def save(kind, request_payload, response_body, model=None, backend=None, max_mb=2048, now=None):
    """Store the outputs of one generation response. Returns the new index rows."""
    try:
        doc = json.loads(response_body)
    except (ValueError, UnicodeDecodeError):
        return []
    items = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(items, list):
        return []
    req = request_payload if isinstance(request_payload, dict) else {}
    now = now or time.time()
    rows = []
    with _LOCK:
        d = media_dir() / kind
        d.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            raw = _decode(item, kind)
            if not raw:
                continue
            mid = f"{int(now)}-{uuid.uuid4().hex[:8]}-{i}"
            name = f"{mid}.{_ext(raw, kind)}"
            (d / name).write_bytes(raw)
            row = {"id": mid, "kind": kind, "file": f"{kind}/{name}", "bytes": len(raw), "ts": round(now, 1),
                   "prompt": str(req.get("prompt") or "")[:2000], "model": model or req.get("model"),
                   "size": req.get("size"), "backend": backend, "seed": item.get("seed")}
            for k in ("seconds", "duration", "fps", "num_frames", "quality", "steps"):
                if req.get(k) is not None:
                    row[k] = req[k]
            rows.append(row)
        if rows:
            with open(_index(), "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            _prune(max_mb)
    return rows


def _read_all():
    try:
        lines = _index().read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _write_all(rows):
    tmp = _index().with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(_index())


def _prune(max_mb):
    rows = [r for r in _read_all() if (media_dir() / r.get("file", "")).is_file()]
    total = sum(r.get("bytes", 0) for r in rows)
    cap = max(1, int(max_mb)) * 1024 * 1024
    while rows and total > cap:
        old = rows.pop(0)
        total -= old.get("bytes", 0)
        try:
            (media_dir() / old["file"]).unlink()
        except OSError:
            pass
    _write_all(rows)


def list_media(kind=None, limit=100):
    rows = [r for r in _read_all() if (kind is None or r.get("kind") == kind)
            and (media_dir() / r.get("file", "")).is_file()]
    rows.reverse()
    return rows[:max(1, min(int(limit), 1000))]


def path_of(media_id):
    for r in _read_all():
        if r.get("id") == media_id:
            p = (media_dir() / r["file"]).resolve()
            if media_dir().resolve() in p.parents and p.is_file():
                return p, r
    return None, None


def delete(media_id):
    with _LOCK:
        rows = _read_all()
        keep = [r for r in rows if r.get("id") != media_id]
        if len(keep) == len(rows):
            return False
        for r in rows:
            if r.get("id") == media_id:
                try:
                    (media_dir() / r["file"]).unlink()
                except OSError:
                    pass
        _write_all(keep)
    return True
