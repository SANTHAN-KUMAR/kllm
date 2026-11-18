"""ntfy.sh URL handoff: the Kaggle kernel publishes JSON status messages; the
orchestrator reads them. We use the JSON polling API (since=...) rather than the
websocket so it works with just urllib/requests and is easy to replay."""
import json, time, urllib.request, urllib.parse

def _get(url, timeout):
    req = urllib.request.Request(url, headers={"Accept": "application/x-ndjson"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

def poll(server, topic, since="all", timeout=30):
    """Return a list of parsed kllm events (dicts) published to the topic.
    `since` can be 'all', a unix ts, or e.g. '10m'. Only message events with JSON bodies."""
    base = server.rstrip("/")
    url = f"{base}/{topic}/json?poll=1&since={urllib.parse.quote(str(since))}"
    events = []
    try:
        raw = _get(url, timeout)
    except Exception:
        return events
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("event") != "message":
            continue
        try:
            body = json.loads(msg.get("message", ""))
        except json.JSONDecodeError:
            continue
        body["_time"] = msg.get("time")
        body["_id"] = msg.get("id")
        events.append(body)
    return events

def wait_for(server, topic, want_event, since, overall_timeout, on_event=None):
    """Block until an event with event==want_event appears, or timeout.
    Returns the matching event dict, or None. Calls on_event(ev) for every event seen."""
    deadline = time.time() + overall_timeout
    seen = set()
    while time.time() < deadline:
        for ev in poll(server, topic, since=since, timeout=30):
            key = ev.get("_id")
            if key in seen:
                continue
            seen.add(key)
            if on_event:
                on_event(ev)
            if ev.get("event") == want_event:
                return ev
            if ev.get("event") in ("error", "crash"):
                return ev  # surface failures to the caller
        time.sleep(4)
    return None
