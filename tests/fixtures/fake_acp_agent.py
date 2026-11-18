"""A tiny ACP agent for tests: speaks JSON-RPC on stdio like `opencode acp`, never calls a model.

Prompt directives (anywhere in the prompt text):
  WRITE:<path>=<text>   request permission, then write <text> to <path> (relative to the session cwd) if allowed
  DELETE:<path>         request permission (kind delete), then remove <path> if allowed
  SLOW                  wait up to 20 s for session/cancel before answering (stopReason cancelled when cancelled)
Every turn streams a thought chunk, a plan and a message chunk, then answers with stopReason end_turn.
"""
import json, os, queue, re, sys, threading, time

out_lock = threading.Lock()
inbox = queue.Queue()
replies = {}
cancelled = threading.Event()
cwd = {"v": os.getcwd()}
next_id = [1000]


def send(msg):
    with out_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def update(sid, u):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": u}})


def ask(sid, tool):
    next_id[0] += 1
    rid = next_id[0]
    ev = threading.Event()
    replies[rid] = {"ev": ev}
    send({"jsonrpc": "2.0", "id": rid, "method": "session/request_permission", "params": {
        "sessionId": sid, "toolCall": tool, "options": [
            {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
            {"optionId": "no", "name": "Reject", "kind": "reject_once"}]}})
    ev.wait(60)
    res = replies.pop(rid).get("result") or {}
    o = res.get("outcome") or {}
    return o.get("outcome") == "selected" and o.get("optionId") == "yes"


def prompt(rid, params):
    sid = params["sessionId"]
    text = "".join(b.get("text", "") for b in params.get("prompt") or [])
    cancelled.clear()
    update(sid, {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "Thinking about it."}})
    update(sid, {"sessionUpdate": "plan", "entries": [{"content": "Do the task", "status": "in_progress",
                                                       "priority": "high"}]})
    update(sid, {"sessionUpdate": "usage_update", "used": 1200, "size": 32768})
    if "SLOW" in text:
        for _ in range(200):
            if cancelled.is_set():
                send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "cancelled"}})
                return
            time.sleep(0.1)
    did = []
    for i, (path, body) in enumerate(dict.fromkeys(re.findall(r"WRITE:(\S+?)=(\S+)", text))):
        full = os.path.join(cwd["v"], path)
        tool = {"toolCallId": f"w{i}", "title": f"Write {path}", "kind": "edit", "status": "pending",
                "locations": [{"path": full}], "rawInput": {"file_path": full}}
        update(sid, dict(tool, sessionUpdate="tool_call"))
        if ask(sid, tool):
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(body + "\n")
            did.append(path)
            update(sid, {"sessionUpdate": "tool_call_update", "toolCallId": f"w{i}", "status": "completed",
                         "content": [{"type": "content", "content": {"type": "text", "text": f"wrote {path}"}}]})
        else:
            update(sid, {"sessionUpdate": "tool_call_update", "toolCallId": f"w{i}", "status": "failed"})
    for i, path in enumerate(dict.fromkeys(re.findall(r"DELETE:(\S+)", text))):
        full = os.path.join(cwd["v"], path)
        tool = {"toolCallId": f"d{i}", "title": f"Delete {path}", "kind": "delete", "status": "pending",
                "locations": [{"path": full}]}
        update(sid, dict(tool, sessionUpdate="tool_call"))
        if ask(sid, tool) and os.path.exists(full):
            os.remove(full)
            did.append("-" + path)
    update(sid, {"sessionUpdate": "agent_message_chunk",
                 "content": {"type": "text", "text": "Done: " + (", ".join(did) or "nothing to change")}})
    send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})


def handle(msg):
    m, rid, p = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1, "agentInfo": {"name": "Fake", "version": "1"},
                                                     "agentCapabilities": {"loadSession": True}, "authMethods": []}})
    elif m in ("session/new", "session/load"):
        cwd["v"] = p.get("cwd") or cwd["v"]
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "sessionId": p.get("sessionId") or "fake-1", "configOptions": [
                {"id": "model", "category": "model", "type": "select", "currentValue": "m1",
                 "options": [{"value": "m1"}, {"value": "m2"}]}]}})
        update(p.get("sessionId") or "fake-1", {"sessionUpdate": "available_commands_update",
                                                 "availableCommands": [{"name": "compact", "description": "x"}]})
    elif m == "session/set_config_option":
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
    elif m == "session/prompt":
        flag = os.environ.get("FAKE_EXIT_ONCE")
        if os.environ.get("FAKE_EXIT_ALWAYS") or (flag and not os.path.exists(flag)):
            if flag:
                open(flag, "w", encoding="utf-8").close()
            sys.stderr.write("fake: crashed on purpose\n"); sys.stderr.flush()
            os._exit(3)
        threading.Thread(target=prompt, args=(rid, p), daemon=True).start()
    elif m == "session/cancel":
        cancelled.set()
    elif rid is not None and m is None:
        r = replies.get(rid)
        if r:
            r["result"] = msg.get("result")
            r["ev"].set()
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "nope"}})


for line in sys.stdin:
    try:
        handle(json.loads(line))
    except ValueError:
        pass
