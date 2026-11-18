"""llama-server's context-overflow error must reach Claude Code in Anthropic's wording (it only compacts on that)."""
import re
from gateway import relay

LLAMA_MSG = "request (132401 tokens) exceeds the available context size (131072 tokens), try increasing it"
CC_PATTERN = re.compile(r"^prompt is too long: \d+ tokens > \d+ maximum$")   # exact regex found in Claude Code 2.1.316


def test_translation_matches_claude_code_regex():
    out = relay._anthropic_ctx_error('{"error":{"code":400,"message":"%s","type":"exceed_context_size_error"}}' % LLAMA_MSG)
    assert out == "prompt is too long: 132401 tokens > 131072 maximum"
    assert CC_PATTERN.match(out)


def test_other_errors_untouched():
    assert relay._anthropic_ctx_error('{"error":{"message":"slot busy"}}') is None
    assert relay._anthropic_ctx_error("") is None


def test_sse_error_type_is_invalid_request_for_overflow():
    ev = relay._sse_error("v1/messages", "prompt is too long: 1 tokens > 2 maximum").decode()
    assert '"invalid_request_error"' in ev and "overloaded_error" not in ev
    ev2 = relay._sse_error("v1/messages", "kllm: backend stalled").decode()
    assert "overloaded_error" in ev2          # transient errors keep the retryable type


def test_stream_chunk_rewrite():
    chunk = (b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error","message":"%s"}}\n\n'
             % LLAMA_MSG.encode())
    out = relay._CTX_ERR_B.sub(lambda m: b"prompt is too long: %s tokens > %s maximum" % (m.group(1), m.group(2)), chunk)
    assert b"prompt is too long: 132401 tokens > 131072 maximum" in out and b"exceeds the available" not in out


def test_compaction_request_detection():
    marker = ("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n- Do NOT use Read...\n"
              "Your task is to create a detailed summary of the conversation so far.\n"
              "Your entire response must be plain text: an <analysis> block followed by a <summary> block.")
    comp = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": [{"type": "text", "text": "Read x."}, {"type": "text", "text": marker}]}], "tools": [{"name": "Read"}]}
    assert relay._is_compaction_request(comp)
    normal = {"messages": [{"role": "user", "content": "Please read the file and summarize it"}], "tools": [{"name": "Read"}]}
    assert not relay._is_compaction_request(normal)
    assert not relay._is_compaction_request(None) and not relay._is_compaction_request({"messages": []})


# ---- relay-managed compaction: survives a cut stream and an unusable generation -----------------------------------------
import asyncio, json as _json


class _Resp:
    status_code = 200
    def __init__(self, events):
        self.events = events
    async def aiter_raw(self):
        for e in self.events:
            yield e
    async def aclose(self):
        pass


def _ev(t, **kw):
    return (f"event: {t}\ndata: " + _json.dumps({"type": t, **kw}) + "\n\n").encode()


def _text_stream(text, finish=True):
    evs = [_ev("message_start", message={"usage": {"input_tokens": 7}}),
           _ev("content_block_start", index=0, content_block={"type": "text", "text": ""}),
           _ev("content_block_delta", index=0, delta={"type": "text_delta", "text": text})]
    if finish:
        evs += [_ev("content_block_stop", index=0), _ev("message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 9}),
                _ev("message_stop")]
    return evs


def test_compaction_stream_retries_until_a_valid_summary(monkeypatch):
    attempts = []
    script = [_text_stream("<analysis>half", finish=False),                     # 1: tunnel died mid-stream
              _text_stream("<tool_call><function=Read>"),                       # 2: model ignored the instruction
              _text_stream("<analysis>a</analysis><summary>GOOD SUMMARY</summary>")]   # 3: fine

    async def fake_admit_wait(model, session, pin, t_arrive, sset, on_tick=None, request=None):
        return [("t:c", "http://x", "k")], "t:c"
    async def fake_open(cands, request, body, path, session, held=None, admit=True):
        relay.ROUTER.begin("t:c"); attempts.append(1)
        return "t:c", _Resp(script[len(attempts) - 1])
    monkeypatch.setattr(relay, "_admit_wait", fake_admit_wait)
    monkeypatch.setattr(relay, "_open", fake_open)
    monkeypatch.setattr(relay, "_mark_bad", lambda *a, **k: None)
    sset = {"enabled": True, "max_wait_seconds": 5}
    resp = relay._compaction_stream("v1/messages", None, b"{}", {"model": "m"}, "m", None, None, time.time(), sset)

    async def run():
        out = b""
        async for c in resp.body_iterator:
            out += c
        return out
    out = asyncio.run(run()).decode()
    assert len(attempts) == 3
    assert "GOOD SUMMARY" in out and "message_stop" in out and "error" not in out.replace("stop_reason", "")
    assert "<tool_call>" not in out and "half" not in out


import time  # noqa: E402


def test_kv_budget_admission(monkeypatch):
    """2nd concurrent request is admitted only if it fits in the shared KV pool next to the running one."""
    lab = "t:kv"
    monkeypatch.setattr(relay, "cfg", lambda: {"autoscale": {"sessions_per_backend": 2}})
    monkeypatch.setattr(relay.scheduler, "settings", lambda c: {"sessions_per_backend": 2})
    relay.ROUTER.inflight[lab] = 1
    relay._kv_recent.pop(lab, None)
    relay._kv_used[lab] = (time.time(), 131072, 104518 + 8192)          # main conversation running
    assert not relay._slot_free(lab, {"est_tokens": 26674})              # the observed failure: must wait
    relay._kv_used[lab] = (time.time(), 131072, 30000 + 8192)
    assert relay._slot_free(lab, {"est_tokens": 26674})                  # small main context: both fit
    relay.ROUTER.inflight[lab] = 0
    assert relay._slot_free(lab, {"est_tokens": 200000})                 # an idle backend always admits (server decides)


def test_compaction_fast_path_then_flattened_fallback(monkeypatch):
    bodies = []
    tool_call = [_ev("message_start", message={"usage": {}}),
                 _ev("content_block_start", index=0, content_block={"type": "tool_use", "name": "Read", "id": "x", "input": {}}),
                 _ev("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": "{}"})]
    script = [tool_call, _text_stream("<analysis>a</analysis><summary>FLAT OK</summary>")]
    async def fake_admit_wait(*a, **k):
        return [("t:c", "http://x", "k")], "t:c"
    async def fake_open(cands, request, body, path, session, held=None, admit=True):
        relay.ROUTER.begin("t:c"); bodies.append(body)
        return "t:c", _Resp(script[len(bodies) - 1])
    monkeypatch.setattr(relay, "_admit_wait", fake_admit_wait)
    monkeypatch.setattr(relay, "_open", fake_open)
    monkeypatch.setattr(relay, "_mark_bad", lambda *a, **k: None)
    resp = relay._compaction_stream("v1/messages", None, b"FLAT", {"model": "m"}, "m", None, None, time.time(),
                                    {"enabled": True, "max_wait_seconds": 5}, fast_body=b"FAST")
    async def run():
        out = b""
        async for c in resp.body_iterator:
            out += c
        return out
    out = asyncio.run(run()).decode()
    assert bodies == [b"FAST", b"FLAT"]            # cached-prefix first, flattened only after the tool call
    assert "FLAT OK" in out and "message_stop" in out


def test_compaction_fast_body_keeps_prefix():
    payload = {"model": "m", "tools": [{"name": "Read"}], "system": "S",
               "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"},
                            {"role": "user", "content": [{"type": "text", "text": "x"}]}]}
    import copy
    fast = copy.deepcopy(payload); relay._append_to_last_user(fast, relay._COMPACT_BREVITY)
    assert fast["tools"] == payload["tools"] and fast["system"] == payload["system"]
    assert fast["messages"][:2] == payload["messages"][:2]           # history unchanged -> backend prompt cache reused
    assert relay._COMPACT_BREVITY in fast["messages"][-1]["content"][-1]["text"]


def test_flatten_trims_old_tool_output_but_keeps_recent_and_findings():
    big = "X" * 20000
    msgs = []
    for i in range(6):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"finding {i}"},
                                                       {"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {"file_path": f"f{i}"}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": big}]})
    msgs.append({"role": "user", "content": "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. <summary>"})
    out = relay._flatten_for_summary({"messages": msgs})["messages"][0]["content"]
    assert all(f"finding {i}" in out for i in range(6))                  # the model's own findings are never trimmed
    assert out.count("omitted from the summary transcript") >= 3         # old tool output trimmed
    assert big in out                                                    # the most recent tool result is verbatim
    assert len(out) < 6 * len(big) / 2


def test_compaction_detector_ignores_marker_inside_other_text():
    """Regression: a normal turn whose content merely CONTAINS the marker (file text, pasted docs) must not be hijacked."""
    marker = ("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. Your task is to create a detailed summary of the "
              "conversation so far ... <summary>")
    as_file = {"messages": [{"role": "user", "content": "### FILE gateway/relay.py\n" + marker + "\n\nIn one sentence, what is this?"}]}
    assert not relay._is_compaction_request(as_file)
    in_tool_result = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": marker}]}]}
    assert not relay._is_compaction_request(in_tool_result)
