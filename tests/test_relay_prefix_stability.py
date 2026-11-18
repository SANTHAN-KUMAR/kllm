"""The prompt the backend renders must be append-only across turns, or the prompt cache misses and the backend
re-prefills the whole conversation (measured live: 2% of a 103K prompt reusable -> ~10 min rebuild)."""
import json
import os

import jinja2

from gateway import relay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _render(messages, **kw):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])

    def _raise(m):
        raise RuntimeError(m)
    env.globals["raise_exception"] = _raise
    src = open(os.path.join(ROOT, "templates", "qwen3.6.jinja"), encoding="utf-8").read()
    return env.from_string(src).render(messages=messages, tools=[], add_generation_prompt=True, **kw)


def _loop(n=6):
    msgs = [{"role": "system", "content": "SYS " * 50}, {"role": "user", "content": "TASK"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": "", "reasoning_content": f"think {i} " * 30,
                  "tool_calls": [{"type": "function", "id": f"c{i}", "function": {"name": "Read", "arguments": {"p": i}}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i} " * 40}]
    return msgs


def _common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def test_template_default_breaks_prefix_on_new_user_message():
    before = _render(_loop())
    after = _render(_loop() + [{"role": "user", "content": "follow-up"}])
    assert _common(before, after) < 0.1 * len(before)          # the bug this guards against


def test_preserve_thinking_keeps_prefix_append_only():
    before = _render(_loop(), preserve_thinking=True)
    after = _render(_loop() + [{"role": "user", "content": "follow-up"}], preserve_thinking=True)
    # everything except the trailing generation prompt is reused
    assert _common(before, after) >= len(before) - len("<|im_start|>assistant\n<think>\n")


def test_relay_sets_preserve_thinking_without_overriding_client():
    p = {"model": "m", "messages": []}
    assert relay._stable_history(p) is True
    assert p["chat_template_kwargs"] == {"preserve_thinking": True}
    q = {"chat_template_kwargs": {"preserve_thinking": False, "x": 1}}
    assert relay._stable_history(q) is False and q["chat_template_kwargs"] == {"preserve_thinking": False, "x": 1}
    r = {"chat_template_kwargs": {"enable_thinking": False}}
    assert relay._stable_history(r) is True
    assert r["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": True}
    json.dumps(p)


# ---- same-session serialisation of large requests (Claude Code background forks) ----
import asyncio


class _Req:
    def __init__(self):
        self.gone = False

    async def is_disconnected(self):
        return self.gone


def _big(uid, n_tools=0):
    return json.dumps({"model": "m", "metadata": {"user_id": uid}, "tools": [{}] * n_tools,
                       "messages": [{"role": "user", "content": "x" * 80000}]}).encode()


def test_gate_only_for_large_message_posts():
    assert relay._session_gate_key("v1/messages", "POST", _big("s1")) is not None
    assert relay._session_gate_key("v1/messages", "GET", _big("s1")) is None
    assert relay._session_gate_key("v1/models", "POST", _big("s1")) is None
    small = json.dumps({"metadata": {"user_id": "s1"}, "messages": [{"role": "user", "content": "hi"}]}).encode()
    assert relay._session_gate_key("v1/messages", "POST", small) is None


def test_same_session_waits_other_session_runs():
    relay._session_gates.clear()
    k1 = relay._session_gate_key("v1/messages", "POST", _big("s1"))
    k2 = relay._session_gate_key("v1/messages", "POST", _big("s2"))

    async def run():
        assert await relay._acquire_session_gate(k1, _Req()) is True
        assert await asyncio.wait_for(relay._acquire_session_gate(k2, _Req()), 1) is True   # other session: no wait
        waiter = asyncio.ensure_future(relay._acquire_session_gate(k1, _Req()))
        await asyncio.sleep(0.8)
        assert not waiter.done()                                  # fork waits for its own session's request
        relay._release_session_gate(k1)
        assert await asyncio.wait_for(waiter, 2) is True
    asyncio.run(run())


def test_gate_never_blocks_forever(monkeypatch):
    relay._session_gates.clear()
    monkeypatch.setattr(relay, "SESSION_GATE_MAX_WAIT", 0.6)
    k = relay._session_gate_key("v1/messages", "POST", _big("s3"))

    async def run():
        assert await relay._acquire_session_gate(k, _Req()) is True
        assert await relay._acquire_session_gate(k, _Req()) is False          # timed out -> proceeds in parallel
        r = _Req(); r.gone = True
        assert await relay._acquire_session_gate(k, r) is None                # client left while queued
    asyncio.run(run())


# ---- reasoning effort -> thinking budget (Claude Code sends thinking.type=adaptive + output_config.effort) ----
def _cc(effort, max_tokens=32768, thinking=None):
    return {"model": "m", "max_tokens": max_tokens, "thinking": thinking or {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": effort}, "messages": [{"role": "user", "content": "x"}]}


def test_effort_levels_become_distinct_budgets():
    got = {}
    for e in ("low", "medium", "high", "xhigh", "max"):
        p = _cc(e)
        assert relay._apply_effort("v1/messages", p) is True
        assert p["thinking"]["type"] == "enabled"
        got[e] = p["thinking"]["budget_tokens"]
    assert got["low"] < got["medium"] < got["high"] < got["xhigh"] <= got["max"]
    assert got["max"] <= 32768 - relay.THINKING_ANSWER_RESERVE        # the answer always keeps room


def test_effort_budget_clamped_by_small_max_tokens():
    p = _cc("xhigh", max_tokens=16384)
    relay._apply_effort("v1/messages", p)
    assert p["thinking"]["budget_tokens"] == 16384 - relay.THINKING_ANSWER_RESERVE


def test_explicit_budget_and_disabled_are_respected():
    p = _cc("high", thinking={"type": "enabled", "budget_tokens": 3000})
    assert relay._apply_effort("v1/messages", p) is False and p["thinking"]["budget_tokens"] == 3000
    p = _cc("high", thinking={"type": "disabled"})
    assert relay._apply_effort("v1/messages", p) is True
    assert p["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_reasoning_effort_maps_to_budget():
    p = {"messages": [], "max_tokens": 32768, "reasoning_effort": "high"}
    assert relay._apply_effort("v1/chat/completions", p) is True
    assert p["reasoning_budget_tokens"] == relay.EFFORT_BUDGET["high"]
    q = {"messages": []}
    assert relay._apply_effort("v1/chat/completions", q) is False and "reasoning_budget_tokens" not in q


def test_effort_budget_small_limits_keep_room_for_answer():
    assert relay._effort_budget("high", 512) == 0                  # tiny request: no thinking at all
    assert relay._effort_budget("high", 2048) == 1024              # thinking gets at most half
    assert relay._effort_budget("max", 6000) == 3000
    assert relay._effort_budget("high", None) == relay.EFFORT_BUDGET["high"]
    p = _cc("high", max_tokens=512)
    assert relay._apply_effort("v1/messages", p) is True and p["chat_template_kwargs"]["enable_thinking"] is False
