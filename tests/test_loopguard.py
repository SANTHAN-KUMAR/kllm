"""Loop guard: catches exact repetition loops, never normal output. Run: .venv/bin/python -m pytest -q tests"""
import glob, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gateway.relay import LoopGuard, _delta_text


def feed(text, step=7):
    g = LoopGuard()
    for i in range(0, len(text), step):        # stream-like small chunks; bounded loop
        g.add(text[i:i + step])
    return g.tripped


def test_sentence_loop_trips():
    r = feed("Plan:\n" + "I will now check the configuration file again to be sure.\n" * 40)
    assert r and r["period"] == 58


def test_code_loop_trips():
    assert feed("def f():\n" + "    x = compute_value(a, b, c)  # recompute\n" * 45)


def test_short_period_runs_are_not_loops():
    assert feed("title\n" + "-" * 3000) is None
    assert feed("data = [\n" + "    None,\n" * 400 + "]\n") is None


def test_few_repeats_are_not_loops():
    assert feed("x\n" + "assert parse(value) == expected_value  # case\n" * 5 + "y" * 50) is None


def test_real_source_code_is_not_a_loop():
    root = pathlib.Path(__file__).resolve().parent.parent
    files = [p for p in glob.glob(str(root / "gateway" / "*.py")) + glob.glob(str(root / "kernel" / "*.py"))]
    assert files
    for f in files:
        assert feed(open(f, encoding="utf-8").read(), 50) is None, f


def test_delta_text_both_protocols():
    assert _delta_text({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}) == "hi"
    assert _delta_text({"choices": [{"delta": {"content": "a", "tool_calls": [{"function": {"arguments": "{}"}}]}}]}) == "a{}"
