#!/usr/bin/env python3
"""Lint the kllm UI against the brand kit (docs/redesign/BRAND_KIT.md).

Usage:  python scripts/ui_lint.py gateway/ui
Exit code 0 when clean, 1 when any problem is found.

Checks every .html, .css and .js file under the given paths, except tokens.css,
which is the one place literal colours are allowed.
"""
import pathlib
import re
import sys

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
RULES = [
    ("em or en dash in text", re.compile("[\u2013\u2014]"), None),
    ("emoji", EMOJI, None),
    ("gradient text", re.compile(r"background-clip:\s*text|-webkit-text-fill-color:\s*transparent"), None),
    ("backdrop blur", re.compile(r"backdrop-filter"), None),
    ("bounce or elastic easing", re.compile(r"cubic-bezier\([^)]*-\d|cubic-bezier\([^)]*,\s*1\.[1-9]|bounce|elastic", re.I), "css-like"),
    ("transition on a layout property", re.compile(r"transition[^;{}]*\b(width|height|margin|padding|top|left)\b"), None),
    ("CDN load (kllm runs offline)", re.compile(r"https?://(fonts\.googleapis|fonts\.gstatic|cdn\.|unpkg\.|cdnjs\.)"), None),
    ("literal colour outside tokens.css", re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\(\s*\d"), "css-like"),
    ("generic button label", re.compile(r">\s*(Submit|OK|Click here)\s*<"), None),
    ("ALL CAPS heading", re.compile(r"<h[1-6][^>]*>[^<a-z]{6,}</h[1-6]>"), None),
]
# Live-data contract (FRONTEND_GUIDE.md, "Is everything live?"): nothing in the shipped UI is mocked.
MOCK_RULES = [
    ("random data (Math.random)", re.compile(r"Math\.random\s*\(")),
    ("hard-coded sample series", re.compile(r"\b(SAMPLE|MOCK|FAKE|DUMMY|EXAMPLE)_?[A-Z_]*\s*=\s*[\[{]")),
    ("deterministic fake generator", re.compile(r"seed\s*\*\s*9301|function\s+rnd\s*\(")),
]
# Example values that exist only in docs/redesign/prototype.html. They must never ship.
PROTOTYPE_ONLY = re.compile(
    r"relay_pump_trace|harbour_dawn"
    r"|Fix relay silence on long prompts|Docs for the Connect page|Scope false positives in spawner"
    r"|933\.8|7h 12m|73\.6 tokens|412 requests|182K Kaggle|6\.1K cloud|kg7 free|preview runs 60"
)
BANNED_WORDS = re.compile(r"\b(seamless(ly)?|robust|leverag(e|ing)|utili[sz]e|cutting-edge|game-changer|supercharge|unlock|empower|delve)\b", re.I)


def css_like(path, line):
    """Literal colours count only in CSS and in style attributes, never in tokens.css."""
    if path.name == "tokens.css":
        return False
    if path.suffix == ".css":
        return True
    return "style=" in line or "fill=" in line or "stroke=" in line


def lint(paths):
    problems = 0
    files = []
    for p in paths:
        p = pathlib.Path(p)
        files += [p] if p.is_file() else [f for f in p.rglob("*") if f.suffix in (".html", ".css", ".js")]
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for name, rx, mode in RULES:
                if mode == "css-like" and not css_like(f, line):
                    continue
                if rx.search(line):
                    problems += 1
                    print(f"{f}:{n}: {name}: {line.strip()[:110]}")
            if f.name != "prototype.html":
                for name, rx in MOCK_RULES:
                    if rx.search(line):
                        problems += 1
                        print(f"{f}:{n}: mock data: {name}: {line.strip()[:110]}")
                if PROTOTYPE_ONLY.search(line):
                    problems += 1
                    print(f"{f}:{n}: mock data: prototype example value: {line.strip()[:110]}")
            if f.suffix == ".html" and BANNED_WORDS.search(re.sub(r"<[^>]+>", " ", line)):
                problems += 1
                print(f"{f}:{n}: banned word: {line.strip()[:110]}")
    print(f"{problems} problems")
    return problems


if __name__ == "__main__":
    sys.exit(1 if lint(sys.argv[1:] or ["gateway/ui"]) else 0)
