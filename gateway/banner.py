"""Compact ANSI startup banner for `kllm serve`.

Geometry and colours come from docs/assets/banner.svg and docs/assets/favicon.svg:
a #0a0a0a squircle, a #ededed pill, a #8cc8ff rounded outline and the lowercase
`kllm` wordmark. Each cell is a half block (▀): the foreground paints the upper
half, the background the lower half, so one terminal row carries two sub-pixels.
"""
import os
import sys

DARK = (10, 10, 10)     # #0a0a0a
WHITE = (237, 237, 237)  # #ededed
BLUE = (140, 200, 255)  # #8cc8ff
GRAY = (143, 143, 143)  # #8f8f8f (muted subtitle, as in banner.svg)

ROWS = 5
ICON_W = 7
GAP = 2

_LETTERS = {
    "k": ["#..#", "##.#", "##..", "##.#", "#..#"],
    "l": ["#..", "#..", "#..", "#..", "###"],
    "m": ["#..#", "####", "####", "#..#", "#..#"],
}
_WORD = [" ".join(_LETTERS[ch][i] for ch in "kllm") for i in range(ROWS)]
WIDTH = 1 + ICON_W + GAP + len(_WORD[0]) + 1

# (upper half, lower half) for each of the 7 icon columns, top row first.
_ICON = [
    [(DARK, WHITE), (DARK, WHITE), (DARK, DARK), (DARK, DARK), (DARK, BLUE), (DARK, BLUE), (DARK, DARK)],
    [(WHITE, WHITE), (WHITE, WHITE), (DARK, DARK), (BLUE, BLUE), (DARK, DARK), (DARK, DARK), (BLUE, BLUE)],
    [(WHITE, WHITE), (WHITE, WHITE), (DARK, DARK), (BLUE, BLUE), (DARK, DARK), (DARK, DARK), (BLUE, BLUE)],
    [(WHITE, WHITE), (WHITE, WHITE), (DARK, DARK), (BLUE, BLUE), (DARK, DARK), (DARK, DARK), (BLUE, BLUE)],
    [(WHITE, DARK), (WHITE, DARK), (DARK, DARK), (DARK, DARK), (BLUE, DARK), (BLUE, DARK), (DARK, DARK)],
]


def _cell(top, bottom):
    return (f"\033[38;2;{top[0]};{top[1]};{top[2]}m"
            f"\033[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀")


def _art_line(row):
    cells = [(DARK, DARK)]                       # left padding
    cells.extend(_ICON[row])
    cells.extend([(DARK, DARK)] * GAP)
    cells.extend((WHITE, WHITE) if ch == "#" else (DARK, DARK) for ch in _WORD[row])
    cells.append((DARK, DARK))                   # right padding
    assert len(cells) == WIDTH
    return "".join(_cell(top, bottom) for top, bottom in cells) + "\033[0m"


def _version():
    try:
        from importlib.metadata import version
        return version("kllm-gateway")
    except Exception:
        return "dev"


def default_subtitle():
    """Muted one-line subtitle shown under the art."""
    return f"v{_version()} · free GPUs for your coding agent"


def render_banner(subtitle=None):
    """Return the full banner as a string (art rows + subtitle, trailing newline)."""
    if subtitle is None:
        subtitle = default_subtitle()
    lines = [_art_line(r) for r in range(ROWS)]
    lines.append(f"\033[38;2;{GRAY[0]};{GRAY[1]};{GRAY[2]}m  {subtitle}\033[0m")
    return "\n".join(lines) + "\n"


def print_banner(subtitle=None, stream=None):
    """Print the banner only for a human at a terminal.

    Skipped for redirected output (background starts, service logs, smoke tests)
    and when KLLM_BANNER=0 is set.
    """
    stream = stream or sys.stdout
    if not stream.isatty() or os.environ.get("KLLM_BANNER") == "0":
        return
    # Windows consoles default to cp1252, which cannot encode the block glyphs.
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    stream.write(render_banner(subtitle))
    stream.flush()
