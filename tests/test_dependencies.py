"""Every third-party import in the shipped package must be a declared dependency (or explicitly optional).

0.3.3 crashed on `kllm serve` in clean installs because gateway/vision.py imported numpy, which the developer
machine had globally but pyproject.toml did not declare. This test fails the moment that happens again.
"""
import ast, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# distribution name -> import name, for the ones that differ
IMPORT_NAME = {"pyyaml": "yaml", "pillow": "PIL", "opencv-python": "cv2", "opencv-python-headless": "cv2"}
# imported inside try/except with a fallback, never required at runtime
OPTIONAL = {"cv2"}
# installed by the declared dependencies themselves (fastapi brings these)
TRANSITIVE = {"starlette", "pydantic", "anyio"}


def declared():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.M | re.S).group(1)
    names = re.findall(r'"([A-Za-z0-9_.\-]+)', block)
    return {IMPORT_NAME.get(n.lower(), n.lower().replace("-", "_")) for n in names}


def third_party_imports():
    std = set(sys.stdlib_module_names)
    out = {}
    for pkg in ("gateway", "swarm"):
        for p in (ROOT / pkg).rglob("*.py"):
            if {"benchmarks", "scripts"} & set(p.parts):
                continue                    # dev-only tools, not imported by the installed app
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for n in names:
                    top = n.split(".")[0]
                    if top not in std and top not in ("gateway", "swarm"):
                        out.setdefault(top, []).append(f"{p.relative_to(ROOT)}:{node.lineno}")
    return out


def test_every_import_is_declared():
    have = declared() | OPTIONAL | TRANSITIVE
    missing = {k: v for k, v in third_party_imports().items() if k not in have}
    assert not missing, "undeclared third-party imports (add them to pyproject.toml dependencies): " + repr(missing)


def test_declared_list_parses():
    d = declared()
    assert {"fastapi", "uvicorn", "httpx", "yaml", "kaggle", "numpy", "PIL", "six"} <= d
