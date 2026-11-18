import json
import os
import pathlib
import pytest
import jsonschema
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OPENCODE_TEMPLATE = ROOT / "clients" / "opencode.jsonc"
# The developer's own opencode config is checked only on request (KLLM_CHECK_USER_CONFIG=1): it is machine state,
# not part of the repo, and a contributor's personal limits must not fail the suite.
USER_OPENCODE_CONFIG = (pathlib.Path(os.path.expanduser("~")) / ".config" / "opencode" / "opencode.jsonc"
                        if os.environ.get("KLLM_CHECK_USER_CONFIG") == "1" else pathlib.Path("/nonexistent/opencode.jsonc"))
MODELS_YAML = ROOT / "models.yaml"
SCHEMA_FIXTURE = ROOT / "tests" / "fixtures" / "opencode_schema.json"


def strip_jsonc_comments(text: str) -> str:
    """Robust stripper for JSONC comments (both // and /* */) that respects strings."""
    out = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        c = text[i]
        if escape:
            out.append(c)
            escape = False
            i += 1
            continue
        if c == "\\":
            out.append(c)
            escape = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            out.append(c)
            i += 1
            continue
        if not in_string:
            if c == "/" and i + 1 < len(text) and text[i + 1] == "/":
                while i < len(text) and text[i] != "\n":
                    i += 1
                continue
            elif c == "/" and i + 1 < len(text) and text[i + 1] == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def load_jsonc(path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    cleaned = strip_jsonc_comments(raw)
    return json.loads(cleaned)


@pytest.fixture(scope="module")
def models_registry():
    with open(MODELS_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


@pytest.fixture(scope="module")
def opencode_schema():
    assert SCHEMA_FIXTURE.exists(), f"Missing schema fixture at {SCHEMA_FIXTURE}"
    with open(SCHEMA_FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_template_jsonc_valid():
    assert OPENCODE_TEMPLATE.exists(), f"Template {OPENCODE_TEMPLATE} not found"
    data = load_jsonc(OPENCODE_TEMPLATE)
    assert "$schema" in data
    assert "provider" in data
    assert "kllm" in data["provider"]
    assert data["provider"]["kllm"]["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert "models" in data["provider"]["kllm"]


def test_user_config_valid():
    if not USER_OPENCODE_CONFIG.exists():
        pytest.skip(f"User opencode config {USER_OPENCODE_CONFIG} does not exist in this environment")
    data = load_jsonc(USER_OPENCODE_CONFIG)
    assert "provider" in data
    assert "kllm" in data["provider"]
    assert data["provider"]["kllm"]["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert "models" in data["provider"]["kllm"]


@pytest.mark.parametrize(
    "config_path,is_user_config",
    [
        (OPENCODE_TEMPLATE, False),
        (USER_OPENCODE_CONFIG, True),
    ],
)
def test_models_coverage_and_attributes(config_path, is_user_config, models_registry):
    if is_user_config and not config_path.exists():
        pytest.skip(f"User opencode config {config_path} does not exist in this environment")

    data = load_jsonc(config_path)
    configured_models = data["provider"]["kllm"]["models"]

    # Dynamic alias routed to active model must exist
    assert "kllm" in configured_models
    assert configured_models["kllm"]["tool_call"] is True
    assert configured_models["kllm"]["limit"]["context"] >= 262144

    for model_id, meta in models_registry.items():
        if meta.get("kind") in ("image", "video"):
            continue  # diffusers image/video backends are not chat/LLM models
        assert model_id in configured_models, f"Model {model_id} from models.yaml is missing in {config_path.name}"
        cfg = configured_models[model_id]

        # Context limits must match models.yaml ctx
        expected_ctx = meta.get("ctx", 32768)
        assert cfg["limit"]["context"] == expected_ctx, (
            f"Context mismatch for {model_id} in {config_path.name}: expected {expected_ctx}, got {cfg['limit']['context']}"
        )
        assert cfg["limit"]["output"] > 0
        assert cfg.get("tool_call") is True

        # Capability checks
        tags = meta.get("tags", [])
        args = meta.get("args", [])
        is_thinking = "thinking" in tags or any(
            "--reasoning-format" in str(a) or "--reasoning-budget" in str(a) for a in args
        )
        if is_thinking:
            assert cfg.get("reasoning") is True, f"Expected reasoning: True for thinking model {model_id} in {config_path.name}"

        if "vision" in tags or "multimodal" in tags:
            assert cfg.get("attachment") is True, f"Expected attachment: True for vision model {model_id} in {config_path.name}"
            assert "modalities" in cfg
            assert "image" in cfg["modalities"].get("input", [])


@pytest.mark.parametrize(
    "config_path,is_user_config",
    [
        (OPENCODE_TEMPLATE, False),
        (USER_OPENCODE_CONFIG, True),
    ],
)
def test_schema_conformance(config_path, is_user_config, opencode_schema):
    if is_user_config and not config_path.exists():
        pytest.skip(f"User opencode config {config_path} does not exist in this environment")

    config = load_jsonc(config_path)
    jsonschema.validate(instance=config, schema=opencode_schema)
