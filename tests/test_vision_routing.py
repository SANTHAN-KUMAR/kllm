"""Vision routing and multimodal payload tests: ensures text/tool history is never dropped."""
import base64
import io
import pathlib
import sys
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gateway import vision


def _create_test_image_b64() -> str:
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_contains_images_pure_text_and_tools():
    payload = {
        "model": "qwen3.8-27b-vision",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Project deep-dive analysis"},
            {
                "role": "assistant",
                "content": "Let me explore the directory.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": "ls -la"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "total 48\ndrwxr-xr-x 5 user ..."},
        ],
    }
    assert vision.contains_images(payload) is False


def test_contains_images_with_image():
    b64 = _create_test_image_b64()
    payload = {
        "model": "qwen3.8-27b-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this image?"},
                    {"type": "image", "data": b64, "mimeType": "image/png"},
                ],
            }
        ],
    }
    assert vision.contains_images(payload) is True


def test_build_vision_chat_payload_preserves_history_and_tools():
    b64 = _create_test_image_b64()
    payload = {
        "model": "qwen3.8-27b-vision",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this screenshot:"},
                    {"type": "image", "data": b64, "mimeType": "image/png"},
                ],
            },
            {
                "role": "assistant",
                "content": "I see the red box. Let me run tests.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_test", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Tests passed: 5/5"},
        ],
    }

    result = vision.build_vision_chat_payload(payload, "qwen3.8-27b-vision")

    # History length and roles must be preserved
    assert len(result["messages"]) == 4
    assert result["messages"][0]["role"] == "system"
    assert "Real-time high-precision computer vision" in result["messages"][0]["content"]

    # Tool calls must not be stripped
    assert result["messages"][2]["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "run_test", "arguments": "{}"}}
    ]
    assert result["messages"][3]["role"] == "tool"
    assert result["messages"][3]["content"] == "Tests passed: 5/5"

    # User message image should be transformed in-place to text report
    user_content = result["messages"][1]["content"]
    assert len(user_content) == 2
    assert user_content[0] == {"type": "text", "text": "Check this screenshot:"}
    assert user_content[1]["type"] == "text"
    assert "[Visual Image Analysis Report for Image 1" in user_content[1]["text"]


def test_qwen38_abliterated_vision_256k_routing():
    model_name = "qwen3.8-27b-vision-abliterated-256k"
    alias_name = "qwen3.8-27b-abliterated-vision-256k"

    assert vision.is_vision_model(model_name) is True
    assert vision.is_vision_model(alias_name) is True
    assert vision.is_vision_model(f"kllm/{model_name}") is True

    b64 = _create_test_image_b64()
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this architecture diagram:"},
                    {"type": "image", "data": b64, "mimeType": "image/png"},
                ],
            }
        ],
    }

    assert vision.contains_images(payload) is True
    transformed = vision.build_vision_chat_payload(payload, model_name)
    assert transformed["model"] == model_name
    assert len(transformed["messages"]) == 2
    assert "Real-time high-precision computer vision" in transformed["messages"][0]["content"]
    assert "[Visual Image Analysis Report for Image 1" in transformed["messages"][1]["content"][1]["text"]

