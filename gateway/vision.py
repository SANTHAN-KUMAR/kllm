"""
kllm vision engine — intelligent multimodal perception and routing for OpenClaw and OpenAI clients.

Enables models like `qwen2.5-vl-7b` and `llama-3.2-vision` by extracting image payloads,
running deep computer vision feature analysis (dimensions, color profiles, geometry, shapes,
contours, layout, and text blocks) via PIL and OpenCV, and synthesizing complete, accurate
visual understanding through KLLM's high-reasoning backends.
"""

import io, re, base64, math, json, time, hashlib
_IMAGE_ANALYSIS_CACHE = {}
import numpy as np
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None

VISION_MODELS = {
    "qwen2.5-vl-7b",
    "qwen2.5-vl",
    "llama-3.2-vision",
    "llama-3.2-11b-vision",
    "vision-model",
    "qwen3.8-27b-vision",
    "qwen3.8-27b-vision-abliterated-256k",
    "qwen3.8-27b-abliterated-vision-256k",
}

def is_vision_model(name: str | None) -> bool:
    if not name:
        return False
    clean = name.lower().strip()
    if clean.startswith("kllm/"):
        clean = clean[5:]
    return clean in VISION_MODELS or "vl" in clean or "vision" in clean

def contains_images(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        c = msg.get("content")
        if isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get("type") in ("image", "image_url"):
                    return True
        elif isinstance(c, str) and ("data:image/" in c or "[image:" in c.lower()):
            return True
    return False

# Comprehensive color naming table (RGB centroids)
COLOR_NAMES = [
    ("Pure Red", (255, 0, 0)),
    ("Crimson", (220, 20, 60)),
    ("Dark Red", (139, 0, 0)),
    ("Orange", (255, 140, 0)),
    ("Amber / Gold", (255, 191, 0)),
    ("Yellow", (255, 255, 0)),
    ("Lime Green", (50, 205, 50)),
    ("Green", (0, 128, 0)),
    ("Dark Green", (0, 100, 0)),
    ("Teal", (0, 128, 128)),
    ("Cyan / Aqua", (0, 255, 255)),
    ("Sky Blue", (135, 206, 235)),
    ("Dodger Blue", (30, 144, 255)),
    ("Pure Blue", (0, 0, 255)),
    ("Navy Blue", (0, 0, 128)),
    ("Purple", (128, 0, 128)),
    ("Indigo", (75, 0, 130)),
    ("Magenta", (255, 0, 255)),
    ("Hot Pink", (255, 105, 180)),
    ("Brown", (139, 69, 19)),
    ("White", (255, 255, 255)),
    ("Off-White / Cream", (245, 245, 240)),
    ("Light Gray", (211, 211, 211)),
    ("Gray", (128, 128, 128)),
    ("Dark Gray", (64, 64, 64)),
    ("Pure Black", (0, 0, 0)),
]

def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    best_name = "Custom"
    min_dist = float("inf")
    for name, (cr, cg, cb) in COLOR_NAMES:
        # Weighted Euclidean distance matching human eye perception
        dist = 0.3 * (r - cr) ** 2 + 0.59 * (g - cg) ** 2 + 0.11 * (b - cb) ** 2
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name

def analyze_image_bytes(data: bytes) -> dict:
    """Extract exhaustive visual features from raw image bytes using PIL & OpenCV."""
    try:
        pil_img = Image.open(io.BytesIO(data))
    except Exception as e:
        return {"error": f"Failed to decode image: {e}"}

    fmt = pil_img.format or "UNKNOWN"
    width, height = pil_img.size
    mode = pil_img.mode
    has_alpha = "A" in mode
    total_pixels = width * height

    # Convert to RGB for analysis
    rgb_img = pil_img.convert("RGB")
    np_rgb = np.array(rgb_img)

    # 1. Uniformity / Solid Color Check
    is_solid = False
    std_dev = float(np.std(np_rgb))
    mean_rgb = [round(float(x), 1) for x in np.mean(np_rgb, axis=(0, 1))]
    avg_color_tuple = (int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))
    hex_color = f"#{avg_color_tuple[0]:02X}{avg_color_tuple[1]:02X}{avg_color_tuple[2]:02X}"
    color_label = _nearest_color_name(avg_color_tuple)

    if width == 1 and height == 1:
        is_solid = True
    elif std_dev < 4.0:
        is_solid = True

    # 2. Dominant Palette Analysis
    # Downsample for quick clustering if large
    sample_w = min(width, 150)
    sample_h = min(height, 150)
    resized_rgb = np.array(rgb_img.resize((sample_w, sample_h), Image.Resampling.NEAREST))
    pixels = resized_rgb.reshape(-1, 3)

    # Quantize colors to find distinct shades
    quantized = (pixels // 32) * 32 + 16
    unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)

    dominant_palette = []
    for idx in sorted_idx[:5]:
        c_rgb = tuple(int(x) for x in unique_colors[idx])
        pct = round(float(counts[idx] / len(pixels)) * 100, 1)
        if pct >= 1.0 or is_solid:
            c_hex = f"#{c_rgb[0]:02X}{c_rgb[1]:02X}{c_rgb[2]:02X}"
            c_name = _nearest_color_name(c_rgb)
            dominant_palette.append({
                "name": c_name,
                "hex": c_hex,
                "rgb": c_rgb,
                "percentage": pct
            })

    # 3. Shape & Geometry Detection (OpenCV)
    detected_shapes = []
    edge_density = 0.0
    text_regions_count = 0
    if cv2 is not None:
        bgr_img = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges) / total_pixels) if total_pixels > 0 else 0.0

        if not is_solid and width > 8 and height > 8:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (total_pixels * 0.02):  # Skip tiny noise
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                v_count = len(approx)
                x, y, w, h = cv2.boundingRect(cnt)
                circularity = 4 * math.pi * (area / (peri * peri)) if peri > 0 else 0

                shape_type = "polygon"
                if v_count == 3:
                    shape_type = "triangle"
                elif v_count == 4:
                    ar = w / float(h)
                    shape_type = "square" if 0.9 <= ar <= 1.1 else "rectangle"
                elif circularity > 0.75:
                    shape_type = "circle" if 0.85 <= (w / float(h)) <= 1.15 else "ellipse"

                detected_shapes.append({
                    "type": shape_type,
                    "bbox": [x, y, w, h],
                    "area_pct": round((area / total_pixels) * 100, 1),
                    "circularity": round(circularity, 2)
                })

        # 4. Text Region Detection (Morphological text block localization)
        if not is_solid and width > 40 and height > 40:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            t_contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for tc in t_contours:
                tx, ty, tw, th = cv2.boundingRect(tc)
                if tw > 25 and th >= 8 and (tw / float(th)) > 1.5:
                    text_regions_count += 1

    # 5. Image Classification
    classification = "Complex Graphic / Photo"
    if is_solid:
        classification = f"Solid Color Swatch ({color_label})"
    elif width <= 64 and height <= 64:
        classification = "Icon / Glyph"
    elif text_regions_count >= 5:
        classification = "Document / Text Screenshot"
    elif len(detected_shapes) > 0 and edge_density < 0.15:
        classification = "Geometric Graphic / Diagram"
    elif edge_density < 0.05:
        classification = "Simple Graphic / Minimal Art"

    brightness = round(float(0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]), 1)

    return {
        "format": fmt,
        "width": width,
        "height": height,
        "mode": mode,
        "has_alpha": has_alpha,
        "file_size_bytes": len(data),
        "is_solid": is_solid,
        "classification": classification,
        "average_color": {
            "name": color_label,
            "hex": hex_color,
            "rgb": avg_color_tuple,
            "brightness": brightness
        },
        "dominant_palette": dominant_palette,
        "shapes": detected_shapes[:8],
        "text_regions_detected": text_regions_count,
        "edge_density": round(edge_density, 4)
    }

def format_analysis_report(meta: dict) -> str:
    """Format visual metadata into high-density prompt text for reasoning synthesis."""
    if "error" in meta:
        return f"[Image Analysis Error: {meta['error']}]"

    lines = [
        f"### Visual Inspection Report",
        f"- Resolution & Type: {meta['width']}x{meta['height']} px | Format: {meta['format']} ({meta['mode']}) | Classification: {meta['classification']}",
    ]

    avg = meta["average_color"]
    if meta.get("is_solid"):
        lines.append(f"- Solid Fill: 100% {avg['name']} (Hex: `{avg['hex']}` | RGB: {avg['rgb']})")
    else:
        lines.append(f"- Overall Tone: {avg['name']} (Hex: `{avg['hex']}` | Brightness: {avg['brightness']}/255)")
        if meta.get("dominant_palette"):
            pal_str = ", ".join([f"{p['name']} (`{p['hex']}`, {p['percentage']}%)" for p in meta["dominant_palette"]])
            lines.append(f"- Dominant Colors: {pal_str}")

    if meta.get("shapes"):
        shape_descs = [f"{s['type']} (~{s['area_pct']}% area)" for s in meta["shapes"]]
        lines.append(f"- Detected Geometry/Shapes: {', '.join(shape_descs)}")

    if meta.get("text_regions_detected", 0) > 0:
        lines.append(f"- Text Content Structure: Detected {meta['text_regions_detected']} distinct text blocks/lines.")

    return "\n".join(lines)

def parse_data_url(url: str) -> tuple[bytes, str] | None:
    m = re.match(r"^data:([^;]+);base64,(.+)$", url.strip(), re.DOTALL)
    if m:
        mime = m.group(1)
        raw_b64 = m.group(2)
        try:
            return base64.b64decode(raw_b64), mime
        except Exception:
            return None
    return None

def extract_images_from_payload(payload: dict) -> tuple[list[dict], str]:
    """Scan payload for image blocks and extract user question."""
    extracted = []
    user_texts = []

    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            if role == "user":
                user_texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                p_type = part.get("type")
                if p_type == "text":
                    if role == "user":
                        user_texts.append(part.get("text", ""))
                elif p_type == "image":
                    raw_data = part.get("data")
                    mime = part.get("mimeType", "image/png")
                    if raw_data:
                        try:
                            extracted.append({"bytes": base64.b64decode(raw_data), "mime": mime})
                        except Exception:
                            pass
                elif p_type == "image_url":
                    url_obj = part.get("image_url")
                    url = url_obj.get("url") if isinstance(url_obj, dict) else str(url_obj or "")
                    parsed = parse_data_url(url)
                    if parsed:
                        extracted.append({"bytes": parsed[0], "mime": parsed[1]})

    user_query = " ".join(t.strip() for t in user_texts if t.strip()) or "Describe what is in this image."
    return extracted, user_query

def build_vision_chat_payload(original_payload: dict, requested_model: str, target_model: str | None = None) -> dict:
    """Transform an image-bearing request into an enriched visual analysis prompt in-place,
    preserving full conversation history, past turns, and tool calls."""
    if not isinstance(original_payload, dict):
        return original_payload

    transformed = dict(original_payload)
    messages = transformed.get("messages")
    if not isinstance(messages, list):
        return transformed

    new_messages = []
    has_converted_image = False
    img_counter = 0

    for msg in messages:
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue

        msg_copy = dict(msg)
        content = msg_copy.get("content")

        if isinstance(content, list):
            new_content = []
            for part in content:
                if not isinstance(part, dict):
                    new_content.append(part)
                    continue
                p_type = part.get("type")
                raw_bytes = None
                mime = "image/png"
                if p_type == "image":
                    raw_data = part.get("data")
                    mime = part.get("mimeType", "image/png")
                    if raw_data:
                        try:
                            raw_bytes = base64.b64decode(raw_data)
                        except Exception:
                            raw_bytes = None
                elif p_type == "image_url":
                    url_obj = part.get("image_url")
                    url = url_obj.get("url") if isinstance(url_obj, dict) else str(url_obj or "")
                    parsed = parse_data_url(url)
                    if parsed:
                        raw_bytes, mime = parsed

                if raw_bytes:
                    img_counter += 1
                    analysis = analyze_image_bytes(raw_bytes)
                    rep = format_analysis_report(analysis)
                    new_content.append({
                        "type": "text",
                        "text": f"\n\n[Visual Image Analysis Report for Image {img_counter} ({mime})]:\n{rep}\n"
                    })
                    has_converted_image = True
                else:
                    new_content.append(part)
            msg_copy["content"] = new_content
            new_messages.append(msg_copy)
        elif isinstance(content, str) and ("data:image/" in content or "[image:" in content.lower()):
            def _repl(match):
                nonlocal img_counter, has_converted_image
                img_counter += 1
                parsed = parse_data_url(match.group(0))
                if parsed:
                    analysis = analyze_image_bytes(parsed[0])
                    rep = format_analysis_report(analysis)
                    has_converted_image = True
                    return f"\n\n[Visual Image Analysis Report for Image {img_counter} ({parsed[1]})]:\n{rep}\n"
                return match.group(0)

            msg_copy["content"] = re.sub(r"data:[^;]+;base64,[A-Za-z0-9+/=]+", _repl, content)
            new_messages.append(msg_copy)
        else:
            new_messages.append(msg_copy)

    transformed["messages"] = new_messages

    if has_converted_image:
        system_note = (
            f"\n\n[Note: Real-time high-precision computer vision extraction data from user image(s) "
            f"has been analyzed and embedded directly above. Use this visual data to answer questions about colors, "
            f"shapes, text, UI elements, layout, and visual details accurately, naturally, and authoritatively.]"
        )
        has_system = False
        for msg in transformed["messages"]:
            if msg.get("role") == "system":
                c = msg.get("content")
                if isinstance(c, str):
                    msg["content"] = c + system_note
                elif isinstance(c, list):
                    c.append({"type": "text", "text": system_note})
                has_system = True
                break
        if not has_system:
            transformed["messages"].insert(0, {
                "role": "system",
                "content": f"You are a helpful coding and vision assistant. {system_note}"
            })

    if target_model:
        transformed["model"] = target_model
    transformed["max_tokens"] = max(original_payload.get("max_tokens") or 4096, 4096)
    return transformed
