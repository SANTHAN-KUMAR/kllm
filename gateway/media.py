"""Model runtimes and kinds: which models are llama.cpp text models and which are diffusers
image/video models, and the small pure helpers the relay and planner share.

models.yaml entries:
  runtime: llama.cpp (default) | diffusers
  kind:    chat (default for llama.cpp) | image | video   (diffusers defaults to image)
Existing entries without these keys are llama.cpp chat models, exactly as before.
"""

RUNTIMES = ("llama.cpp", "diffusers", "swarm", "fastvideo", "comfyui")
KINDS = ("chat", "image", "video")

# relay path (no leading slash) -> the model kind that serves it
MEDIA_PATHS = {"v1/images/generations": "image", "v1/videos/generations": "video"}
MEDIA_SLOTS = 1            # diffusers generates one request at a time


def runtime(m):
    return (m or {}).get("runtime") or "llama.cpp"


def kind(m):
    m = m or {}
    k = m.get("kind")
    if k in KINDS:
        return k
    if runtime(m) == "comfyui":
        return "video"
    return "image" if runtime(m) == "diffusers" else "chat"


def is_media(m):
    return kind(m) in ("image", "video")


def model_id(m):
    """Human label for a registry entry: repo[:quant]. Aliases resolve to their target."""
    m = m or {}
    if m.get("alias_of"):
        return m.get("alias_of")
    q = m.get("quant")
    hf = m.get("hf")
    return f"{hf}:{q}" if (hf and q) else (hf or m.get("key") or "?")


def problems(m):
    """Validation errors for one registry entry (empty list = fine). Aliases inherit
    their target's validity, so they are not flagged for missing hf/quant."""
    m = m or {}
    if m.get("alias_of"):
        return []
    out = []
    if not m.get("hf"):
        out.append("missing 'hf' (HuggingFace repo id)")
    rt = runtime(m)
    if rt not in RUNTIMES:
        out.append(f"unknown runtime '{rt}' (use one of {', '.join(RUNTIMES)})")
    if m.get("kind") not in (None, *KINDS):
        out.append(f"unknown kind '{m.get('kind')}' (use one of {', '.join(KINDS)})")
    if rt == "diffusers" and kind(m) == "chat":
        out.append("runtime diffusers serves image or video models, not chat")
    if rt == "comfyui" and kind(m) != "video":
        out.append("runtime comfyui serves video models")
    if rt in ("llama.cpp", "swarm") and kind(m) != "chat":
        out.append("runtime llama.cpp/swarm serves chat models; use runtime: diffusers for image/video")
    if rt == "llama.cpp" and not (m.get("quant") or m.get("file")):
        out.append("llama.cpp models need 'file' or 'quant'")
    elif rt == "swarm" and not (m.get("quant") or m.get("file") or m.get("dataset")):
        out.append("swarm models need 'file', 'quant', or 'dataset'")
    return out


def resolve_media_model(path, requested, registry, running, cfg=None):
    """Which registry model serves an images/videos request. -> (model_key, None) or (None, error).
    A known model of the right kind is used as asked. Anything else (missing, 'kllm', an OpenAI
    alias like 'dall-e-3') means: config default_image_model / default_video_model, else a
    running model of that kind, else the only one of that kind in the registry."""
    want = MEDIA_PATHS[path]
    of_kind = [k for k, m in registry.items() if kind(m) == want]
    if requested in registry:
        if kind(registry[requested]) != want:
            return None, (f"'{requested}' is a {kind(registry[requested])} model; "
                          f"/{path} needs a {want} model ({', '.join(of_kind) or 'none registered'})")
        return requested, None
    default = (cfg or {}).get(f"default_{want}_model")
    if default in of_kind:
        return default, None
    for k in running:
        if k in of_kind:
            return k, None
    if len(of_kind) == 1:
        return of_kind[0], None
    if not of_kind:
        return None, f"no {want} model in models.yaml; add one with `kllm add <huggingface repo>`"
    return None, (f"which {want} model? set \"model\" to one of: {', '.join(of_kind)}")
