"""Inspect a HuggingFace repo and turn it into a models.yaml entry: `kllm add <repo or URL>`.

  GGUF repo        -> runtime llama.cpp: pick the best quant that fits 2xT4 (30 GB) with room for context
  diffusers repo   -> runtime diffusers, kind image or video (model_index.json; fp16 footprint from the
                      safetensors sizes)
  anything else    -> a clear error saying what kllm can run

Only read-only GETs against huggingface.co (public repos need no token; `hf_token` in config.yaml or
$HF_TOKEN is sent for gated ones). The network primitive `_get` is the single seam tests replace.
"""
import json, math, os, re, struct, time, urllib.parse, urllib.request, urllib.error
import yaml

HF = "https://huggingface.co"
UA = {"User-Agent": "kllm-inspect/1"}

WEIGHT_FIT_GB = 22.0      # GGUF weights that leave room for KV cache + compute buffers on 30 GB
VRAM_TOTAL_GB = 27.0      # weights + KV + buffers budget the context estimate works to (2x15 GB minus headroom)
BUFFER_GB = 1.5
SMALL_Q5_GB = 12.0        # a model this small can afford Q5_K_M
DIFFUSERS_RAM_GB = 28.0   # offloading keeps the whole pipeline in the kernel's ~31 GB RAM
DIFFUSERS_GPU_GB = 11.5   # whole pipeline resident on one T4
CTX_STEPS = [4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144]
DEFAULT_CTX = 32768


class InspectError(Exception):
    """Something the user can act on (bad repo, unsupported type, network)."""


# ---- network ------------------------------------------------------------------------------------
def _get(url, token=None, headers=None, timeout=20):
    """-> (body bytes, response headers dict). Raises InspectError with a useful message."""
    h = dict(UA, **(headers or {}))
    if token:
        h["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
            return r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise InspectError("HuggingFace refused access (gated or private repo). Accept the license on the "
                               "model page and set hf_token in config.yaml (a read token).") from None
        if e.code == 404:
            raise InspectError("repo not found on HuggingFace (check the name; private repos need hf_token)") from None
        raise InspectError(f"HuggingFace returned HTTP {e.code}") from None
    except (urllib.error.URLError, OSError) as e:
        raise InspectError(f"cannot reach huggingface.co: {getattr(e, 'reason', e)}") from None


def _json(url, token=None):
    body, hdrs = _get(url, token)
    try:
        return json.loads(body), hdrs
    except ValueError:
        raise InspectError(f"unexpected response from {url}") from None


# ---- source parsing -----------------------------------------------------------------------------
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def parse_source(src):
    """'org/name', 'org/name:Q4_K_M', or any huggingface.co / hf.co URL (incl. /blob|/resolve/<rev>/<file>)
    -> (repo id, wanted file or quant or None)."""
    s = (src or "").strip()
    if not s:
        raise InspectError("give a HuggingFace repo id (org/name) or URL")
    wanted = None
    m = re.match(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(.+)$|^hf://(.+)$", s)
    if m:
        parts = [p for p in re.split(r"[?#]", m.group(1) or m.group(2))[0].split("/") if p]
        if parts and parts[0] in ("models",):
            parts = parts[1:]
        if len(parts) < 2:
            raise InspectError(f"cannot find org/name in {src!r}")
        repo = "/".join(parts[:2])
        if len(parts) > 4 and parts[2] in ("blob", "resolve"):
            wanted = "/".join(parts[4:])
        elif ":" in parts[1]:
            repo, wanted = repo.split(":", 1)
    else:
        repo = s
        if ":" in s:
            repo, wanted = s.split(":", 1)
    repo = repo.strip("/")
    if not _REPO.match(repo):
        raise InspectError(f"{src!r} is not a HuggingFace repo id (expected org/name) or URL")
    return repo, (wanted or None)


def fetch_tree(repo, token=None):
    """All files of the repo's main branch: [{path, size}]."""
    files, url = [], f"{HF}/api/models/{repo}/tree/main?recursive=true&expand=false"
    for _ in range(20):                                   # paginated via Link: <...>; rel="next"
        body, hdrs = _get(url, token)
        try:
            page = json.loads(body)
        except ValueError:
            raise InspectError("unexpected file listing from HuggingFace") from None
        for e in page:
            if e.get("type") == "file":
                files.append({"path": e["path"], "size": (e.get("lfs") or {}).get("size") or e.get("size") or 0})
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', hdrs.get("link", ""))
        if not nxt:
            break
        url = nxt.group(1)
    return files


# ---- GGUF ---------------------------------------------------------------------------------------
_SPLIT = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
_QUANT = re.compile(r"(?<![A-Za-z0-9])((?:i1-)?(?:IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|MXFP4(?:_MOE)?|BF16|F16|F32))(?![A-Za-z0-9])",
                    re.I)
# quality/size preference, best first. i1 (imatrix) beats plain at the same quant.
_PREF = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_K_L", "MXFP4_MOE", "MXFP4", "Q4_1", "Q4_0", "IQ4_NL", "Q5_K_S",
         "Q5_K_M", "Q3_K_L", "IQ3_M", "Q3_K_M", "IQ3_S", "Q3_K_S", "IQ3_XS", "IQ3_XXS", "Q2_K_L", "Q2_K",
         "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "Q6_K", "Q8_0"]


def norm_quant(q):
    q = (q or "").strip()
    imat = q.lower().startswith("i1-")
    base = q[3:] if imat else q
    return ("i1-" if imat else "") + base.upper()


def quant_of(path):
    base = os.path.basename(path)
    base = _SPLIT.sub(lambda m: m.group("stem") + ".gguf", base)
    found = _QUANT.findall(base[:-5] if base.lower().endswith(".gguf") else base)
    return norm_quant(found[-1]) if found else None


def group_ggufs(files):
    """Candidate GGUFs, split files (…-00001-of-00003.gguf) grouped. Returns (candidates, skipped notes)."""
    groups, notes = {}, []
    for f in files:
        p = f["path"]
        low = p.lower()
        if ".gguf.part" in low or re.search(r"\.gguf\.\d+of\d+$", low) or re.search(r"part\d+of\d+$", low):
            notes.append(f"{os.path.basename(p)}: raw-split file (needs joining by hand); skipped")
            continue
        if not low.endswith(".gguf") or "mmproj" in low or "imatrix" in low:
            continue
        m = _SPLIT.match(os.path.basename(p))
        key = (os.path.dirname(p), m.group("stem")) if m else (p, "")
        g = groups.setdefault(key, {"size": 0, "parts": 0, "first": None, "n": 1})
        g["size"] += f["size"]; g["parts"] += 1
        if m:
            g["n"] = int(m.group("n"))
            if int(m.group("i")) == 1:
                g["first"] = p
        else:
            g["first"] = p
    out = []
    for key, g in groups.items():
        if not g["first"] or g["parts"] < g["n"]:
            notes.append(f"{key[1] or key[0]}: incomplete split ({g['parts']}/{g['n']} parts); skipped")
            continue
        out.append({"file": g["first"], "size_gb": round(g["size"] / 1e9, 2), "parts": g["parts"],
                    "quant": quant_of(g["first"])})
    out.sort(key=lambda c: c["size_gb"])
    return out, notes


def _rank(c):
    q = c["quant"] or ""
    base = q[3:] if q.startswith("i1-") else q
    try:
        idx = _PREF.index(base)
    except ValueError:
        idx = len(_PREF)
    return (idx, not q.startswith("i1-"), c["size_gb"])


def pick_gguf(cands, wanted=None, repo_imatrix=False):
    """-> (chosen, warnings). `wanted` = exact path, basename, or a quant name (Q5_K_M, i1-Q4_K_S, substring)."""
    warn = []
    if not cands:
        return None, ["no usable .gguf files"]
    if wanted:
        w = wanted.strip()
        exact = [c for c in cands if c["file"] == w or os.path.basename(c["file"]) == os.path.basename(w)]
        byq = [c for c in cands if (c["quant"] or "").lower() == w.lower() or (c["quant"] or "").lower().endswith("-" + w.lower())]
        sub = [c for c in cands if w.lower() in c["file"].lower()]
        hit = exact or sorted(byq, key=_rank) or sorted(sub, key=_rank)
        if not hit:
            raise InspectError(f"no GGUF matching {wanted!r}. Available: "
                               + ", ".join(f"{os.path.basename(c['file'])} ({c['size_gb']} GB)" for c in cands))
        c = hit[0]
        if c["size_gb"] > WEIGHT_FIT_GB:
            warn.append(f"{c['size_gb']} GB of weights leaves little VRAM for context on 2xT4 (30 GB)")
        return c, warn
    fits = [c for c in cands if c["size_gb"] <= WEIGHT_FIT_GB]
    if not fits:
        c = min(cands, key=lambda c: c["size_gb"])
        return c, [f"smallest GGUF is {c['size_gb']} GB; it will not leave room for context on 2xT4 (30 GB). "
                   "Look for a smaller quant or model."]
    q5 = [c for c in fits if (c["quant"] or "").replace("i1-", "") == "Q5_K_M" and c["size_gb"] <= SMALL_Q5_GB]
    if q5:
        return sorted(q5, key=_rank)[0], warn
    best = sorted(fits, key=_rank)[0]
    if best["quant"] is None or best["quant"].replace("i1-", "") not in _PREF:
        warn.append("could not recognise the quantization names; picked the largest file that fits")
        best = max(fits, key=lambda c: c["size_gb"])
    return best, warn


# GGUF header (metadata only: a Range request for the first few hundred KB)
_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
            7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def parse_gguf_header(buf, want=None):
    """Scalar/short-array metadata from the first bytes of a GGUF file. Stops at the first key it
    cannot finish (truncated buffer) or once every `want` suffix has been seen. Returns a dict."""
    out = {}
    if len(buf) < 24 or buf[:4] != b"GGUF":
        return out
    pos = 4
    try:
        _ver, _nt, nkv = struct.unpack_from("<IQQ", buf, pos); pos += 20

        def rd(fmt):
            nonlocal pos
            v = struct.unpack_from(fmt, buf, pos); pos += struct.calcsize(fmt)
            return v[0]

        def rstr():
            nonlocal pos
            n = rd("<Q")
            if n > 1 << 24 or pos + n > len(buf):
                raise struct.error("truncated")
            s = buf[pos:pos + n].decode("utf-8", "replace"); pos += n
            return s

        def rval(t):
            if t in _SCALARS:
                return rd(_SCALARS[t][0])
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("<I"), rd("<Q")
                if n > 4096:
                    raise struct.error("big array")        # tokenizer tables: everything we need came before
                return [rval(et) for _ in range(n)]
            raise struct.error("bad type")

        for _ in range(nkv):
            key = rstr()
            out[key] = rval(rd("<I"))
            if want and all(any(k.endswith(w) for k in out) for w in want):
                break
    except (struct.error, IndexError):
        pass
    return out


def _arch_info(meta):
    """Context length and KV-cache bytes per token (f16) from GGUF metadata, or None parts."""
    arch = meta.get("general.architecture")
    if not arch:
        return {}
    g = lambda k: meta.get(f"{arch}.{k}")
    n_layer, emb, heads, kv = g("block_count"), g("embedding_length"), g("attention.head_count"), g("attention.head_count_kv")
    heads = max(heads) if isinstance(heads, list) else heads
    kv = max(kv) if isinstance(kv, list) else (kv or heads)
    hd = g("attention.key_length") or (emb // heads if emb and heads else None)
    vd = g("attention.value_length") or hd
    info = {"arch": arch, "native_ctx": g("context_length"), "layers": n_layer}
    if n_layer and kv and hd:
        interval = g("full_attention_interval") or 1       # hybrid (linear + full attention) models
        attn_layers = n_layer / interval if interval and interval > 1 else n_layer
        info["kv_bytes_per_token"] = attn_layers * kv * (hd + (vd or hd)) * 2
    return info


def _pick_ctx(native, kv_per_tok, weights_gb):
    if not kv_per_tok:
        return min(native or DEFAULT_CTX, DEFAULT_CTX)
    room = (VRAM_TOTAL_GB - weights_gb) * 1e9 / kv_per_tok
    cap = min(native or CTX_STEPS[-1], room)
    ok = [c for c in CTX_STEPS if c <= cap]
    return ok[-1] if ok else CTX_STEPS[0]


def _sampling(name):
    n = name.lower()
    if "gpt-oss" in n:
        return ["--temp", "1.0", "--top-p", "1.0"], ["thinking"]
    if re.search(r"coder|code(?!x)|codestral|devstral", n):
        return ["--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0"], ["coder"]
    instruct_only = "instruct" in n and "thinking" not in n
    if not instruct_only and re.search(r"qwen3|qwq|r1|thinking|reason|magistral|glm-z1|deepseek-r1|phi-4-reasoning", n):
        return (["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0",
                 "--reasoning-format", "deepseek", "--reasoning-budget", "16384"], ["thinking"])
    return ["--temp", "0.7", "--top-p", "0.9"], []


def _tags(name, extra):
    n = name.lower()
    tags = list(extra)
    if re.search(r"abliterat|uncensor|dolphin|heretic", n):
        tags.append("abliterated")
    if re.search(r"a\d+b|moe", n):
        tags.append("moe")
    return sorted(set(tags)) + ["unverified"]


def _key_for(repo, taken):
    name = repo.split("/")[1]
    name = re.sub(r"[-_.]?(i1[-_.])?gguf$", "", name, flags=re.I)
    key = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-.") or "model"
    base, i = key, 2
    while key in taken:
        key = f"{base}-{i}"; i += 1
    return key


def inspect_gguf(repo, files, token, wanted, ctx, info):
    cands, notes = group_ggufs(files)
    chosen, warn = pick_gguf(cands, wanted)
    warn = notes + warn
    if chosen is None:
        raise InspectError("this repo lists .gguf files but none are usable (" + "; ".join(notes) + ")")
    meta = {}
    try:      # first ~512 KB of the chosen file: the architecture keys come before the tokenizer tables
        body, _ = _get(f"{HF}/{repo}/resolve/main/{urllib.parse.quote(chosen['file'])}", token,
                       {"Range": "bytes=0-524287"}, timeout=25)
        meta = parse_gguf_header(body, want=("context_length", "attention.head_count_kv", "full_attention_interval"))
    except InspectError:
        pass
    arch = _arch_info(meta)
    if not arch.get("native_ctx"):
        native = ((info or {}).get("gguf") or {}).get("context_length") or None    # HF's own parsed copy
        arch["native_ctx"] = native
    weights = chosen["size_gb"]
    auto_ctx = _pick_ctx(arch.get("native_ctx"), arch.get("kv_bytes_per_token"), weights)
    use_ctx = int(ctx) if ctx else auto_ctx
    if arch.get("kv_bytes_per_token"):
        vram = weights + use_ctx * arch["kv_bytes_per_token"] / 1e9 + BUFFER_GB
    else:
        vram = weights * 1.1 + 2.5
        warn.append("could not read the model's attention shape; context and VRAM are rough estimates")
    if ctx and arch.get("native_ctx") and int(ctx) > arch["native_ctx"]:
        warn.append(f"ctx {ctx} is beyond the model's native {arch['native_ctx']}")
    if vram > 29:
        warn.append(f"~{math.ceil(vram)} GB needed at ctx {use_ctx}: does not fit 2xT4 (30 GB); lower --ctx or pick a smaller quant")
    args, ex = _sampling(repo.split("/")[1])
    args += ["-b", "2048", "-ub", "1024", "--cache-type-k", "f16", "--cache-type-v", "f16", "-sm", "tensor"]
    quant = chosen["quant"] or "Q4_K_M"
    entry = {"hf": repo, "quant": quant, "file": chosen["file"], "vram_gb": math.ceil(vram), "ctx": use_ctx, "args": args,
             "desc": f"{repo.split('/')[1]} ({quant}, {weights:g} GB) added with kllm add. Unverified on kllm.",
             "tags": _tags(repo.split("/")[1], ex)}
    if chosen["parts"] > 1:
        entry["file"] = chosen["file"]
    return {"runtime": "llama.cpp", "kind": "chat", "repo": repo, "entry": entry, "files": cands, "chosen": chosen["file"],
            "weights_gb": weights, "vram_gb": entry["vram_gb"], "ctx": use_ctx, "native_ctx": arch.get("native_ctx"),
            "warnings": warn, "fits": vram <= 29 and weights <= 28}


# ---- diffusers ----------------------------------------------------------------------------------
_VIDEO = re.compile(r"video|wan|ltx|cogvideo|mochi|allegro|animatediff|i2vgen", re.I)
_I2V = re.compile(r"imagetovideo|i2v|stablevideodiffusion|img2vid|video2video|videotovideo|inpaint|img2img|imagetoimage", re.I)

# (class-name regex, kind-specific defaults); first match wins
_DEFAULTS = [
    (r"^Wan", {"size": "832x480", "num_frames": 81, "fps": 15, "steps": 30, "guidance_scale": 5.0,
               "size_multiple": 16, "fp32": ["vae"], "warmup": False}),
    (r"LTX", {"size": "704x480", "num_frames": 97, "fps": 24, "steps": 30, "guidance_scale": 3.0,
              "size_multiple": 32, "warmup": False}),
    (r"CogVideoX", {"size": "720x480", "num_frames": 49, "fps": 8, "steps": 50, "guidance_scale": 6.0,
                    "size_multiple": 16, "warmup": False}),
    (r"HunyuanVideo", {"size": "512x320", "num_frames": 61, "fps": 15, "steps": 30, "guidance_scale": 6.0,
                       "size_multiple": 16, "warmup": False}),
    (r"Mochi", {"size": "848x480", "num_frames": 49, "fps": 30, "steps": 50, "guidance_scale": 4.5,
                "size_multiple": 16, "warmup": False}),
    (r"Video|Allegro|AnimateDiff", {"size": "512x512", "num_frames": 49, "fps": 16, "steps": 30, "guidance_scale": 6.0,
                                    "size_multiple": 16, "warmup": False}),
    (r"StableDiffusionXL", {"size": "1024x1024", "steps": 30, "guidance_scale": 7.0}),
    (r"StableDiffusion3", {"size": "1024x1024", "steps": 28, "guidance_scale": 7.0}),
    (r"Flux", {"size": "1024x1024", "steps": 28, "guidance_scale": 3.5}),
    (r"^StableDiffusionPipeline|^StableDiffusion$", {"size": "512x512", "steps": 30, "guidance_scale": 7.5}),
    (r"PixArt|Sana|Lumina|AuraFlow|Kolors|HiDream", {"size": "1024x1024", "steps": 25, "guidance_scale": 4.5}),
]


def _comp_dirs(index):
    return [k for k, v in index.items() if not k.startswith("_") and isinstance(v, list) and v and v[0]]


def safetensors_dtype(repo, path, token=None):
    """Stored dtype of the first tensor of a .safetensors file (header via a Range request), or None."""
    try:
        url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(path)}"
        body, _ = _get(url, token, {"Range": "bytes=0-131071"}, timeout=25)
        n = struct.unpack_from("<Q", body, 0)[0]
        if 8 + n > len(body):
            body, _ = _get(url, token, {"Range": f"bytes=0-{8 + n - 1}"}, timeout=25)
        hdr = json.loads(body[8:8 + n])
        return next((v.get("dtype") for k, v in hdr.items() if k != "__metadata__"), None)
    except (InspectError, struct.error, ValueError, TypeError):
        return None


def diffusers_footprint(files, index, dtype_of=None):
    """fp16 footprint from file sizes: per component folder, the fp16-variant weights if the repo has
    them, else the plain safetensors (or .bin), halved when stored as fp32 (kllm loads fp16).
    -> (total GB, largest component GB, has_fp16_variant)."""
    comps, any_variant = {}, False
    for d in _comp_dirs(index):
        fs = [f for f in files if f["path"].startswith(d + "/")]
        st = [f for f in fs if f["path"].endswith(".safetensors")] or \
             [f for f in fs if f["path"].endswith((".bin", ".ckpt", ".pt", ".pth"))]
        var = [f for f in st if re.search(r"\.fp16([.-])", os.path.basename(f["path"]))]
        if var:
            any_variant = True
            use = var
        else:
            use = [f for f in st if not re.search(r"\.(fp32|bf16|non_ema|ema)([.-])", os.path.basename(f["path"]))] or st
        size = sum(f["size"] for f in use) / 1e9
        if not var and dtype_of and use and use[0]["path"].endswith(".safetensors") \
                and str(dtype_of(use[0]["path"]) or "").upper() in ("F32", "FLOAT32"):
            size /= 2
        if size:
            comps[d] = size
    return sum(comps.values()), max(comps.values() or [0]), any_variant


def inspect_diffusers(repo, files, token, ctx, name_hint=None):
    body, _ = _get(f"{HF}/{repo}/resolve/main/model_index.json", token)
    try:
        index = json.loads(body)
    except ValueError:
        raise InspectError("model_index.json is not valid JSON") from None
    cls = str(index.get("_class_name") or "")
    if _I2V.search(cls):
        raise InspectError(f"{cls} needs an input image/video; kllm serves text-to-image and text-to-video "
                           "(/v1/images/generations, /v1/videos/generations)")
    kind = "video" if _VIDEO.search(cls) else "image"
    total, largest, variant = diffusers_footprint(files, index, lambda p: safetensors_dtype(repo, p, token))
    warn = []
    if total == 0:
        raise InspectError("no safetensors weights found in the pipeline's component folders")
    if total > DIFFUSERS_RAM_GB:
        warn.append(f"~{total:.0f} GB of weights cannot be held in the kernel's ~31 GB RAM even with offloading")
    if total <= DIFFUSERS_GPU_GB:
        vram, how = math.ceil(total * 1.15 + 1), "fits one T4"
    elif largest <= 13.5:
        vram, how = math.ceil(min(largest + 3, 15)), "model CPU offload (components move to the GPU one at a time; slower)"
    else:
        vram, how = 4, "sequential CPU offload (one layer at a time; very slow)"
        warn.append(f"largest component is {largest:.0f} GB, bigger than a T4: sequential offload, expect minutes per step")
    opts = {}
    low = repo.lower()
    for rx, d in _DEFAULTS:
        if re.search(rx, cls):
            opts = json.loads(json.dumps(d)); break
    if not opts:
        opts = {"size": "512x512" if kind == "video" else "1024x1024", "steps": 30, "guidance_scale": 7.0}
        if kind == "video":
            opts.update(num_frames=49, fps=16, size_multiple=16, warmup=False)
    if re.search(r"turbo|lightning|lcm|schnell|hyper", low):
        opts.update(steps=4, guidance_scale=0.0)
        if "turbo" in low:
            opts["size"] = "512x512"
    if not variant:
        opts["variant"] = None                           # no fp16 variant files: do not try
    if re.search(r"flux|wan|ltx|hunyuan|mochi|cogvideox-5b|sd3|stable-diffusion-3", low):
        warn.append("this family is published for bf16; kllm runs fp16 on T4 (no bf16), which can produce "
                    "black/NaN output on some models: verify before relying on it")
    nm = repo.split("/")[1]
    entry = {"runtime": "diffusers", "kind": kind, "hf": repo, "vram_gb": vram, "opts": opts,
             "desc": f"{nm} ({cls}, ~{total:.1f} GB fp16, {how}) added with kllm add. Unverified on kllm.",
             "tags": sorted({kind, "diffusers"}) + ["unverified"]}
    return {"runtime": "diffusers", "kind": kind, "repo": repo, "entry": entry, "pipeline": cls, "files": [],
            "chosen": None, "weights_gb": round(total, 1), "vram_gb": vram, "ctx": None, "placement": how,
            "warnings": warn, "fits": total <= DIFFUSERS_RAM_GB}


# ---- entry point ----------------------------------------------------------------------------------
def _suggest_gguf(repo, token):
    name = re.sub(r"[-_.]?(gguf)$", "", repo.split("/")[1], flags=re.I)
    try:
        d, _ = _json(f"{HF}/api/models?search={urllib.parse.quote(name + ' gguf')}&sort=downloads&limit=4", token)
        return [m["id"] for m in d if "gguf" in m["id"].lower()][:3]
    except (InspectError, TypeError, KeyError):
        return []


def inspect_source(source, file=None, ctx=None, name=None, token=None, taken=()):
    """Everything `kllm add` and the panel need. -> dict with runtime, kind, entry (the models.yaml entry),
    key (suggested/validated name), files (GGUF candidates with sizes), chosen, warnings, fits."""
    token = token or os.environ.get("HF_TOKEN") or None
    repo, wanted = parse_source(source)
    wanted = file or wanted
    info, _ = _json(f"{HF}/api/models/{repo}", token)
    files = fetch_tree(repo, token)
    paths = {f["path"] for f in files}
    res = None
    if any(p.lower().endswith(".gguf") for p in paths) or "gguf" in (info.get("tags") or []):
        res = inspect_gguf(repo, files, token, wanted, ctx, info)
    elif "model_index.json" in paths:
        res = inspect_diffusers(repo, files, token, ctx)
    else:
        sample = ", ".join(sorted(paths)[:6])
        sugg = _suggest_gguf(repo, token)
        raise InspectError(
            f"{repo} is neither a GGUF repo nor a diffusers pipeline (files: {sample or 'none'}). kllm runs "
            "GGUF models (llama.cpp text models) and diffusers pipelines (model_index.json: image/video). "
            + ("A transformers-format LLM must be converted to GGUF; GGUF versions on HF: " + ", ".join(sugg)
               if sugg else "A transformers-format LLM must be converted to GGUF: search the hub for '<name> GGUF'."))
    if info.get("gated") and not token:
        res["warnings"].append("this repo is gated: accept the license on HuggingFace and set hf_token in config.yaml")
    key = (name or "").strip() or _key_for(repo, set(taken))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", key):
        raise InspectError(f"name {key!r} must be letters, digits, '.', '_' or '-'")
    if key in taken:
        raise InspectError(f"'{key}' already exists in models.yaml; choose another --name")
    res["key"] = key
    return res


# ---- models.yaml (append only: comments and existing entries are never touched) ---------------------
def entry_yaml(key, entry, comment=None):
    lines = [""]
    if comment:
        lines.append(f"  # {comment}")
    lines.append(f"  {key}:")
    for k, v in entry.items():
        if v is None:
            continue
        if isinstance(v, str):
            lines.append(f"    {k}: {json.dumps(v)}")
        else:
            lines.append(f"    {k}: {json.dumps(v)}")           # JSON is valid YAML flow syntax
    return "\n".join(lines) + "\n"


def write_entry(key, entry, path, comment=None):
    """Append the entry to models.yaml; the file stays valid YAML or is restored untouched."""
    text = open(path, encoding="utf-8").read()
    try:
        models = (yaml.safe_load(text) or {}).get("models") or {}
    except yaml.YAMLError as e:
        raise InspectError(f"{path} is not valid YAML before the change: {e}") from None
    if key in models:
        raise InspectError(f"'{key}' already exists in models.yaml")
    new = text.rstrip("\n") + "\n" + entry_yaml(key, entry, comment or f"added {time.strftime('%Y-%m-%d')} with kllm add")
    try:
        got = (yaml.safe_load(new) or {})["models"][key]
    except (yaml.YAMLError, KeyError, TypeError) as e:
        raise InspectError(f"generated entry for '{key}' is not valid YAML ({e}); models.yaml unchanged") from None
    if got.get("hf") != entry["hf"]:
        raise InspectError("generated entry did not round-trip; models.yaml unchanged")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new)
    os.replace(tmp, path)
    return key


def summary(res):
    """Human-readable lines for the CLI."""
    kind = res["kind"]
    out = [f"{res['repo']}: {kind} model, runtime {res['runtime']}"]
    if res["runtime"] == "llama.cpp":
        out.append(f"  picked {res['chosen']} ({res['weights_gb']} GB), ctx {res['ctx']}"
                   + (f" (native {res['native_ctx']})" if res.get("native_ctx") else "")
                   + f", ~{res['vram_gb']} GB VRAM")
        others = [c for c in res["files"] if c["file"] != res["chosen"]]
        if others:
            out.append("  other files: " + ", ".join(f"{c['quant'] or os.path.basename(c['file'])} {c['size_gb']}G" for c in others[:12]))
    else:
        out.append(f"  {res['pipeline']}: ~{res['weights_gb']} GB fp16, {res['placement']}, ~{res['vram_gb']} GB VRAM")
    out += [f"  warning: {w}" for w in res["warnings"]]
    return out
