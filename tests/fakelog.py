"""Deterministic fake requests.jsonl records shaped like relay._Meter.close() writes them."""
import json, random

MODELS = ["qwen3.6-35b-abliterated", "qwen3-coder-30b-abliterated"]
BACKENDS = ["kaggle:kg1", "kaggle:kg2", "kaggle:kg3"]


def make_records(now, days=7, per_hour=6, seed=1):
    rng = random.Random(seed)
    out = []
    for h in range(days * 24):
        base = int(now // 3600 * 3600) - h * 3600
        for _ in range(per_hour if h % 5 else 0):         # every 5th hour is silent
            ts = base + rng.randrange(0, 3600)
            if ts > now:
                continue
            m = rng.choice(MODELS)
            ctx = rng.choice([2000, 9000, 25000, 50000, 90000, 150000])
            cached = int(ctx * rng.uniform(0.85, 0.99)) if ctx > 4000 else 0
            out_tok = rng.randrange(40, 2500)
            think = int(out_tok * (0.7 if m.startswith("qwen3.6") else 0.0) * 4)
            r = {"ts": ts, "backend": rng.choice(BACKENDS), "path": "v1/messages", "model": m, "n_tools": 12,
                 "new_in": ctx - cached, "cached_in": cached, "out": out_tok,
                 "prompt_tps": 900, "gen_tps": round(75 - ctx / 5000 + rng.uniform(-4, 4), 1),
                 "think_chars": think, "text_chars": out_tok * 4 - think,
                 "status": 200, "total_s": round(out_tok / 60 + 1.5, 2), "ttft_s": round(0.6 + ctx / 60000 + rng.random(), 2)}
            if rng.random() < 0.05:
                r.update(status=502, total_s=0.4, ttft_s=None, out=0, think_chars=0, text_chars=0)
                r.pop("gen_tps")
            elif rng.random() < 0.03:
                r["loop_cut"] = {"period": 40, "span": 1500, "sample": "abc"}
            out.append(r)
    out.append({"ts": int(now) - 50, "backend": "kaggle:kg1", "path": "/v1/images/generations",
                "model": "sdxl-turbo", "kind": "image", "status": 200, "total_s": 21.0})
    out.sort(key=lambda r: r["ts"])
    return out


def write(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
