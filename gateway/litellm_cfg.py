"""Generate litellm/config.yaml from models.yaml (OPTIONAL LiteLLM proxy; see README).
The gateway then advertises exactly the registry models plus a generic 'kllm' handle.
Run `kllm litellm-config` after editing models.yaml."""
import pathlib
from . import common, media


def generate(out=None):
    cfg = common.load_config()
    RELAY = f"http://127.0.0.1:{cfg.get('relay_port',8080)}/v1"
    KEY = cfg.get("gateway_api_key", "sk-kllm-local")
    # chat models only: image/video models are called on the relay directly (/v1/images|videos/generations)
    models = {k: m for k, m in common.load_models().items() if media.kind(m) == "chat"}

    def entry(name):
        return f"""  - model_name: {name}
    litellm_params:
      model: openai/{name}
      api_base: {RELAY}
      api_key: dummy
      timeout: 1800
      stream_timeout: 1800
"""

    lines = ["# AUTO-GENERATED from models.yaml by `kllm litellm-config` — edit models.yaml, not here.",
             "# LiteLLM gateway in front of the kllm relay. 'kllm' = whatever the pool serves;",
             "# the registry keys route to the backend serving that model when you run several.",
             "", "model_list:", entry("kllm").rstrip()]
    for k in models:
        lines.append(entry(k).rstrip())
    lines += ["",
              "litellm_settings:",
              "  drop_params: true",
              "  request_timeout: 1800",
              "  num_retries: 0               # the relay fails over; a retry would re-run a whole generation",
              "  telemetry: false",
              "",
              "general_settings:",
              f"  master_key: {KEY}",
              ""]
    out = pathlib.Path(out or common.ROOT / "litellm" / "config.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out, ["kllm", *models.keys()]
