"""Render the kernel template for a registry entry and import it as a module, without running main()
(no Kaggle, no torch, no network)."""
import types, subprocess
from gateway import common, orchestrator as orch

CFG = {"accounts": [{"name": "kg1", "user": "u"}], "kernel_slug": "kllm", "gist_id": "g",
       "gist_token": "t", "gist_user": "x", "extra_server_args": []}


def load_kernel(model, key="kllm-testkey", cfg=None):
    """model: a registry key (str) or an entry dict (gets key 'test-model')."""
    if isinstance(model, str):
        m = common.resolve_model(model)
    else:
        m = dict(model); m.setdefault("key", "test-model")
    bdir, _ = orch.render_kernel(dict(CFG, **(cfg or {})), "kg1", m, key, 39000)
    src = (bdir / "server_kernel.py").read_text(encoding="utf-8")
    mod = types.ModuleType("kllm_kernel"); mod.__name__ = "kllm_kernel"
    exec(compile(src, "server_kernel.py", "exec"), mod.__dict__)
    mod.ntfy = lambda *a, **k: True
    return mod


def llama_argv(model):
    """The llama-server argv the kernel would launch for a registry entry (Popen is intercepted)."""
    import tempfile, os
    mod = load_kernel(model)
    class Got(Exception): pass
    cap = {}
    def popen(args, **kw):
        cap["args"] = args; raise Got()
    d = tempfile.mkdtemp(prefix="kllm-kernel-")
    real_open = open
    def fake_open(path, *a, **k):
        if isinstance(path, str) and (path.startswith("/kaggle/") or path.startswith("/tmp/chat_template")):
            path = os.path.join(d, os.path.basename(path))
        return real_open(path, *a, **k)
    mod.open = fake_open
    mod.stop_requested = lambda *a, **k: False
    mod.get_server_binary = lambda: "/bin/llama-server"
    mod.get_cloudflared = lambda: "/bin/cf"
    mod.gpu_info = lambda: "gpu"
    mod.subprocess = types.SimpleNamespace(Popen=popen, CalledProcessError=Exception, run=subprocess.run,
                                           check_output=subprocess.check_output, STDOUT=subprocess.STDOUT)
    try:
        mod.main()
    except Got:
        pass
    return cap["args"]
