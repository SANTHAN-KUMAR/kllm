"""`kllm uninstall`: remove everything kllm put on this machine, after showing exactly what will go.

Order matters: GPUs first (so no Kaggle quota keeps burning), then the relay and the background service (so
nothing restarts kllm), then the icon and the tool configs it wrote, then its own files, and the package last.

Never removed: your own Kaggle credentials (~/.kaggle/kaggle.json or ~/.kaggle/access_token), anything kllm did not
create, and, in a git checkout, the repository itself (only kllm's runtime files inside it).
"""
import os, pathlib, shutil, subprocess, sys, time, urllib.request

from . import common

# runtime files kllm writes into its root folder (all of them when the root is kllm's own config folder)
RUNTIME = ["config.yaml", "config.yaml.bak", "models.yaml", "state", "logs", "media", "kaggle_accounts", "litellm",
           "bin/cloudflared", "bin/cloudflared.exe"]


def _root_is_dedicated(root):
    """True when the root is kllm's own folder (pip/uv/pipx install), not a git checkout or another project."""
    root = pathlib.Path(root)
    return not (root / "pyproject.toml").exists() and not (root / ".git").exists() and root.name.lower() == "kllm"


def files_to_remove(root=None):
    root = pathlib.Path(root or common.ROOT)
    if not root.exists():
        return []
    if _root_is_dedicated(root):
        return [root]
    return [root / r for r in RUNTIME if (root / r).exists()]


def credential_files(cfg):
    """Credential copies kllm itself created for its accounts (never the user's own ~/.kaggle)."""
    from . import accounts
    out = []
    for a in cfg.get("accounts") or []:
        try:
            out += accounts.removable_credentials(a.get("name", ""))
        except Exception:
            pass
    return out


def tool_undos():
    """(tool id, what, label) for every tool setting kllm's Connect page wrote and can undo."""
    from . import connect
    out = []
    try:
        rows = connect.detect(force=True)
    except Exception:
        return out
    for r in rows:
        for o in r.get("options") or []:
            if o.get("can_undo"):
                out.append((r["id"], o["what"], f"{r['name']}: {o['label']}"))
    return out


def package_command():
    """How to remove the package itself, matching how it was installed (see updates.upgrade_command)."""
    from . import updates
    method = updates.upgrade_command()["method"]
    if method == "uv":
        return method, ["uv", "tool", "uninstall", updates.PACKAGE]
    if method == "pipx":
        return method, ["pipx", "uninstall", updates.PACKAGE]
    if method == "pip":
        return method, [sys.executable, "-m", "pip", "uninstall", "-y", updates.PACKAGE]
    return method, None                                   # a git checkout: the folder is the user's to delete


def relay_running():
    try:
        with urllib.request.urlopen(common.relay_base() + "/__relay/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def stop_relay(wait_s=15):
    """Ask the local relay to exit; True when it is gone."""
    try:
        req = urllib.request.Request(common.relay_base() + "/__relay/shutdown", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if not relay_running():
            return True
        time.sleep(0.5)
    return not relay_running()


def plan(keep_config=False):
    """What `kllm uninstall` would do, as [(label, kind, detail)], for the preview."""
    try:
        cfg = common.load_config()
    except Exception:
        cfg = {}
    steps = []
    if (common.read_state() or {}).get("backends"):
        steps.append(("Stop running Kaggle GPUs", "gpus", None))
    if relay_running():
        steps.append((f"Stop the kllm relay at {common.relay_base()}", "relay", None))
    steps.append(("Remove the background service, if installed", "service", None))
    steps.append(("Remove the kllm desktop and app-menu icon, if present", "shortcut", None))
    for tool, what, label in tool_undos():
        steps.append((f"Undo kllm's change to {label}", "tool", (tool, what)))
    if not keep_config:
        for p in credential_files(cfg):
            steps.append((f"Delete kllm's copy of Kaggle credentials: {p}", "path", p))
        for p in files_to_remove():
            steps.append((f"Delete {p}", "path", p))
    method, cmd = package_command()
    if cmd:
        steps.append((f"Uninstall the kllm-gateway package ({method})", "package", cmd))
    else:
        steps.append((f"kllm runs from a git checkout ({common.PKG_ROOT}); delete that folder yourself if you no "
                      "longer want it", "note", None))
    return steps


def run(steps, log=print):
    """Carry out the plan. Each step is independent: one failing does not stop the rest."""
    failed = 0
    for label, kind, detail in steps:
        try:
            if kind == "gpus":
                from . import orchestrator
                orchestrator.stop(account=None, log=lambda *a: None)
            elif kind == "relay":
                if not stop_relay():
                    raise RuntimeError("the relay is still running")
            elif kind == "service":
                from . import service
                service.install(uninstall=True, log=lambda *a: None)
            elif kind == "shortcut":
                from . import shortcut
                shortcut.remove(log=lambda *a: None)
            elif kind == "tool":
                from . import connect
                connect.undo(*detail)
            elif kind == "path":
                p = pathlib.Path(detail)
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            elif kind == "package":
                _remove_package(detail, log)
            elif kind == "note":
                pass
            log(f"  done  {label}")
        except Exception as e:
            failed += 1
            log(f"  FAIL  {label}: {e}")
    return failed


def _remove_package(cmd, log):
    if os.name == "nt":
        # kllm.exe is running this very command and Windows will not delete a running file: finish in a new
        # console after this process exits.
        script = (f"import subprocess, time; time.sleep(2); r = subprocess.call({cmd!r}); print(); "
                  f"print('kllm is uninstalled.' if r == 0 else 'uninstall failed (exit %d)' % r); "
                  f"input('Press Enter to close this window')")
        subprocess.Popen([sys.executable, "-c", script], creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10),
                         close_fds=True, cwd=os.path.expanduser("~"))
        log("  ...   the package is removed in a new window once this command exits")
        return
    r = subprocess.call(cmd)
    if r != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} exited with {r}")
