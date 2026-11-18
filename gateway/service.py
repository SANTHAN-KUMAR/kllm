"""`kllm install-service`: run `kllm serve` in the background and at login.

  Linux   systemd user unit  ~/.config/systemd/user/kllm.service  (one process: relay + supervisor)
  macOS   launchd agent      ~/Library/LaunchAgents/com.kllm.serve.plist
  other   prints how to run `kllm serve` manually
The file generators are pure functions (unit-tested); nothing here launches a Kaggle kernel.
"""
import os, plistlib, shutil, subprocess, sys, pathlib
from . import common

UNIT_NAME = "kllm.service"
LAUNCHD_LABEL = "com.kllm.serve"
LEGACY_UNITS = ("kllm-relay", "kllm-supervisor", "kllm-litellm")


def _q(s):
    """Quote one systemd ExecStart/Environment word."""
    s = str(s).replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"' if any(c in s for c in ' \t"\'') else s


def serve_command(python=None):
    return [str(python or sys.executable), "-m", "gateway.cli", "serve"]


def _path_env(home):
    h = str(home).replace("\\", "/")
    parts = [f"{h}/.local/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    return ":".join(parts)


def systemd_unit(root, python=None, pkg_root=None, home=None):
    """Text of ~/.config/systemd/user/kllm.service. `pkg_root` is only needed (and added to
    PYTHONPATH) for a source checkout; an installed package imports normally."""
    home = home or pathlib.Path.home()
    lines = [
        "[Unit]",
        "Description=kllm: Kaggle GPU LLM gateway (relay + supervisor in one process)",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"Environment={_q('KLLM_ROOT=' + str(root))}",
        f"Environment={_q('PATH=' + _path_env(home))}",
    ]
    if pkg_root:
        lines.append(f"Environment={_q('PYTHONPATH=' + str(pkg_root))}")
    lines += [
        f"ExecStart={' '.join(_q(a) for a in serve_command(python))}",
        f"WorkingDirectory={root}",
        "Restart=always",
        "RestartSec=3",
        "Nice=5",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ]
    return "\n".join(lines)


def launchd_plist(root, python=None, pkg_root=None, home=None):
    """Bytes of the launchd agent plist."""
    home = home or pathlib.Path.home()
    env = {"KLLM_ROOT": str(root).replace("\\", "/"), "PATH": _path_env(home) + ":/opt/homebrew/bin"}
    if pkg_root:
        env["PYTHONPATH"] = str(pkg_root)
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": serve_command(python),
        "EnvironmentVariables": env,
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": (str(root) + "/logs/serve.log").replace("\\", "/"),
        "StandardErrorPath": (str(root) + "/logs/serve.log").replace("\\", "/"),
    })


def manual_instructions(root):
    return (f"No service manager support for this platform. Run it in a terminal (or your own "
            f"autostart mechanism):\n  kllm serve\n  (KLLM_ROOT={root})\n"
            + ("On Windows: Task Scheduler -> 'At log on' -> program: kllm, arguments: serve.\n"
               if os.name == "nt" else ""))


def windows_task_command(root, python=None, pkg_root=None):
    """Pure: the `schtasks /Create` argv that keeps `kllm serve` running at logon.

    The relay itself is tiny (~55 MB); keeping it always on is what makes GPU
    backends fully on-demand: the first request for a model starts it (autoscale),
    and idle backends stop after `idle_stop_minutes` — no manual `kllm up/down`."""
    root = str(root)
    py = str(python or sys.executable)
    inner = f'"{py}" -m gateway.cli serve'
    if pkg_root:
        inner = f'set PYTHONPATH={pkg_root}&& {inner}'
    # schtasks /TR runs via cmd.exe; the whole inner command is quoted once.
    tr = f'cmd /d /s /c "cd /d \\"{root}\\" && set KLLM_ROOT={root}&& {inner}"'
    return ["schtasks", "/Create", "/TN", "kllm serve", "/TR", tr,
            "/SC", "ONLOGON", "/RL", "LIMITED", "/F"]


def windows_task_xml(root, python=None):
    """Pure: Task Scheduler XML for a SILENT at-logon `kllm serve`.

    Unlike the /TR form, this runs with no console window ever (Hidden), starts
    on battery too, and never auto-restarts (the supervisor lock + `ensure_serve`
    make duplicates impossible anyway). `python -m` needs no PYTHONPATH/KLLM_ROOT
    overrides when ROOT is a checkout: cwd is on sys.path and the checkout dir
    is the default root.
    """
    root = str(root)
    py = str(python or sys.executable)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>kllm: Kaggle GPU LLM gateway (relay + supervisor in one process). Tiny always-on relay; GPU backends start on first use and stop when idle.</Description>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Hidden>true</Hidden>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
  </Settings>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{py}</Command>
      <Arguments>-m gateway.cli serve</Arguments>
      <WorkingDirectory>{root}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def install_windows_task(root=None, python=None, log=print):
    """Create (or replace) the at-logon 'kllm serve' task. Returns True on success."""
    import tempfile
    root = pathlib.Path(root or common.ROOT)
    xml = windows_task_xml(root, python)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                         encoding="utf-16") as f:
            f.write(xml)
            xml_path = f.name
        r = subprocess.run(["schtasks", "/Create", "/TN", "kllm serve",
                            "/XML", xml_path, "/F"],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        log(f"  ! schtasks /Create: {e}")
        return False
    finally:
        try:
            os.unlink(xml_path)
        except (NameError, OSError):
            pass
    if r.returncode != 0:
        log(f"  ! schtasks /Create: {(r.stderr or r.stdout).strip()[:300]}")
        log("  re-run this command elevated (right-click -> Run as administrator): "
            "python -m gateway.cli install-service")
        return False
    log("kllm serve will start at logon (Task Scheduler: 'kllm serve', hidden, no window). "
        "Starting it now...")
    _run(["schtasks", "/Run", "/TN", "kllm serve"], log)
    return True


def _run(cmd, log):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        log(f"  ! {' '.join(cmd)}: {e}"); return False
    if r.returncode != 0:
        log(f"  ! {' '.join(cmd)}: {(r.stderr or r.stdout).strip()[:200]}")
    return r.returncode == 0


def _legacy_active():
    out = []
    for u in LEGACY_UNITS:
        try:
            if subprocess.run(["systemctl", "--user", "is-active", u + ".service"],
                              capture_output=True, text=True, timeout=10).stdout.strip() == "active":
                out.append(u)
        except Exception:
            pass
    return out


def install(platform=None, dest_dir=None, start=True, uninstall=False, print_only=False,
            log=print, root=None, python=None):
    """Install (or remove) the background service. Returns the path written (or None)."""
    platform = platform or sys.platform
    root = pathlib.Path(root or common.ROOT)
    pkg_root = None if not common.is_checkout() else common.PKG_ROOT
    if platform.startswith("linux"):
        d = pathlib.Path(dest_dir or pathlib.Path.home() / ".config" / "systemd" / "user")
        path = d / UNIT_NAME
        if uninstall:
            _run(["systemctl", "--user", "disable", "--now", UNIT_NAME], log)
            path.unlink(missing_ok=True); _run(["systemctl", "--user", "daemon-reload"], log)
            log(f"removed {path}"); return None
        text = systemd_unit(root, python, pkg_root)
        if print_only:
            print(text); return None
        d.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
        log(f"wrote {path}")
        if start and shutil.which("systemctl"):
            legacy = _legacy_active()
            if legacy:
                log(f"  ! legacy units are active ({', '.join(legacy)}) and would clash with kllm.service "
                    "(same port). Stop them first:  systemctl --user disable --now "
                    + " ".join(u + ".service" for u in legacy))
            _run(["systemctl", "--user", "daemon-reload"], log)
            if not legacy and _run(["systemctl", "--user", "enable", "--now", UNIT_NAME], log):
                log("kllm is running (journalctl --user -u kllm -f). To keep it up after logout:  loginctl enable-linger $USER")
            else:
                log(f"then:  systemctl --user enable --now {UNIT_NAME}")
        return path
    if platform == "darwin":
        d = pathlib.Path(dest_dir or pathlib.Path.home() / "Library" / "LaunchAgents")
        path = d / f"{LAUNCHD_LABEL}.plist"
        if uninstall:
            _run(["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"], log)
            path.unlink(missing_ok=True); log(f"removed {path}"); return None
        data = launchd_plist(root, python, pkg_root)
        if print_only:
            print(data.decode()); return None
        d.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        log(f"wrote {path}")
        if start and shutil.which("launchctl"):
            _run(["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"], lambda *_: None)
            if not _run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], log):
                log(f"then:  launchctl load -w {path}")
        return path
    if platform in ("win32", "nt") or os.name == "nt":
        # Windows: register the at-logon task instead of only printing hints.
        return install_other(root=root, start=start, uninstall=uninstall,
                             print_only=print_only, log=log, python=python)
    log(manual_instructions(root))
    return None


def install_other(root, start=True, uninstall=False, print_only=False, log=print,
                  python=None):
    """Windows / other platforms: an at-logon Task Scheduler task (schtasks)."""
    root = pathlib.Path(root or common.ROOT)
    pkg_root = None if not common.is_checkout() else common.PKG_ROOT
    if uninstall:
        _run(["schtasks", "/Delete", "/TN", "kllm serve", "/F"], log)
        log("removed the 'kllm serve' logon task")
        return None
    cmd = windows_task_command(root, python, pkg_root)
    if print_only:
        print(" ".join(f'"{a}"' if " " in a else a for a in cmd))
        return None
    if start and shutil.which("schtasks"):
        install_windows_task(root=root, python=python, log=log)
    else:
        log("kllm serve (the tiny always-on relay) is what makes GPU backends start "
            "on first use and stop when idle — no manual `kllm up/down`.")
        log("to run it at every logon, run once (elevated if needed):")
        log("  " + " ".join(f'"{a}"' if " " in a else a for a in cmd))
        if start:
            log("then start it now with:  kllm serve")
    return None
