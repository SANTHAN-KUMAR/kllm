"""`kllm shortcut`: a desktop / start-menu icon that starts kllm and opens the panel, like any desktop app.

Clicking the icon runs the same thing as typing `kllm`: start the relay in the background if it is not running,
then open the panel in the browser (the welcome page on a first run). No terminal window stays open.

  Windows  Desktop and Start menu shortcuts (.lnk) running pythonw -m gateway.cli setup (no console window)
  macOS    ~/Applications/kllm.app, a minimal app bundle (also visible in Launchpad and Spotlight)
  Linux    ~/.local/share/applications/kllm.desktop (app menus), plus a copy on ~/Desktop when it exists
"""
import os, pathlib, shutil, subprocess, sys

from . import common

NAME = "kllm"
COMMENT = "Start kllm and open its control panel"
ASSETS = common.PKG_DIR / "assets"


def _home():
    return pathlib.Path(os.path.expanduser("~"))


def _python_gui():
    """The interpreter that runs kllm, preferring pythonw on Windows (no console window)."""
    exe = pathlib.Path(sys.executable)
    if os.name == "nt":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return w
    return exe


def _env_pythonpath():
    """A checkout run without an install needs PKG_ROOT on sys.path; an installed package does not."""
    root = common.PKG_ROOT
    return str(root) if (root / "pyproject.toml").exists() and "site-packages" not in str(root).lower() else None


# ---- Windows ---------------------------------------------------------------------------------------------------------
def _win_dirs():
    desktop = _home() / "Desktop"
    try:                                     # OneDrive-redirected desktops: ask Windows where the desktop is
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                             capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL).stdout.strip()
        if out:
            desktop = pathlib.Path(out)
    except Exception:
        pass
    start = pathlib.Path(os.environ.get("APPDATA", str(_home() / "AppData" / "Roaming"))) / \
        "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return [desktop, start]


def _ps_quote(s):
    return "'" + str(s).replace("'", "''") + "'"


def _win_create(lnk, target, args, workdir, icon):
    ps = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(lnk)}); "
          f"$s.TargetPath = {_ps_quote(target)}; $s.Arguments = {_ps_quote(args)}; "
          f"$s.WorkingDirectory = {_ps_quote(workdir)}; $s.IconLocation = {_ps_quote(str(icon) + ',0')}; "
          f"$s.Description = {_ps_quote(COMMENT)}; $s.Save()")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not pathlib.Path(lnk).exists():
        raise RuntimeError((r.stderr or r.stdout or "PowerShell could not create the shortcut").strip()[:300])


# ---- macOS -----------------------------------------------------------------------------------------------------------
def _mac_app():
    return _home() / "Applications" / f"{NAME}.app"


def _mac_create(app):
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    (app / "Contents" / "Resources").mkdir(parents=True, exist_ok=True)
    pp = _env_pythonpath()
    launcher = macos / NAME
    launcher.write_text("#!/bin/sh\n"
                        + (f'export PYTHONPATH="{pp}${{PYTHONPATH:+:$PYTHONPATH}}"\n' if pp else "")
                        + f'exec "{sys.executable}" -m gateway.cli setup >/dev/null 2>&1\n', encoding="utf-8")
    launcher.chmod(0o755)
    (app / "Contents" / "Info.plist").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>{NAME}</string>
  <key>CFBundleDisplayName</key><string>{NAME}</string>
  <key>CFBundleIdentifier</key><string>dev.kllm.launcher</string>
  <key>CFBundleExecutable</key><string>{NAME}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSUIElement</key><true/>
</dict></plist>
""", encoding="utf-8")
    png = ASSETS / "kllm.png"
    if png.exists() and shutil.which("sips") and shutil.which("iconutil"):   # turn the PNG into an .icns
        iconset = app / "Contents" / "Resources" / "kllm.iconset"
        iconset.mkdir(exist_ok=True)
        for size in (16, 32, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                px = size * scale
                if px <= 512:
                    subprocess.run(["sips", "-z", str(px), str(px), str(png), "--out",
                                    str(iconset / f"icon_{size}x{size}{suffix}.png")], capture_output=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o",
                        str(app / "Contents" / "Resources" / "kllm.icns")], capture_output=True)
        shutil.rmtree(iconset, ignore_errors=True)
        plist = app / "Contents" / "Info.plist"
        plist.write_text(plist.read_text(encoding="utf-8").replace(
            "  <key>LSUIElement</key>", "  <key>CFBundleIconFile</key><string>kllm</string>\n  <key>LSUIElement</key>"),
            encoding="utf-8")


# ---- Linux -----------------------------------------------------------------------------------------------------------
def _linux_paths():
    apps = pathlib.Path(os.environ.get("XDG_DATA_HOME", str(_home() / ".local" / "share"))) / "applications"
    paths = [apps / f"{NAME}.desktop"]
    desk = _home() / "Desktop"
    if desk.is_dir():
        paths.append(desk / f"{NAME}.desktop")
    return paths


def _linux_entry():
    pp = _env_pythonpath()
    exec_line = f'"{sys.executable}" -m gateway.cli setup'
    if pp:
        exec_line = f'env PYTHONPATH="{pp}" ' + exec_line
    return ("[Desktop Entry]\nType=Application\n"
            f"Name={NAME}\nComment={COMMENT}\nExec={exec_line}\n"
            f"Icon={ASSETS / 'kllm.png'}\nTerminal=false\nCategories=Development;\n")


def _linux_create():
    made, entry = [], _linux_entry()
    for p in _linux_paths():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(entry, encoding="utf-8")
        p.chmod(0o755)
        made.append(p)
    if shutil.which("gio"):                      # GNOME: mark the desktop copy as trusted so it launches on click
        for p in made[1:]:
            subprocess.run(["gio", "set", str(p), "metadata::trusted", "true"], capture_output=True)
    return made


def _remove_paths(cands):
    gone = []
    for p in cands:
        if p.is_dir() and p.suffix == ".app" and (p / "Contents" / "Info.plist").exists() \
                and "dev.kllm.launcher" in (p / "Contents" / "Info.plist").read_text(encoding="utf-8", errors="ignore"):
            shutil.rmtree(p); gone.append(p)
        elif p.is_file() and _ours(p):
            p.unlink(); gone.append(p)
    return gone


# ---- public ----------------------------------------------------------------------------------------------------------
def create(log=print):
    """Create the shortcut(s) for this OS. Returns the list of paths written."""
    made = []
    if os.name == "nt":
        target, pp = _python_gui(), _env_pythonpath()
        args = "-m gateway.cli setup"
        workdir = str(common.PKG_ROOT if pp else _home())
        for d in _win_dirs():
            d.mkdir(parents=True, exist_ok=True)
            lnk = d / f"{NAME}.lnk"
            _win_create(lnk, target, args, workdir, ASSETS / "kllm.ico")
            made.append(lnk)
    elif sys.platform == "darwin":
        app = _mac_app()
        _mac_create(app)
        made.append(app)
    else:
        made = _linux_create()
    for p in made:
        log(f"kllm: created {p}")
    return made


def _ours(p):
    """True only for a shortcut kllm wrote: its command is `gateway.cli setup` (a .lnk stores it as UTF-16)."""
    try:
        data = p.read_bytes()
    except OSError:
        return False
    return b"gateway.cli setup" in data or "gateway.cli setup".encode("utf-16-le") in data


def remove(log=print):
    """Delete only the shortcuts `create` makes. Returns the paths removed."""
    if os.name == "nt":
        cands = [d / f"{NAME}.lnk" for d in _win_dirs()]
    elif sys.platform == "darwin":
        cands = [_mac_app()]
    else:
        cands = _linux_paths()
    gone = _remove_paths(cands)
    for p in gone:
        log(f"kllm: removed {p}")
    return gone
