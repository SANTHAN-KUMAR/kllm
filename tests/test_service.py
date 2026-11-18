"""Generated service files (systemd unit, launchd plist) and install-service file placement."""
import configparser, plistlib, pathlib
from gateway import service


def parse_unit(text):
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read_string(text)
    return cp


def test_systemd_unit_runs_one_serve_process():
    u = service.systemd_unit("/home/me/.config/kllm", python="/home/me/.local/share/pipx/venvs/kllm/bin/python",
                             home="/home/me")
    cp = parse_unit(u)
    assert cp["Service"]["ExecStart"] == "/home/me/.local/share/pipx/venvs/kllm/bin/python -m gateway.cli serve"
    assert cp["Service"]["Restart"] == "always" and cp["Service"]["Type"] == "simple"
    assert cp["Service"]["WorkingDirectory"] == "/home/me/.config/kllm"
    assert "KLLM_ROOT=/home/me/.config/kllm" in u
    assert "/home/me/.local/bin" in u                     # PATH has the legacy kgN wrappers + user bin
    assert cp["Install"]["WantedBy"] == "default.target"
    assert "PYTHONPATH" not in u                          # installed package: no path hacks
    assert "relay" not in cp["Service"]["ExecStart"] and "watch" not in cp["Service"]["ExecStart"]


def test_systemd_unit_checkout_sets_pythonpath_and_quotes_spaces():
    u = service.systemd_unit("/run/media/me/New Volume/kllm", python="/run/media/me/New Volume/kllm/.venv/bin/python",
                             pkg_root="/run/media/me/New Volume/kllm", home="/home/me")
    assert 'Environment="KLLM_ROOT=/run/media/me/New Volume/kllm"' in u
    assert 'Environment="PYTHONPATH=/run/media/me/New Volume/kllm"' in u
    assert 'ExecStart="/run/media/me/New Volume/kllm/.venv/bin/python" -m gateway.cli serve' in u
    assert 'WorkingDirectory=/run/media/me/New Volume/kllm' in u


def test_systemd_percent_is_escaped():
    assert "100%%" in service.systemd_unit("/tmp/100%", python="/usr/bin/python3", home="/h")


def test_launchd_plist():
    d = plistlib.loads(service.launchd_plist("/Users/me/.config/kllm", python="/opt/venv/bin/python", home="/Users/me"))
    assert d["Label"] == "com.kllm.serve"
    assert d["ProgramArguments"] == ["/opt/venv/bin/python", "-m", "gateway.cli", "serve"]
    assert d["EnvironmentVariables"]["KLLM_ROOT"] == "/Users/me/.config/kllm"
    assert "/Users/me/.local/bin" in d["EnvironmentVariables"]["PATH"]
    assert d["RunAtLoad"] is True and d["KeepAlive"] is True
    assert d["StandardOutPath"].endswith("logs/serve.log")


def test_install_linux_writes_unit_only_when_start_false(tmp_path):
    root = tmp_path / "root"
    path = service.install(platform="linux", dest_dir=tmp_path / "units", start=False, root=root,
                           python="/usr/bin/python3", log=lambda *a: None)
    assert path == tmp_path / "units" / "kllm.service"
    assert "ExecStart=/usr/bin/python3 -m gateway.cli serve" in path.read_text(encoding="utf-8")


def test_install_macos_writes_plist(tmp_path):
    path = service.install(platform="darwin", dest_dir=tmp_path / "LA", start=False, root=tmp_path / "r",
                           python="/usr/bin/python3", log=lambda *a: None)
    assert path.name == "com.kllm.serve.plist"
    assert plistlib.loads(path.read_bytes())["ProgramArguments"][-1] == "serve"
    assert (tmp_path / "r" / "logs").is_dir()


def test_install_other_platform_prints_manual_instructions(tmp_path, monkeypatch):
    # no schtasks here (e.g. CI on linux): prints the exact command to run once
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    out = []
    assert service.install(platform="win32", root=tmp_path, log=out.append) is None
    assert any("schtasks" in line for line in out)


def test_windows_task_command_starts_serve_at_logon(tmp_path):
    cmd = service.windows_task_command(tmp_path, python="/usr/bin/python3")
    assert cmd[:3] == ["schtasks", "/Create", "/TN"]
    assert "kllm serve" in cmd
    assert "/SC" in cmd and "ONLOGON" in cmd
    assert str(tmp_path) in cmd[-2] or any(str(tmp_path) in a for a in cmd)


def test_windows_task_xml_is_hidden_and_windowless(tmp_path):
    import xml.etree.ElementTree as ET
    xml = service.windows_task_xml(tmp_path, python="C:\\Python\\python.exe")
    root = ET.fromstring(xml)
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert root.find("t:Settings/t:Hidden", ns).text == "true"
    assert root.find("t:Settings/t:DisallowStartIfOnBatteries", ns).text == "false"
    assert root.find("t:Triggers/t:LogonTrigger", ns) is not None
    action = root.find("t:Actions/t:Exec", ns)
    assert action.find("t:Command", ns).text == "C:\\Python\\python.exe"
    assert "serve" in action.find("t:Arguments", ns).text
    assert "cmd" not in action.find("t:Command", ns).text
    assert action.find("t:WorkingDirectory", ns).text == str(tmp_path)


def test_print_only_writes_nothing(tmp_path, capsys):
    assert service.install(platform="linux", dest_dir=tmp_path / "u", print_only=True, root=tmp_path,
                           python="/usr/bin/python3") is None
    assert "ExecStart" in capsys.readouterr().out and not (tmp_path / "u").exists()
