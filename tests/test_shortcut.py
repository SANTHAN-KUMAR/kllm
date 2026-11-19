"""Desktop / app-menu shortcut: launches `gateway.cli setup` with the shipped icon, and removal only deletes ours."""
import os, sys

import pytest

from gateway import shortcut


def test_icons_ship_with_the_package():
    assert (shortcut.ASSETS / "kllm.ico").stat().st_size > 1000
    assert (shortcut.ASSETS / "kllm.png").stat().st_size > 1000


def test_linux_entry_launches_setup_without_a_terminal():
    e = shortcut._linux_entry()
    assert "Type=Application" in e and "Terminal=false" in e and "-m gateway.cli setup" in e
    assert "kllm.png" in e and "Name=kllm" in e


def test_linux_create_and_remove(monkeypatch, tmp_path):
    monkeypatch.setattr(shortcut, "_home", lambda: tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    (tmp_path / "Desktop").mkdir()
    made = shortcut._linux_create()
    assert {p.name for p in made} == {"kllm.desktop"} and len(made) == 2
    foreign = tmp_path / "Desktop" / "notes.desktop"
    foreign.write_text("[Desktop Entry]\nExec=gedit\n", encoding="utf-8")
    gone = shortcut._remove_paths(shortcut._linux_paths() + [foreign])
    assert len(gone) == 2 and foreign.exists()


def test_mac_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(shortcut, "_home", lambda: tmp_path)
    app = shortcut._mac_app()
    shortcut._mac_create(app)
    launcher = app / "Contents" / "MacOS" / "kllm"
    assert "-m gateway.cli setup" in launcher.read_text(encoding="utf-8")
    assert "dev.kllm.launcher" in (app / "Contents" / "Info.plist").read_text(encoding="utf-8")


def test_remove_never_touches_a_foreign_shortcut(tmp_path):
    other = tmp_path / "kllm.lnk"
    other.write_bytes(b"\x4c\x00\x00\x00" + "notepad.exe".encode("utf-16-le"))
    assert shortcut._ours(other) is False
    ours = tmp_path / "ours.lnk"
    ours.write_bytes(b"\x4c\x00\x00\x00" + "-m gateway.cli setup".encode("utf-16-le"))
    assert shortcut._ours(ours) is True
