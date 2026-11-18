"""Tests for Subagent Spawner REST endpoints and CLI commands."""

import argparse
import asyncio
import io
import json
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from gateway import cli, relay, spawner
from gateway.spawner import FleetManager, SubagentStatus

LOCAL = ("127.0.0.1", 50000)


def test_api_spawn_subagent_sync(tmp_path):
    client = TestClient(relay.app, client=LOCAL)
    resp = client.post(
        "/admin/api/subagents/spawn",
        json={
            "harness": "shell",
            "task_instructions": "echo 'api sync test'",
            "cwd": str(tmp_path),
            "wait": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == SubagentStatus.COMPLETED.value
    assert data["subagent_summary"] == "api sync test"
    assert "subagent_id" in data


def test_api_spawn_subagent_async(tmp_path):
    client = TestClient(relay.app, client=LOCAL)
    resp = client.post(
        "/admin/api/subagents/spawn",
        json={
            "harness": "shell",
            "task_instructions": "sleep 5",
            "cwd": str(tmp_path),
            "wait": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == SubagentStatus.RUNNING.value
    sid = data["subagent_id"]

    # Query status
    status_resp = client.get(f"/admin/api/subagents/{sid}")
    assert status_resp.status_code == 200
    assert status_resp.json()["subagent_id"] == sid

    # Cancel
    cancel_resp = client.post(f"/admin/api/subagents/{sid}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["ok"] is True


def test_api_list_subagents(tmp_path):
    client = TestClient(relay.app, client=LOCAL)
    # Spawn one
    client.post(
        "/admin/api/subagents/spawn",
        json={
            "harness": "shell",
            "task_instructions": "echo list",
            "cwd": str(tmp_path),
            "wait": True,
        },
    )

    resp = client.get("/admin/api/subagents")
    assert resp.status_code == 200
    data = resp.json()
    assert "subagents" in data
    assert len(data["subagents"]) >= 1


def test_api_get_subagent_404():
    client = TestClient(relay.app, client=LOCAL)
    resp = client.get("/admin/api/subagents/nonexistent_subagent_id")
    assert resp.status_code == 404


def test_api_cancel_subagent_404():
    client = TestClient(relay.app, client=LOCAL)
    resp = client.post("/admin/api/subagents/nonexistent_subagent_id/cancel")
    assert resp.status_code == 404


def test_cli_subagents_list(capsys):
    # Test cmd_subagents listing
    args = argparse.Namespace(cancel=None, id=None, status=None)
    cli.cmd_subagents(args)
    out = capsys.readouterr().out
    assert "ID" in out or "No subagents registered." in out


def test_cli_subagents_detail(tmp_path, capsys):
    fleet = spawner.FLEET
    res = asyncio.run(fleet.spawn(
        harness="shell",
        task_instructions="echo 'cli detail'",
        cwd=str(tmp_path),
        wait=True,
    ))
    sid = res["subagent_id"]

    args = argparse.Namespace(cancel=None, id=sid, status=None)
    cli.cmd_subagents(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["subagent_id"] == sid
    assert data["status"] == SubagentStatus.COMPLETED.value


def test_cli_subagents_cancel(tmp_path, capsys):
    fleet = spawner.FLEET
    res = asyncio.run(fleet.spawn(
        harness="shell",
        task_instructions="sleep 10",
        cwd=str(tmp_path),
        wait=False,
    ))
    sid = res["subagent_id"]

    args = argparse.Namespace(cancel=sid, id=None, status=None)
    cli.cmd_subagents(args)
    out = capsys.readouterr().out
    assert f"Cancelled {sid}: success" in out


def test_cli_spawn_sync(tmp_path, capsys):
    args = argparse.Namespace(
        task="echo 'cli spawn test'",
        harness="shell",
        scope=[],
        verify=None,
        timeout=30,
        cwd=str(tmp_path),
        model=None,
        async_run=False,
    )
    cli.cmd_spawn(args)
    out = capsys.readouterr().out
    assert "COMPLETED" in out
    assert "Kaggle tokens consumed:" in out
    assert "Summary" in out


def test_cli_spawn_async(tmp_path, capsys):
    args = argparse.Namespace(
        task="sleep 5",
        harness="shell",
        scope=[],
        verify=None,
        timeout=30,
        cwd=str(tmp_path),
        model=None,
        async_run=True,
    )
    cli.cmd_spawn(args)
    out = capsys.readouterr().out
    assert "Subagent spawned in background:" in out
