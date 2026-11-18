import sys
"""Comprehensive unit and integration tests for Cross-Harness Subagent Spawner."""

import asyncio
import json
import os
import pathlib
import subprocess
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gateway import common, spawner
from gateway.spawner import (
    BaseHarnessDriver,
    ClaudeCodeDriver,
    FleetManager,
    OpenCodeDriver,
    ShellDriver,
    SubagentResult,
    SubagentSession,
    SubagentStatus,
    VerificationStatus,
    check_scope_violations,
    format_subagent_prompt,
    get_git_diff,
    get_git_status,
    run_verification_command,
)


PY = '"' + sys.executable.replace("\\", "/") + '"'   # POSIX sh and Git Bash; python3 is a Store stub on Windows

def test_format_subagent_prompt_basic():
    prompt = format_subagent_prompt("Refactor token verification")
    assert "TASK INSTRUCTIONS FROM TECH LEAD" in prompt
    assert "Refactor token verification" in prompt
    assert "MANDATORY KAGGLE ACCOUNT RESTRICTION" in prompt
    assert "kg" in prompt
    assert "lk*" in prompt
    assert "kx*" in prompt
    assert "STRICT SCOPE CONSTRAINTS" not in prompt
    assert "VERIFICATION REQUIREMENT" not in prompt


def test_format_subagent_prompt_full():
    prompt = format_subagent_prompt(
        task_instructions="Add Bearer support",
        permitted_files=["gateway/auth.py", "tests/test_gateway_key.py"],
        verification_command="pytest tests/test_gateway_key.py",
    )
    assert "TASK INSTRUCTIONS FROM TECH LEAD" in prompt
    assert "Add Bearer support" in prompt
    assert "MANDATORY KAGGLE ACCOUNT RESTRICTION" in prompt
    assert "kg" in prompt
    assert "STRICT SCOPE CONSTRAINTS" in prompt
    assert "- gateway/auth.py" in prompt
    assert "- tests/test_gateway_key.py" in prompt
    assert "VERIFICATION REQUIREMENT" in prompt
    assert "pytest tests/test_gateway_key.py" in prompt


def test_check_scope_violations():
    # Empty permitted -> no violations
    assert check_scope_violations(["gateway/auth.py"], permitted_files=None) == []
    assert check_scope_violations(["gateway/auth.py"], permitted_files=[]) == []

    # Within permitted
    permitted = ["gateway/auth.py", "tests/"]
    assert check_scope_violations(["gateway/auth.py"], permitted_files=permitted) == []
    assert check_scope_violations(["tests/test_gateway_key.py"], permitted_files=permitted) == []

    # Violations
    violations = check_scope_violations(
        ["gateway/auth.py", "gateway/relay.py", "config.yaml"],
        permitted_files=permitted,
    )
    assert violations == ["config.yaml", "gateway/relay.py"]


def test_git_diff_and_untracked_capture(tmp_path):
    # Initialize a temporary git repository
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.name", "TestUser"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)

    base_file = tmp_path / "hello.txt"
    base_file.write_text("initial content\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "hello.txt"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=tmp_path)

    # Initial baseline
    baseline = get_git_status(tmp_path)
    assert len(baseline["untracked"]) == 0

    # Modify base_file and create a new untracked file
    base_file.write_text("initial content\nnew line added\n", encoding="utf-8")
    new_file = tmp_path / "created.py"
    new_file.write_text("def foo():\n    return 42\n", encoding="utf-8")

    diff_str, modified_files = get_git_diff(tmp_path, baseline_untracked=baseline["untracked"])

    assert "hello.txt" in modified_files
    assert "created.py" in modified_files
    assert "+new line added" in diff_str
    assert "def foo():" in diff_str


def test_run_verification_command_success(tmp_path):
    async def _test():
        code, output = await run_verification_command("echo 'pytest: 10 passed'", cwd=tmp_path)
        assert code == 0
        assert "pytest: 10 passed" in output
    asyncio.run(_test())


def test_run_verification_command_failure(tmp_path):
    async def _test():
        code, output = await run_verification_command(f"{PY} -c 'import sys; sys.stderr.write(\"AssertionError\\n\"); sys.exit(1)'", cwd=tmp_path)
        assert code == 1
        assert "AssertionError" in output
    asyncio.run(_test())


def test_run_verification_command_timeout(tmp_path):
    async def _test():
        code, output = await run_verification_command("sleep 5", cwd=tmp_path, timeout=1)
        assert code == -1
        assert "timed out" in output
    asyncio.run(_test())


def test_shell_driver_execution(tmp_path):
    async def _test():
        driver = ShellDriver()
        exit_code, output, tokens = await driver.execute(
            subagent_id="subagent-test-1",
            prompt="echo 'shell execution test'",
            cwd=str(tmp_path),
            timeout=10,
        )
        assert exit_code == 0
        assert "shell execution test" in output
    asyncio.run(_test())


class _FakeStream:
    """Async-iterable stdout / readable stderr of a fake worker process (stream-json lines)."""
    def __init__(self, lines=(), err=b""):
        self._lines, self._err = list(lines), err
    def __aiter__(self):
        async def gen():
            for l in self._lines:
                yield l
        return gen()
    async def read(self):
        return self._err


def _fake_worker(events, returncode=0, err=b""):
    proc = AsyncMock()
    proc.stdout = _FakeStream([(json.dumps(e) + "\n").encode() for e in events])
    proc.stderr = _FakeStream(err=err)
    proc.returncode = returncode
    proc.pid = 4242
    async def _wait():
        return returncode
    proc.wait = _wait
    return proc


def test_claude_driver_mocked(tmp_path):
    async def _test():
        driver = ClaudeCodeDriver()
        events = [
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {"type": "assistant", "message": {"usage": {"input_tokens": 100, "output_tokens": 20}, "content": [
                {"type": "text", "text": "Reading auth.py"},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "auth.py"}}]}},
            {"type": "assistant", "message": {"usage": {"input_tokens": 5300, "output_tokens": 600}, "content": [
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "auth.py"}}]}},
            {"type": "result", "subtype": "success", "is_error": False,
             "result": "Successfully updated auth.py with Bearer token parsing.",
             "usage": {"input_tokens": 5400, "output_tokens": 620}},
        ]
        tracker = SubagentSession("subagent-claude-mock", "claude-code", "Update auth.py", cwd=str(tmp_path))
        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", return_value=_fake_worker(events)) as spawn:
                exit_code, summary, tokens = await driver.execute(
                    subagent_id="subagent-claude-mock", prompt="Update auth.py", cwd=str(tmp_path),
                    timeout=60, session_tracker=tracker)
        assert exit_code == 0
        assert "Successfully updated auth.py" in summary
        assert tokens == 6020
        # the worker gets a session id chosen by us (so it can be resumed even if killed) ...
        argv = list(spawn.call_args.args)
        assert "--session-id" in argv and argv[argv.index("--session-id") + 1] == tracker.session_id
        assert "stream-json" in argv
        # ... never inherits the MCP server's stdin (that pipe carries JSON-RPC) ...
        import asyncio as _a
        assert spawn.call_args.kwargs["stdin"] == _a.subprocess.DEVNULL
        # ... and its progress is visible while it runs
        assert tracker.progress["turns"] == 2 and tracker.progress["tool_calls"] == 2
        assert tracker.progress["last_tool"].startswith("Edit(")
    asyncio.run(_test())


def test_claude_driver_reports_api_error_in_final_event(tmp_path):
    async def _test():
        driver = ClaudeCodeDriver()
        events = [{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "API Error: overloaded",
                   "usage": {"input_tokens": 10, "output_tokens": 0}}]
        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", return_value=_fake_worker(events, returncode=0)):
                exit_code, summary, _ = await driver.execute(subagent_id="x", prompt="p", cwd=str(tmp_path), timeout=60)
        assert exit_code != 0, "a worker whose final event is an error must not be reported as success"
        assert "overloaded" in summary
    asyncio.run(_test())


def test_claude_driver_missing_exe(tmp_path):
    async def _test():
        driver = ClaudeCodeDriver()
        with patch("shutil.which", return_value=None):
            with patch.object(pathlib.Path, "exists", return_value=False):
                with pytest.raises(FileNotFoundError, match="Claude Code executable"):
                    await driver.execute(
                        subagent_id="subagent-claude-fail",
                        prompt="dummy",
                        cwd=str(tmp_path),
                        timeout=10,
                    )
    asyncio.run(_test())


def test_fleet_manager_sync_spawn(tmp_path):
    async def _test():
        subprocess.check_call(["git", "init"], cwd=tmp_path)
        subprocess.check_call(["git", "config", "user.name", "Tester"], cwd=tmp_path)
        subprocess.check_call(["git", "config", "user.email", "tester@test.com"], cwd=tmp_path)

        test_file = tmp_path / "sample.py"
        test_file.write_text("# initial\n", encoding="utf-8")
        subprocess.check_call(["git", "add", "sample.py"], cwd=tmp_path)
        subprocess.check_call(["git", "commit", "-m", "init"], cwd=tmp_path)

        fleet = FleetManager()
        cmd = 'echo "x = 10" >> sample.py && echo "done" > notes.txt'

        res = await fleet.spawn(
            harness="shell",
            task_instructions=cmd,
            permitted_files=["sample.py", "notes.txt"],
            verification_command=f'{PY} -c "import sample; assert sample.x == 10"',
            cwd=str(tmp_path),
            wait=True,
        )

        assert res["status"] == SubagentStatus.COMPLETED.value
        assert res["verification_status"] == VerificationStatus.PASSED.value
        assert "sample.py" in res["files_modified"]
        assert "notes.txt" in res["files_modified"]
        assert res["scope_violations"] == []
        assert "+x = 10" in res["git_diff"]
        assert res["cloud_tokens_consumed"] == 0
        assert res["kaggle_tokens_consumed"] > 0
    asyncio.run(_test())


def test_fleet_manager_scope_violation(tmp_path):
    async def _test():
        subprocess.check_call(["git", "init"], cwd=tmp_path)
        subprocess.check_call(["git", "config", "user.name", "Tester"], cwd=tmp_path)
        subprocess.check_call(["git", "config", "user.email", "tester@test.com"], cwd=tmp_path)

        allowed = tmp_path / "allowed.py"
        allowed.write_text("# allowed\n", encoding="utf-8")
        subprocess.check_call(["git", "add", "allowed.py"], cwd=tmp_path)
        subprocess.check_call(["git", "commit", "-m", "init"], cwd=tmp_path)

        fleet = FleetManager()
        cmd = 'echo "a = 1" >> allowed.py && echo "f = 2" > forbidden.py'

        res = await fleet.spawn(
            harness="shell",
            task_instructions=cmd,
            permitted_files=["allowed.py"],
            cwd=str(tmp_path),
            wait=True,
        )

        assert res["status"] == SubagentStatus.NEEDS_REVIEW.value
        assert "forbidden.py" in res["scope_violations"]
        assert "allowed.py" not in res["scope_violations"]
    asyncio.run(_test())


def test_fleet_manager_verification_failure(tmp_path):
    async def _test():
        subprocess.check_call(["git", "init"], cwd=tmp_path)
        subprocess.check_call(["git", "config", "user.name", "Tester"], cwd=tmp_path)
        subprocess.check_call(["git", "config", "user.email", "tester@test.com"], cwd=tmp_path)

        fleet = FleetManager()
        res = await fleet.spawn(
            harness="shell",
            task_instructions='echo "testing failure"',
            verification_command=f'{PY} -c "raise ValueError(\'test broke\')"',
            cwd=str(tmp_path),
            wait=True,
        )

        assert res["status"] == SubagentStatus.FAILED.value
        assert res["verification_status"] == VerificationStatus.FAILED.value
        assert "ValueError: test broke" in res["verification_output"]
    asyncio.run(_test())


def test_fleet_manager_async_spawn_and_cancel(tmp_path):
    async def _test():
        fleet = FleetManager()
        res = await fleet.spawn(
            harness="shell",
            task_instructions="sleep 10",
            cwd=str(tmp_path),
            wait=False,
        )

        subagent_id = res["subagent_id"]
        assert res["status"] == SubagentStatus.RUNNING.value

        info = fleet.get(subagent_id)
        assert info is not None
        assert info["subagent_id"] == subagent_id

        assert fleet.cancel(subagent_id) is True
        cancelled_info = fleet.get(subagent_id)
        assert cancelled_info["status"] == SubagentStatus.CANCELLED.value
    asyncio.run(_test())


def test_fleet_manager_persistence(tmp_path):
    async def _test():
        fleet1 = FleetManager()
        custom_state_file = tmp_path / "subagents.json"

        with patch.object(FleetManager, "_state_file", new_callable=lambda: custom_state_file):
            res = await fleet1.spawn(
                harness="shell",
                task_instructions="echo 'persisted task'",
                cwd=str(tmp_path),
                wait=True,
            )
            sid = res["subagent_id"]

            fleet2 = FleetManager()
            item = fleet2.get(sid)
            assert item is not None
            assert item["subagent_id"] == sid
            assert item["status"] == SubagentStatus.COMPLETED.value
            assert item["subagent_summary"] == "persisted task"

            all_items = fleet2.list_all()
            assert any(x["subagent_id"] == sid for x in all_items)
    asyncio.run(_test())


def test_check_scope_violations_absolute_and_globs(tmp_path):
    cwd = tmp_path
    abs_file = str(tmp_path / "gateway" / "auth.py")
    abs_dir = str(tmp_path / "tests")

    # Absolute permitted path matching relative modified file
    assert check_scope_violations(["gateway/auth.py"], permitted_files=[abs_file], cwd=cwd) == []

    # Relative permitted path matching absolute modified file
    assert check_scope_violations([abs_file], permitted_files=["gateway/auth.py"], cwd=cwd) == []

    # Absolute directory permitted
    assert check_scope_violations(["tests/test_foo.py"], permitted_files=[abs_dir], cwd=cwd) == []

    # Glob matching
    assert check_scope_violations(["gateway/auth.py", "gateway/spawner.py"], permitted_files=["gateway/*.py"], cwd=cwd) == []
    assert check_scope_violations(["tests/unit/test_one.py"], permitted_files=["tests/**/*.py"], cwd=cwd) == []

    # Glob violation
    v = check_scope_violations(["gateway/auth.py", "config.yaml"], permitted_files=["gateway/*.py"], cwd=cwd)
    assert v == ["config.yaml"]


def test_git_diff_in_empty_repo_without_commits(tmp_path):
    # Repo with git init but zero commits
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    f = tmp_path / "new_file.txt"
    f.write_text("hello world\n", encoding="utf-8")

    diff_str, modified = get_git_diff(tmp_path)
    assert "new_file.txt" in modified
    assert "+hello world" in diff_str


def test_git_diff_filters_untouched_baseline_dirty_files(tmp_path):
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.name", "Tester"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.email", "tester@test.com"], cwd=tmp_path)

    # Initial commit
    dirty_preexisting = tmp_path / "dirty.txt"
    dirty_preexisting.write_text("line 1\n", encoding="utf-8")
    subagent_target = tmp_path / "target.txt"
    subagent_target.write_text("target 1\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "."], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=tmp_path)

    # User modifies dirty.txt BEFORE subagent starts
    dirty_preexisting.write_text("line 1\nuser edit\n", encoding="utf-8")
    baseline = get_git_status(tmp_path)
    assert "dirty.txt" in baseline["modified"]

    # Sleep slightly so mtime is strictly greater for subagent edits
    time.sleep(0.05)

    # Subagent only modifies target.txt
    subagent_target.write_text("target 1\nsubagent edit\n", encoding="utf-8")

    diff_str, modified = get_git_diff(
        cwd=tmp_path,
        baseline_untracked=baseline["untracked"],
        baseline_modified=baseline["modified"],
        baseline_mtimes=baseline["mtimes"],
    )

    assert "target.txt" in modified
    assert "dirty.txt" not in modified
    assert "+subagent edit" in diff_str
    assert "user edit" not in diff_str


def test_subagent_timeout_status(tmp_path):
    async def _test():
        fleet = FleetManager()
        res = await fleet.spawn(
            harness="shell",
            task_instructions="sleep 5",
            timeout_seconds=1,
            cwd=str(tmp_path),
            wait=True,
        )
        assert res["status"] == SubagentStatus.TIMEOUT.value
        assert res["exit_code"] == -1
        assert "timed out after 1 seconds" in res["raw_output"].lower()
    asyncio.run(_test())


def test_preexisting_untracked_file_modification(tmp_path):
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.name", "Tester"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.email", "tester@test.com"], cwd=tmp_path)

    # Initial committed file
    f_init = tmp_path / "init.txt"
    f_init.write_text("initial\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "init.txt"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=tmp_path)

    # Pre-existing untracked file BEFORE subagent runs
    untracked_pre = tmp_path / "untracked_notes.txt"
    untracked_pre.write_text("line 1\nline 2\n", encoding="utf-8")

    baseline = get_git_status(tmp_path)
    assert "untracked_notes.txt" in baseline["untracked"]

    time.sleep(0.05)
    # Subagent modifies the pre-existing untracked file
    untracked_pre.write_text("line 1\nline 2\nsubagent added line\n", encoding="utf-8")

    diff_str, modified = get_git_diff(
        cwd=tmp_path,
        baseline_untracked=baseline["untracked"],
        baseline_modified=baseline["modified"],
        baseline_mtimes=baseline["mtimes"],
        baseline_untracked_contents=baseline.get("untracked_contents", {}),
    )

    assert "untracked_notes.txt" in modified
    assert "+subagent added line" in diff_str


