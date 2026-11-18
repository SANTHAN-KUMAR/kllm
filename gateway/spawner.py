"""
Subagent Spawner Engine for Cross-Harness Subagent Spawning (CHSS).

Coordinates worker subagents inside target harnesses (Claude Code CLI, OpenCode, Shell)
powered 100% by local/Kaggle GPU LLMs via the local KLLM relay (http://127.0.0.1:8080).

Main Agent (Frontier Cloud LLM / Tech Lead) dispatches scoped work orders.
Subagents perform heavy code exploration, file editing, and test runs for free,
returning structured deliverable packets (git diff, test output, status) back to the caller.
"""

from __future__ import annotations

import asyncio
import difflib
import enum
import fnmatch
import json
import logging
import os
import pathlib
import posixpath
import re
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import common

logger = logging.getLogger("kllm.spawner")


class SubagentStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class VerificationStatus(str, enum.Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class SubagentResult:
    subagent_id: str
    harness: str
    status: str = SubagentStatus.PENDING.value
    execution_time_seconds: float = 0.0
    kaggle_tokens_consumed: int = 0
    cloud_tokens_consumed: int = 0
    files_modified: List[str] = field(default_factory=list)
    git_diff: str = ""
    verification_output: str = ""
    verification_status: str = VerificationStatus.SKIPPED.value
    subagent_summary: str = ""
    scope_violations: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    raw_output: str = ""
    claude_session_id: Optional[str] = None   # worker session id: pass as resume_session_id to continue after an interruption

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SubagentSpawnRequest(BaseModel):
    harness: str = Field(default="claude-code", description="Target harness: claude-code, claude-kllm, opencode, shell, or agy")
    task_instructions: str = Field(..., description="Detailed instructions for the worker subagent")
    permitted_files: Optional[List[str]] = Field(default=None, description="Optional list of permitted files/paths")
    verification_command: Optional[str] = Field(default=None, description="Optional command to run to verify changes")
    timeout_seconds: int = Field(default=600, ge=5, le=3600, description="Max execution timeout in seconds")
    cwd: Optional[str] = Field(default=None, description="Working directory (defaults to workspace root)")
    model: Optional[str] = Field(default=None, description="Model key override")
    wait: bool = Field(default=True, description="Whether to wait for completion or return immediately")
    resume_session_id: Optional[str] = Field(default=None, description="Optional Claude Code session ID to resume")
    open_terminal: bool = Field(default=False, description="Whether to launch in a visible desktop terminal window")


# ---- Git & State Extraction Helpers ------------------------------------------

def get_git_status(cwd: str | pathlib.Path) -> Dict[str, Any]:
    """Capture snapshot of git status: modified files, untracked files, current HEAD, and file mtimes."""
    cwd_path = str(cwd or common.ROOT)
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd_path,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        rev = ""

    untracked: Set[str] = set()
    modified: Set[str] = set()

    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd_path,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        for line in out.splitlines():
            line = line.rstrip()
            if not line:
                continue
            status_code = line[:2]
            filepath = line[3:].strip()
            # If path has quotes or arrow (rename), handle clean name
            if " -> " in filepath:
                filepath = filepath.split(" -> ")[1].strip()
            filepath = filepath.strip('"')

            if status_code == "??":
                untracked.add(filepath)
            else:
                modified.add(filepath)
    except Exception:
        pass

    mtimes: Dict[str, float] = {}
    untracked_contents: Dict[str, str] = {}
    for f in modified | untracked:
        fp = pathlib.Path(cwd_path) / f
        try:
            if fp.exists():
                mtimes[f] = fp.stat().st_mtime
                if f in untracked and fp.is_file() and fp.stat().st_size < 1_000_000:
                    untracked_contents[f] = fp.read_text(errors="replace", encoding="utf-8")
        except Exception:
            pass

    return {
        "head": rev,
        "untracked": untracked,
        "modified": modified,
        "mtimes": mtimes,
        "untracked_contents": untracked_contents,
    }


def get_git_diff(
    cwd: str | pathlib.Path,
    baseline_untracked: Optional[Set[str]] = None,
    baseline_modified: Optional[Set[str]] = None,
    baseline_mtimes: Optional[Dict[str, float]] = None,
    baseline_untracked_contents: Optional[Dict[str, str]] = None,
    max_bytes: int = 150_000,
) -> Tuple[str, List[str]]:
    """
    Extract unified git diff comparing working tree against baseline.
    Also captures newly created untracked files as added diff chunks.
    Filters out pre-existing modified files that were untouched during the run.
    Returns (unified_diff_str, list_of_modified_file_paths).
    """
    cwd_path = pathlib.Path(cwd or common.ROOT).resolve()
    baseline_untracked = baseline_untracked or set()
    baseline_modified = baseline_modified or set()
    baseline_mtimes = baseline_mtimes or {}

    modified_files: List[str] = []
    diff_chunks: List[str] = []

    # 1. Detect candidate tracked modified files
    tracked_candidates: List[str] = []
    try:
        name_out = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=cwd_path,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        try:
            name_out = subprocess.check_output(
                ["git", "diff", "--name-only"],
                cwd=cwd_path,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except Exception:
            name_out = ""

    for line in name_out.splitlines():
        line = line.strip()
        if line and line not in tracked_candidates:
            tracked_candidates.append(line)

    # Also check git status modified
    current_status = get_git_status(cwd_path)
    for line in current_status["modified"]:
        if line not in tracked_candidates:
            tracked_candidates.append(line)

    # Filter candidate files: if file was already modified at baseline, check if its mtime changed
    for f in tracked_candidates:
        full_f = cwd_path / f
        if f in baseline_modified and full_f.exists():
            try:
                curr_mtime = full_f.stat().st_mtime
                base_mtime = baseline_mtimes.get(f)
                if base_mtime is not None and abs(curr_mtime - base_mtime) < 1e-4:
                    continue
            except Exception:
                pass
        if f not in modified_files:
            modified_files.append(f)

    # 2. Extract git diff for modified tracked files
    tracked_diff = ""
    if modified_files:
        try:
            diff_cmd = ["git", "diff", "HEAD", "--"] + modified_files
            tracked_diff = subprocess.check_output(
                diff_cmd,
                cwd=cwd_path,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
        except Exception:
            try:
                diff_cmd = ["git", "diff", "--"] + modified_files
                tracked_diff = subprocess.check_output(
                    diff_cmd,
                    cwd=cwd_path,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=10,
                )
            except Exception:
                tracked_diff = ""

    if tracked_diff and tracked_diff.strip():
        diff_chunks.append(tracked_diff.strip())

    # 3. Find newly untracked files created during this run
    new_untracked = current_status["untracked"] - baseline_untracked

    for file_rel in sorted(new_untracked):
        full_path = cwd_path / file_rel
        if full_path.is_file():
            if file_rel not in modified_files:
                modified_files.append(file_rel)
            try:
                # Use git diff --no-index against /dev/null with relative path
                diff_cmd = ["git", "diff", "--no-index", "/dev/null", file_rel]
                p = subprocess.run(diff_cmd, cwd=cwd_path, capture_output=True, text=True, timeout=5)
                chunk = p.stdout.strip()
                if chunk:
                    diff_chunks.append(chunk)
                else:
                    content = full_path.read_text(errors="replace", encoding="utf-8")
                    lines = content.splitlines()
                    diff_chunks.append(
                        f"diff --git a/{file_rel} b/{file_rel}\n"
                        f"new file mode 100644\n"
                        f"--- /dev/null\n"
                        f"+++ b/{file_rel}\n"
                        f"@@ -0,0 +1,{len(lines)} @@\n" +
                        "\n".join("+" + l for l in lines)
                    )
            except Exception:
                pass

    # 4. Find pre-existing untracked files modified or deleted during this run
    baseline_untracked_contents = baseline_untracked_contents or {}
    preexisting_untracked = current_status["untracked"] & baseline_untracked
    for file_rel in sorted(preexisting_untracked):
        full_path = cwd_path / file_rel
        if not full_path.is_file():
            continue
        try:
            curr_mtime = full_path.stat().st_mtime
            base_mtime = baseline_mtimes.get(file_rel)
            if base_mtime is not None and abs(curr_mtime - base_mtime) < 1e-4:
                continue
            curr_content = full_path.read_text(errors="replace", encoding="utf-8")
            base_content = baseline_untracked_contents.get(file_rel)
            if base_content is not None and curr_content == base_content:
                continue

            if file_rel not in modified_files:
                modified_files.append(file_rel)

            if base_content is not None:
                diff = "".join(
                    difflib.unified_diff(
                        base_content.splitlines(keepends=True),
                        curr_content.splitlines(keepends=True),
                        fromfile=f"a/{file_rel}",
                        tofile=f"b/{file_rel}",
                    )
                )
                if diff.strip():
                    diff_chunks.append(diff.strip())
            else:
                diff_chunks.append(f"# Modified untracked file: {file_rel}")
        except Exception:
            pass

    deleted_untracked = baseline_untracked - current_status["untracked"]
    for file_rel in sorted(deleted_untracked):
        full_path = cwd_path / file_rel
        if not full_path.exists():
            if file_rel not in modified_files:
                modified_files.append(file_rel)
            base_content = baseline_untracked_contents.get(file_rel)
            if base_content is not None:
                lines = base_content.splitlines()
                diff_chunks.append(
                    f"diff --git a/{file_rel} b/{file_rel}\n"
                    f"deleted file mode 100644\n"
                    f"--- a/{file_rel}\n"
                    f"+++ /dev/null\n"
                    f"@@ -1,{len(lines)} +0,0 @@\n" +
                    "\n".join("-" + l for l in lines)
                )

    combined_diff = "\n\n".join(chunk for chunk in diff_chunks if chunk).strip()

    if len(combined_diff.encode("utf-8")) > max_bytes:
        truncated_msg = f"\n\n[... Diff truncated: {len(combined_diff)} bytes exceeds limit of {max_bytes} bytes ...]"
        combined_diff = combined_diff[:max_bytes] + truncated_msg

    return combined_diff, sorted(list(set(modified_files)))


def check_scope_violations(
    modified_files: Optional[List[str]] = None,
    permitted_files: Optional[List[str]] = None,
    cwd: Optional[str | pathlib.Path] = None,
    files_modified: Optional[List[str]] = None,
) -> List[str]:
    """
    Check if any modified files fall outside the permitted scope.
    Paths are checked relative to cwd. Supports exact matches, directory prefixes, and glob patterns.
    """
    files = files_modified if files_modified is not None else (modified_files or [])
    if not permitted_files:
        return []

    cwd_path = pathlib.Path(cwd or common.ROOT).resolve()

    def _normalize(p: str) -> str:
        # always forward slashes, so results and patterns compare the same on Windows and POSIX
        s = str(p).strip()
        path = pathlib.Path(s)
        if path.is_absolute():
            try:
                s = str(path.relative_to(cwd_path))
            except ValueError:
                pass
        return posixpath.normpath(s.replace("\\", "/"))

    normalized_permitted = [_normalize(p) for p in permitted_files]

    violations = []
    for f in files:
        clean_f = _normalize(f)
        match = False
        for perm in normalized_permitted:
            if clean_f == perm:
                match = True
                break
            # Directory prefix
            if clean_f.startswith(perm.rstrip("/") + "/") or clean_f.startswith(perm.rstrip("\\") + "\\"):
                match = True
                break
            # Glob pattern match (e.g. *.py, tests/**, etc.)
            if fnmatch.fnmatch(clean_f, perm):
                match = True
                break
        if not match:
            violations.append(clean_f)

    return sorted(violations)


def _posix_shell() -> Optional[str]:
    """On Windows, Git for Windows' bash (never WSL's System32 bash): task and verification commands are POSIX sh."""
    if os.name != "nt":
        return None
    git = shutil.which("git")
    cands = []
    if git:
        g = pathlib.Path(git).resolve()
        cands += [g.parent.parent / "bin" / "bash.exe", g.parent.parent / "usr" / "bin" / "bash.exe",
                  g.parent.parent.parent / "bin" / "bash.exe"]
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"), os.environ.get("LOCALAPPDATA")):
        if base:
            cands += [pathlib.Path(base) / "Git" / "bin" / "bash.exe", pathlib.Path(base) / "Programs" / "Git" / "bin" / "bash.exe"]
    for c in cands:
        if c.is_file():
            return str(c)
    return None


async def _shell(cmd: str, **kw):
    """create_subprocess_shell, but through POSIX bash on Windows when Git Bash is installed (cmd.exe otherwise)."""
    bash = _posix_shell()
    if bash:
        kw.pop("start_new_session", None)
        return await asyncio.create_subprocess_exec(bash, "-c", cmd, **kw)
    return await asyncio.create_subprocess_shell(cmd, **kw)


async def run_verification_command(
    cmd: str,
    cwd: str | pathlib.Path,
    timeout: int = 180,
) -> Tuple[int, str]:
    """Run verification command asynchronously and capture combined stdout/stderr."""
    if not cmd or not cmd.strip():
        return 0, ""

    cwd_path = str(pathlib.Path(cwd or common.ROOT).resolve())
    try:
        proc = await _shell(
            cmd,
            cwd=cwd_path,
            stdin=asyncio.subprocess.DEVNULL,   # never inherit the MCP server's stdin (the JSON-RPC pipe)
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout_data, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout_data.decode("utf-8", errors="replace").strip()
            return proc.returncode or 0, output
        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return -1, f"Verification command timed out after {timeout} seconds."
    except Exception as e:
        return -1, f"Error executing verification command: {e}"


def count_tokens_from_log(start_time: float, start_offset: int = 0) -> int:
    """
    Read newly logged requests from logs/requests.jsonl and sum tokens processed.
    Total = new_in + cached_in + out.
    """
    log_file = common.LOG_DIR / "requests.jsonl"
    if not log_file.exists():
        return 0

    total_tokens = 0
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            if start_offset > 0:
                f.seek(start_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("ts", 0) >= (start_time - 1.0):
                        total_tokens += int(record.get("new_in", 0))
                        total_tokens += int(record.get("cached_in", 0))
                        total_tokens += int(record.get("out", 0))
                except Exception:
                    continue
    except Exception:
        pass
    return total_tokens


def format_subagent_prompt(
    task_instructions: str,
    permitted_files: Optional[List[str]] = None,
    verification_command: Optional[str] = None,
) -> str:
    """Construct an explicit execution lease prompt for the worker subagent."""
    parts = [
        "# TASK INSTRUCTIONS FROM TECH LEAD (MAIN AGENT)",
        task_instructions.strip(),
        "",
    ]

    if permitted_files:
        parts.append("## STRICT SCOPE CONSTRAINTS")
        parts.append("You are strictly PERMITTED to inspect and modify ONLY these files/directories:")
        for pf in permitted_files:
            parts.append(f"- {pf}")
        parts.append("DO NOT modify or delete any files outside this permitted list.")
        parts.append("")

    if verification_command:
        parts.append("## VERIFICATION REQUIREMENT")
        parts.append("Before finishing, you MUST run this verification command in the project directory:")
        parts.append(f"```bash\n{verification_command}\n```")
        parts.append("Ensure all tests pass cleanly without errors.")
        parts.append("")

    parts.append("## MANDATORY KAGGLE ACCOUNT RESTRICTION")
    parts.append("Use ONLY Kaggle accounts that follow the 'kg' naming convention (such as kg1, kg2, kg3, ..., kg22).")
    parts.append("DO NOT touch, launch, or use any 'lk*' accounts or 'kx*' accounts under any circumstances.")
    parts.append("All operations, tests, and kernel executions must strictly be confined to 'kg*' accounts only.")
    parts.append("DO NOT delete or purge any accounts from config.yaml: All accounts (including lk*, kx*, etc.) in config.yaml must remain permanently in config.yaml. Operational restriction to kg* accounts is enforced via account_allow_prefixes, NEVER by deleting accounts from config.yaml.")
    parts.append("")

    parts.append("## DELIVERABLE SUMMARY")
    parts.append("When done, provide a concise summary of the changes you made and the verification outcome.")

    return "\n".join(parts)


def _progress_view(pr: Dict[str, Any]) -> Dict[str, Any]:
    """Live worker progress for get_subagent_status: lets a supervisor tell 'working' from 'stalled'."""
    if not pr:
        return {}
    v = {k: pr[k] for k in ("turns", "tool_calls", "last_tool", "last_text", "events", "input_tokens", "output_tokens") if k in pr}
    if pr.get("last_event_at"):
        v["idle_seconds"] = round(time.time() - pr["last_event_at"], 1)
    return v


def _track_event(pr: Dict[str, Any], ev: Dict[str, Any]) -> None:
    """Update progress from one Claude Code stream-json event."""
    pr["events"] = pr.get("events", 0) + 1
    pr["last_event_at"] = time.time()
    t = ev.get("type")
    if t == "assistant":
        pr["turns"] = pr.get("turns", 0) + 1
        msg = ev.get("message") or {}
        u = msg.get("usage") or {}
        pr["input_tokens"] = pr.get("input_tokens", 0) + int(u.get("input_tokens") or 0)
        pr["output_tokens"] = pr.get("output_tokens", 0) + int(u.get("output_tokens") or 0)
        for c in msg.get("content") or []:
            if c.get("type") == "tool_use":
                pr["tool_calls"] = pr.get("tool_calls", 0) + 1
                inp = c.get("input") or {}
                arg = inp.get("file_path") or inp.get("command") or inp.get("pattern") or inp.get("path") or ""
                pr["last_tool"] = f"{c.get('name')}({str(arg)[:80]})"
            elif c.get("type") == "text" and c.get("text"):
                pr["last_text"] = c["text"][:200]


# ---- Harness Drivers ---------------------------------------------------------

class BaseHarnessDriver:
    """Abstract driver interface for executing subagents inside a harness."""

    async def execute(
        self,
        subagent_id: str,
        prompt: str,
        cwd: str,
        timeout: int,
        model: Optional[str] = None,
        session_tracker: Optional[SubagentSession] = None,
    ) -> Tuple[int, str, int]:
        """
        Execute subagent task in the harness.
        Returns: (exit_code, raw_output, tokens_consumed_estimate)
        """
        raise NotImplementedError


class ClaudeCodeDriver(BaseHarnessDriver):
    """
    Executes a subagent inside headless Claude Code (`claude -p ...`).
    Configured with ANTHROPIC_BASE_URL pointing to local KLLM relay.
    """

    def __init__(self, relay_url: Optional[str] = None):
        self.relay_url = (relay_url or self._detect_relay()).rstrip("/")

    def _detect_relay(self) -> str:
        try:
            cfg = common.load_config()
            port = cfg.get("relay_port", 8080)
            return f"http://127.0.0.1:{port}"
        except Exception:
            return "http://127.0.0.1:8080"

    def _resolve_claude_bin(self) -> str:
        # On Windows, prioritize claude.exe directly (bin/claude-kllm is a bash script)
        if os.name == "nt":
            exe = shutil.which("claude")
            if exe:
                return exe
            win_candidate = pathlib.Path.home() / ".local" / "bin" / "claude.exe"
            if win_candidate.exists():
                return str(win_candidate)

        # 1. Project-local bin/claude-kllm wrapper (preconfigured with optimal KLLM settings)
        project_kllm = common.ROOT / "bin" / "claude-kllm"
        if os.name != "nt" and project_kllm.exists() and os.access(project_kllm, os.X_OK):
            return str(project_kllm)

        # 2. claude-kllm on PATH
        exe_kllm = shutil.which("claude-kllm")
        if exe_kllm:
            return exe_kllm

        # 3. Direct claude executable on system PATH
        exe = shutil.which("claude")
        if exe:
            return exe

        # 4. Check common local installation paths
        candidates = [
            pathlib.Path.home() / ".local" / "bin" / "claude",
            pathlib.Path("/usr/local/bin/claude"),
            pathlib.Path.home() / ".npm-global" / "bin" / "claude",
        ]
        for c in candidates:
            if c.exists() and os.access(c, os.X_OK):
                return str(c)

        raise FileNotFoundError(
            "Claude Code executable ('claude-kllm' or 'claude') not found on PATH. Install it or ensure it is accessible."
        )

    def _detect_context_and_model(self, model: Optional[str] = None) -> Tuple[str, int]:
        from . import claude_launcher
        try:
            raw = claude_launcher._get(f"{self.relay_url}/admin/api/state", timeout=2)
            state_data = json.loads(raw)
            want_model, ctx = claude_launcher.pick_model(state_data, model)
            return want_model, int(os.environ.get("KLLM_CTX") or ctx)
        except Exception:
            return model or os.environ.get("KLLM_MODEL") or "kllm", int(os.environ.get("KLLM_CTX") or 131072)

    def _build_env(self, model: Optional[str] = None, ctx: int = 131072) -> Dict[str, str]:
        from . import claude_launcher

        target_model = model or os.environ.get("KLLM_MODEL") or "kllm"
        return claude_launcher.build_env(self.relay_url, target_model, ctx)

    async def execute(
        self,
        subagent_id: str,
        prompt: str,
        cwd: str,
        timeout: int,
        model: Optional[str] = None,
        session_tracker: Optional[SubagentSession] = None,
    ) -> Tuple[int, str, int]:
        claude_bin = self._resolve_claude_bin()
        # a blocking HTTP call to the relay: off the event loop, or it deadlocks against the relay serving it
        detected_model, detected_ctx = await asyncio.to_thread(self._detect_context_and_model, model)
        chosen_model = model or detected_model
        env = self._build_env(model=chosen_model, ctx=detected_ctx)

        is_kllm_wrapper = "claude-kllm" in os.path.basename(claude_bin)

        cmd = [
            claude_bin,
            "-p", prompt,
            "--permission-mode", "bypassPermissions",
            "--output-format", "stream-json", "--verbose",
        ]
        if chosen_model:
            cmd.extend(["--model", chosen_model])
        if session_tracker is not None:
            if getattr(session_tracker, "resume_session_id", None):
                cmd.extend(["--resume", session_tracker.resume_session_id])
                session_tracker.session_id = session_tracker.resume_session_id
            else:
                # choose the session id ourselves so a supervisor can resume the worker even if it is killed
                # or times out before it ever prints one
                session_tracker.session_id = str(uuid.uuid4())
                cmd.extend(["--session-id", session_tracker.session_id])

        if not is_kllm_wrapper:
            # When invoking raw claude directly, apply all KLLM optimization flags
            cmd.extend([
                "--strict-mcp-config",
                "--mcp-config", '{"mcpServers":{}}',
                "--tools", "Bash,Read,Edit,Write,Glob,Grep",
                "--settings", json.dumps({
                    "env": {
                        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                        "CLAUDE_CODE_AUTO_MODE_SERVER": "0",
                        "CLAUDE_CODE_AUTO_MODE_CLASSIFY_EDITS": "0",
                    }
                }),
                "--allow-dangerously-skip-permissions",
                "--exclude-dynamic-system-prompt-sections",
            ])

        logger.info(f"Spawning Claude Code subagent {subagent_id} in {cwd} using {claude_bin} (model={chosen_model}, ctx={detected_ctx})")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,   # never inherit the MCP server's stdin (the JSON-RPC pipe)
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=64 * 1024 * 1024,             # stream-json lines carry whole tool results
        )

        if session_tracker:
            session_tracker.process = proc

        progress = session_tracker.progress if session_tracker is not None else {}
        out_lines: List[str] = []
        final_event: Dict[str, Any] = {}
        last_text = {"v": ""}

        async def _read_stdout():
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                out_lines.append(line)
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if isinstance(ev, dict):
                    _track_event(progress, ev)
                    if ev.get("type") == "result":
                        final_event.update(ev)
                    elif ev.get("type") == "assistant":
                        for c in (ev.get("message") or {}).get("content") or []:
                            if c.get("type") == "text" and c.get("text"):
                                last_text["v"] = c["text"]

        async def _read_stderr():
            return (await proc.stderr.read()).decode("utf-8", errors="replace")

        try:
            t_out = asyncio.ensure_future(_read_stdout())
            t_err = asyncio.ensure_future(_read_stderr())
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            await asyncio.wait_for(asyncio.gather(t_out, t_err, return_exceptions=True), timeout=30)
            raw_stderr = t_err.result() if t_err.done() and not t_err.cancelled() and not t_err.exception() else ""
            exit_code = proc.returncode or 0

            tokens = 0
            summary = ""
            if final_event:
                usage = final_event.get("usage") or {}
                tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
                summary = final_event.get("result") or ""
                if final_event.get("is_error") and exit_code == 0:
                    exit_code = 1          # the worker reported an API/model error in its final event
            if not summary:
                summary = last_text["v"] or "".join(out_lines)[-4000:] or raw_stderr
            return exit_code, str(summary), tokens

        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            sid = getattr(session_tracker, "session_id", None)
            return -1, (f"Claude Code subagent timed out after {timeout} seconds."
                        + (f" It can be continued with resume_session_id={sid}." if sid else "")
                        + (f" Last activity: {_progress_view(progress)}" if progress else "")), 0


class ShellDriver(BaseHarnessDriver):
    """
    Executes subagent tasks via standard bash shell with KLLM environment variables.
    Can be used for running scripts, custom subagent harnesses, or automated tasks.
    """

    def __init__(self, relay_url: Optional[str] = None):
        self.relay_url = (relay_url or "http://127.0.0.1:8080").rstrip("/")

    def _build_env(self, model: Optional[str] = None) -> Dict[str, str]:
        env = dict(os.environ)
        m = model or env.get("KLLM_MODEL") or "kllm"
        env.update({
            "OPENAI_BASE_URL": f"{self.relay_url}/v1",
            "OPENAI_API_KEY": "sk-kllm-local",
            "ANTHROPIC_BASE_URL": self.relay_url,
            "ANTHROPIC_AUTH_TOKEN": "sk-kllm-local",
            "ANTHROPIC_MODEL": m,
            "KLLM_RELAY": self.relay_url,
        })
        return env

    async def execute(
        self,
        subagent_id: str,
        prompt: str,
        cwd: str,
        timeout: int,
        model: Optional[str] = None,
        session_tracker: Optional[SubagentSession] = None,
    ) -> Tuple[int, str, int]:
        env = self._build_env(model=model)
        cmd_to_run = session_tracker.task_instructions if session_tracker else prompt
        proc = await _shell(
            cmd_to_run,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,   # never inherit the MCP server's stdin (the JSON-RPC pipe)
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

        if session_tracker:
            session_tracker.process = proc

        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            raw_output = stdout_bytes.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode or 0
            return exit_code, raw_output, 0
        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return -1, f"Shell subagent timed out after {timeout} seconds.", 0


class OpenCodeDriver(BaseHarnessDriver):
    """Executes subagent tasks via OpenCode CLI if installed."""

    def __init__(self, relay_url: Optional[str] = None):
        self.relay_url = (relay_url or "http://127.0.0.1:8080").rstrip("/")

    def _resolve_opencode_bin(self) -> str:
        exe = common.which("opencode")
        if exe:
            return exe
        candidates = [
            pathlib.Path.home() / ".opencode" / "bin" / "opencode",
            pathlib.Path.home() / ".local" / "bin" / "opencode",
            pathlib.Path("/usr/local/bin/opencode"),
        ]
        for c in candidates:
            if c.exists() and os.access(c, os.X_OK):
                return str(c)
        raise FileNotFoundError("OpenCode executable ('opencode') not found on PATH.")

    async def execute(
        self,
        subagent_id: str,
        prompt: str,
        cwd: str,
        timeout: int,
        model: Optional[str] = None,
        session_tracker: Optional[SubagentSession] = None,
    ) -> Tuple[int, str, int]:
        opencode_bin = self._resolve_opencode_bin()

        cmd = [opencode_bin, "run", prompt, "--auto", "--dir", cwd]
        if model:
            cmd.extend(["--model", model])

        env = dict(os.environ)
        env["OPENAI_BASE_URL"] = f"{self.relay_url}/v1"
        env["OPENAI_API_KEY"] = "sk-kllm-local"

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,   # never inherit the MCP server's stdin (the JSON-RPC pipe)
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

        if session_tracker:
            session_tracker.process = proc

        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            raw_output = stdout_bytes.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode or 0
            return exit_code, raw_output, 0
        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return -1, f"OpenCode subagent timed out after {timeout} seconds.", 0


# ---- Subagent Session & Fleet Manager ----------------------------------------

class SubagentSession:
    """Tracks a single running or completed subagent session."""

    def __init__(
        self,
        subagent_id: str,
        harness: str,
        task_instructions: str,
        permitted_files: Optional[List[str]] = None,
        verification_command: Optional[str] = None,
        timeout_seconds: int = 600,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        resume_session_id: Optional[str] = None,
        open_terminal: bool = False,
    ):
        self.subagent_id = subagent_id
        self.harness = harness
        self.task_instructions = task_instructions
        self.permitted_files = permitted_files or []
        self.verification_command = verification_command
        self.timeout_seconds = timeout_seconds
        self.cwd = str(pathlib.Path(cwd or common.ROOT).resolve())
        self.model = model
        self.resume_session_id = resume_session_id
        self.open_terminal = open_terminal
        self.session_id: Optional[str] = resume_session_id   # worker (Claude Code) session id, known before the worker starts
        self.progress: Dict[str, Any] = {}                     # live worker progress parsed from its event stream

        self.status = SubagentStatus.PENDING
        self.created_at = time.time()
        self.completed_at: Optional[float] = None
        self.result: Optional[SubagentResult] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self.task: Optional[asyncio.Task] = None

    def cancel(self) -> bool:
        """Cancel the running subagent and its entire process tree."""
        self.status = SubagentStatus.CANCELLED
        if self.process:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        if self.task and not self.task.done():
            self.task.cancel()
        if not self.result:
            self.result = SubagentResult(
                subagent_id=self.subagent_id,
                harness=self.harness,
                status=SubagentStatus.CANCELLED.value,
                error="Subagent cancelled by user or coordinator.",
                completed_at=time.time(),
            )
        else:
            self.result.status = SubagentStatus.CANCELLED.value
            self.result.error = "Subagent cancelled."
        return True


class FleetManager:
    """Central registry and spawner manager for all subagent sessions."""

    def __init__(self):
        self._sessions: Dict[str, SubagentSession] = {}
        self._lock = asyncio.Lock()
        self.drivers: Dict[str, BaseHarnessDriver] = {
            "claude-code": ClaudeCodeDriver(),
            "claude-kllm": ClaudeCodeDriver(),
            "claude": ClaudeCodeDriver(),
            "shell": ShellDriver(),
            "bash": ShellDriver(),
            "opencode": OpenCodeDriver(),
            "agy": ShellDriver(),
        }

    def register_driver(self, harness: str, driver: BaseHarnessDriver):
        self.drivers[harness.lower()] = driver

    def get_driver(self, harness: str) -> BaseHarnessDriver:
        h = harness.lower()
        if h not in self.drivers:
            raise ValueError(
                f"Unknown harness: '{harness}'. Supported harnesses: {list(self.drivers.keys())}"
            )
        return self.drivers[h]

    async def spawn(
        self,
        harness: str = "claude-code",
        task_instructions: str = "",
        permitted_files: Optional[List[str]] = None,
        verification_command: Optional[str] = None,
        timeout_seconds: int = 600,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        wait: bool = True,
        resume_session_id: Optional[str] = None,
        open_terminal: bool = False,
    ) -> Dict[str, Any]:
        """Spawn a subagent inside the selected harness."""
        if not task_instructions or not task_instructions.strip():
            raise ValueError("task_instructions must not be empty.")

        h_clean = harness.lower().replace("-", "")[:6]
        subagent_id = f"subagent-{h_clean}-{uuid.uuid4().hex[:8]}"

        session = SubagentSession(
            subagent_id=subagent_id,
            harness=harness,
            task_instructions=task_instructions,
            permitted_files=permitted_files,
            verification_command=verification_command,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
            model=model,
            resume_session_id=resume_session_id,
            open_terminal=open_terminal,
        )

        async with self._lock:
            self._sessions[subagent_id] = session
            self._save_state()

        # Inner runner coroutine
        async def _run_session():
            return await self._execute_session(session)

        if wait:
            res = await _run_session()
            return res.to_dict()
        else:
            session.task = asyncio.create_task(_run_session())
            return {
                "subagent_id": subagent_id,
                "harness": harness,
                "status": SubagentStatus.RUNNING.value,
                "message": f"Subagent {subagent_id} launched in background.",
                "created_at": session.created_at,
            }

    async def _execute_session(self, session: SubagentSession) -> SubagentResult:
        """Execute the subagent lifecycle end-to-end and extract deliverables."""
        session.status = SubagentStatus.RUNNING
        start_time = time.time()

        # 1. Baseline status
        baseline_git = get_git_status(session.cwd)
        baseline_untracked = baseline_git["untracked"]
        baseline_modified = baseline_git["modified"]
        baseline_mtimes = baseline_git.get("mtimes", {})

        # Log offset for token tracking
        log_file = common.LOG_DIR / "requests.jsonl"
        log_offset = log_file.stat().st_size if log_file.exists() else 0

        # Build prompt envelope
        subagent_prompt = format_subagent_prompt(
            session.task_instructions,
            permitted_files=session.permitted_files,
            verification_command=session.verification_command,
        )

        driver = self.get_driver(session.harness)
        exit_code: Optional[int] = None
        raw_output = ""
        token_estimate = 0
        error_msg: Optional[str] = None

        try:
            exit_code, raw_output, token_estimate = await driver.execute(
                subagent_id=session.subagent_id,
                prompt=subagent_prompt,
                cwd=session.cwd,
                timeout=session.timeout_seconds,
                model=session.model,
                session_tracker=session,
            )
        except Exception as e:
            logger.exception(f"Subagent {session.subagent_id} execution failed: {e}")
            exit_code = -1
            error_msg = str(e)
            raw_output = error_msg

        exec_duration = round(time.time() - start_time, 2)

        # 2. Extract git diff and modified files
        git_diff, files_modified = get_git_diff(
            cwd=session.cwd,
            baseline_untracked=baseline_untracked,
            baseline_modified=baseline_modified,
            baseline_mtimes=baseline_mtimes,
            baseline_untracked_contents=baseline_git.get("untracked_contents", {}),
        )

        # 3. Check scope violations
        violations = check_scope_violations(
            files_modified=files_modified,
            permitted_files=session.permitted_files,
            cwd=session.cwd,
        )

        # 4. Run verification command if requested
        verification_status = VerificationStatus.SKIPPED.value
        verification_output = ""

        if session.verification_command and exit_code == 0:
            v_code, v_out = await run_verification_command(
                session.verification_command,
                cwd=session.cwd,
                timeout=max(60, min(300, session.timeout_seconds)),
            )
            verification_output = v_out
            verification_status = VerificationStatus.PASSED.value if v_code == 0 else VerificationStatus.FAILED.value
        elif session.verification_command and exit_code != 0:
            verification_status = VerificationStatus.FAILED.value
            verification_output = "Subagent process failed before verification could complete."

        # 5. Extract Kaggle tokens consumed
        kaggle_tokens = count_tokens_from_log(start_time=start_time, start_offset=log_offset)
        if kaggle_tokens == 0 and token_estimate > 0:
            kaggle_tokens = token_estimate
        elif kaggle_tokens == 0:
            # Approximate fallback: ~1 token per 4 chars of prompt + output
            kaggle_tokens = max(10, (len(subagent_prompt) + len(raw_output)) // 4)

        # 6. Determine final status
        is_timeout = (exit_code == -1 and "timed out after" in raw_output.lower())
        final_status = SubagentStatus.COMPLETED
        if session.status == SubagentStatus.CANCELLED:
            final_status = SubagentStatus.CANCELLED
        elif is_timeout:
            final_status = SubagentStatus.TIMEOUT
        elif exit_code != 0:
            final_status = SubagentStatus.FAILED
        elif verification_status == VerificationStatus.FAILED.value:
            final_status = SubagentStatus.FAILED
        elif violations:
            final_status = SubagentStatus.NEEDS_REVIEW

        # 7. Build summary
        subagent_summary = raw_output.strip()
        # Truncate summary if excessively long
        if len(subagent_summary) > 4000:
            subagent_summary = subagent_summary[:4000] + "... [truncated]"

        result = SubagentResult(
            subagent_id=session.subagent_id,
            harness=session.harness,
            status=final_status.value,
            execution_time_seconds=exec_duration,
            kaggle_tokens_consumed=kaggle_tokens,
            cloud_tokens_consumed=0,
            files_modified=files_modified,
            git_diff=git_diff,
            verification_output=verification_output,
            verification_status=verification_status,
            subagent_summary=subagent_summary,
            scope_violations=violations,
            exit_code=exit_code,
            error=error_msg,
            created_at=session.created_at,
            completed_at=time.time(),
            raw_output=raw_output,
            claude_session_id=session.session_id,
        )

        session.status = final_status
        session.completed_at = time.time()
        session.result = result
        self._save_state()
        return result

    @property
    def _state_file(self) -> pathlib.Path:
        p = common.ROOT / "state" / "subagents.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _save_state(self):
        try:
            persisted = self._load_state()
            for sid, s in self._sessions.items():
                if s.result:
                    persisted[sid] = s.result.to_dict()
                else:
                    persisted[sid] = {
                        "subagent_id": s.subagent_id,
                        "harness": s.harness,
                        "status": s.status.value,
                        "created_at": s.created_at,
                        "cwd": s.cwd,
                        "task_instructions": s.task_instructions,
                    }
            tmp = self._state_file.with_name(f"{self._state_file.stem}_{uuid.uuid4().hex[:8]}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(persisted, f, indent=2)
            tmp.replace(self._state_file)
        except Exception as e:
            logger.warning(f"Failed to save subagents state: {e}")

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_file.exists():
            return {}
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get(self, subagent_id: str) -> Optional[Dict[str, Any]]:
        """Get session details or result dictionary."""
        session = self._sessions.get(subagent_id)
        if session:
            if session.result:
                return session.result.to_dict()
            return {
                "subagent_id": session.subagent_id,
                "harness": session.harness,
                "status": session.status.value,
                "created_at": session.created_at,
                "cwd": session.cwd,
                "task_instructions": session.task_instructions,
                "elapsed_seconds": round(time.time() - session.created_at, 1),
                "claude_session_id": session.session_id,
                "progress": _progress_view(session.progress),
            }
        persisted = self._load_state()
        return persisted.get(subagent_id)

    def list_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered subagent sessions."""
        persisted = self._load_state()
        combined: Dict[str, Dict[str, Any]] = dict(persisted)
        for s in self._sessions.values():
            if s.result:
                combined[s.subagent_id] = s.result.to_dict()
            else:
                combined[s.subagent_id] = {
                    "subagent_id": s.subagent_id,
                    "harness": s.harness,
                    "status": s.status.value,
                    "created_at": s.created_at,
                    "completed_at": s.completed_at,
                    "execution_time_seconds": round(time.time() - s.created_at, 2),
                    "files_modified": [],
                    "claude_session_id": s.session_id,
                    "progress": _progress_view(s.progress),
                }
        items = list(combined.values())
        if status:
            items = [x for x in items if str(x.get("status", "")).upper() == status.upper()]
        return sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)

    def cancel(self, subagent_id: str) -> bool:
        """Cancel a running subagent session."""
        session = self._sessions.get(subagent_id)
        if not session:
            persisted = self._load_state()
            if subagent_id in persisted:
                persisted[subagent_id]["status"] = SubagentStatus.CANCELLED.value
                try:
                    with open(self._state_file, "w", encoding="utf-8") as f:
                        json.dump(persisted, f, indent=2)
                except Exception:
                    pass
                return True
            return False
        ok = session.cancel()
        self._save_state()
        return ok


# Global Singleton Fleet Instance
FLEET = FleetManager()


# ---- FastAPI REST Router -----------------------------------------------------

router = APIRouter(prefix="", tags=["spawner"])


@router.post("/admin/api/subagents/spawn")
async def api_spawn_subagent(req: SubagentSpawnRequest):
    """API endpoint to spawn a worker subagent."""
    try:
        res = await FLEET.spawn(
            harness=req.harness,
            task_instructions=req.task_instructions,
            permitted_files=req.permitted_files,
            verification_command=req.verification_command,
            timeout_seconds=req.timeout_seconds,
            cwd=req.cwd,
            model=req.model,
            wait=req.wait,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/admin/api/subagents")
async def api_list_subagents(status: Optional[str] = Query(None)):
    """API endpoint to list subagent sessions."""
    return {"subagents": FLEET.list_all(status=status)}


@router.get("/admin/api/subagents/{subagent_id}")
async def api_get_subagent(subagent_id: str):
    """API endpoint to fetch deliverable for a specific subagent."""
    res = FLEET.get(subagent_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Subagent '{subagent_id}' not found.")
    return res


@router.post("/admin/api/subagents/{subagent_id}/cancel")
async def api_cancel_subagent(subagent_id: str):
    """API endpoint to cancel a running subagent."""
    ok = FLEET.cancel(subagent_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Subagent '{subagent_id}' not found.")
    return {"ok": True, "message": f"Subagent {subagent_id} cancelled."}
