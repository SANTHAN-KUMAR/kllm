"""Git plumbing for agent sessions: one worktree per session, checkpoints on hidden refs, apply, restore, fork.

A snapshot is the full worktree state (tracked and untracked, minus ignored files) written as a tree through a
temporary index, so it never touches the worktree's own index or HEAD. A checkpoint is a commit of a snapshot on
refs/kllm/checkpoints/<session>/<n>; its parent is the previous checkpoint, so `git log` on the ref shows the
session's history.
"""
import os, pathlib, re, shutil, subprocess, tempfile

GIT_ENV = {"GIT_AUTHOR_NAME": "kllm agents", "GIT_AUTHOR_EMAIL": "agents@kllm.local",
           "GIT_COMMITTER_NAME": "kllm agents", "GIT_COMMITTER_EMAIL": "agents@kllm.local",
           "GIT_TERMINAL_PROMPT": "0"}


class GitError(Exception):
    pass


def git(cwd, *args, env=None, input=None, check=True, timeout=120):
    e = dict(os.environ, **GIT_ENV, **(env or {}))
    p = subprocess.run(["git", *args], cwd=str(cwd), env=e, input=input, capture_output=True, text=True,
                       timeout=timeout)
    if check and p.returncode != 0:
        raise GitError((p.stderr or p.stdout).strip()[:800] or f"git {' '.join(args)} failed")
    return p.stdout


def repo_root(path):
    """Top of the git checkout that contains path, or None."""
    try:
        return pathlib.Path(git(path, "rev-parse", "--show-toplevel").strip())
    except (GitError, FileNotFoundError, NotADirectoryError, OSError):
        return None


def head(cwd):
    return git(cwd, "rev-parse", "HEAD").strip()


def slug(text, n=32):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:n].rstrip("-") or "session")


def exec_bit_reliable(directory):
    """False on filesystems that report every file as executable (NTFS or exFAT through FUSE)."""
    try:
        fd, p = tempfile.mkstemp(prefix=".kllm-mode-", dir=str(directory))
        os.close(fd)
        try:
            os.chmod(p, 0o644)
            return not (os.stat(p).st_mode & 0o111)
        finally:
            os.unlink(p)
    except OSError:
        return True


def add_worktree(repo, path, commit):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "--detach", str(path), commit)
    if not exec_bit_reliable(path):
        # Every file would show a 100644 -> 100755 change: the agent chases phantom edits and an apply would
        # carry mode changes into the user's branch. Ignore the exec bit in this worktree only.
        git(repo, "config", "extensions.worktreeConfig", "true")
        git(path, "config", "--worktree", "core.fileMode", "false")
    return path


def remove_worktree(repo, path):
    if not path:
        return
    try:
        git(repo, "worktree", "remove", "--force", str(path))
    except GitError:
        shutil.rmtree(path, ignore_errors=True)
        git(repo, "worktree", "prune", check=False)


def snapshot(cwd):
    """-> tree sha of the current worktree contents (tracked + untracked, excluding ignored)."""
    with tempfile.TemporaryDirectory(prefix="kllm-idx-") as d:
        env = {"GIT_INDEX_FILE": os.path.join(d, "index")}
        git(cwd, "read-tree", "HEAD", env=env)
        git(cwd, "add", "-A", ".", env=env)
        return git(cwd, "write-tree", env=env).strip()


def commit_tree(cwd, tree, parent, message):
    return git(cwd, "commit-tree", tree, "-p", parent, "-m", message).strip()


def tree_of(cwd, commit):
    return git(cwd, "rev-parse", f"{commit}^{{tree}}").strip()


def changed_files(cwd, a, b):
    out = git(cwd, "diff", "--name-only", "--no-renames", a, b)
    return [x for x in out.splitlines() if x]


def diff(cwd, a, b, path=None, max_bytes=400_000):
    args = ["diff", "--no-color", "--no-renames", a, b] + (["--", path] if path else [])
    out = git(cwd, *args)
    return out if len(out) <= max_bytes else out[:max_bytes] + "\n[diff truncated]\n"


def numstat(cwd, a, b):
    rows = []
    for line in git(cwd, "diff", "--numstat", "--no-renames", a, b).splitlines():
        p = line.split("\t", 2)
        if len(p) == 3:
            rows.append({"file": p[2], "added": None if p[0] == "-" else int(p[0]),
                         "removed": None if p[1] == "-" else int(p[1])})
    return rows


def checkpoint(cwd, session_id, n, parent, label):
    """Commit the worktree state as checkpoint n. -> (sha, tree, changed files vs parent)."""
    tree = snapshot(cwd)
    if tree == tree_of(cwd, parent):
        return None, tree, []
    sha = commit_tree(cwd, tree, parent, label)
    git(cwd, "update-ref", f"refs/kllm/checkpoints/{session_id}/{n}", sha)
    return sha, tree, changed_files(cwd, parent, sha)


def restore(cwd, commit):
    """Make the worktree exactly match commit (removes files created later; ignored files stay)."""
    git(cwd, "reset", "--hard", commit)
    git(cwd, "clean", "-fd")


def drop_refs(repo, session_id):
    out = git(repo, "for-each-ref", "--format=%(refname)", f"refs/kllm/checkpoints/{session_id}/", check=False)
    for ref in out.split():
        git(repo, "update-ref", "-d", ref, check=False)


def apply_new_branch(cwd, base, name, message):
    tree = snapshot(cwd)
    if tree == tree_of(cwd, base):
        raise GitError("there are no changes to apply")
    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", name or "") or ".." in name:
        raise GitError("branch names use letters, digits, dot, dash, underscore and slash")
    if git(cwd, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).strip():
        raise GitError(f"branch {name} already exists")
    sha = commit_tree(cwd, tree, base, message)
    git(cwd, "branch", name, sha)
    return sha


def apply_commit(repo, cwd, base, message):
    """Commit the session's changes on the main checkout's current branch (3-way if it moved)."""
    tree = snapshot(cwd)
    if tree == tree_of(cwd, base):
        raise GitError("there are no changes to apply")
    # bytes, not text: a text-mode pipe on Windows rewrites \n as \r\n and the patch no longer applies
    d = subprocess.run(["git", "diff", "--binary", "--no-renames", base, tree], cwd=str(cwd), capture_output=True,
                       env=dict(os.environ, **GIT_ENV), timeout=120)
    if d.returncode != 0:
        raise GitError(d.stderr.decode("utf-8", "replace").strip()[:800] or "git diff failed")
    patch = d.stdout
    files = changed_files(cwd, base, tree)
    dirty = [f for f in git(repo, "status", "--porcelain", "--", *files).splitlines() if f.strip()]
    if dirty:
        raise GitError("the main checkout has uncommitted changes in " + ", ".join(x[3:] for x in dirty[:6])
                       + ": commit or stash them, or apply to a new branch")
    p = subprocess.run(["git", "apply", "--3way", "--index", "-"], cwd=str(repo), input=patch,
                       capture_output=True, env=dict(os.environ, **GIT_ENV))
    p.stderr = p.stderr.decode("utf-8", "replace")
    if p.returncode != 0:
        git(repo, "reset", "-q", "--", *files, check=False)
        git(repo, "checkout", "--", *files, check=False)
        git(repo, "clean", "-fq", "--", *files, check=False)     # these paths were clean before: only new files go
        raise GitError("the changes conflict with the current branch: apply them to a new branch instead. "
                       + (p.stderr or "").strip()[:300])
    git(repo, "commit", "-q", "-m", message, "--", *files)
    return head(repo)
