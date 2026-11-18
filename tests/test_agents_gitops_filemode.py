"""Worktrees on filesystems without a real exec bit (NTFS through FUSE) must not show phantom mode changes."""
import subprocess

from gateway.agents import gitops


def _repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    (r / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=r, check=True)
    return r


def test_worktree_ignores_exec_bit_when_filesystem_cannot_keep_it(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(gitops, "exec_bit_reliable", lambda d: False)
    wt = gitops.add_worktree(repo, tmp_path / "wt", gitops.head(repo))
    (wt / "a.py").chmod(0o755)                       # what NTFS reports for every file
    assert gitops.git(wt, "config", "--worktree", "core.fileMode").strip() == "false"
    assert gitops.git(wt, "status", "--porcelain").strip() == ""


def test_worktree_left_alone_on_posix_filesystems(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(gitops, "exec_bit_reliable", lambda d: True)
    wt = gitops.add_worktree(repo, tmp_path / "wt", gitops.head(repo))
    assert gitops.git(wt, "config", "--worktree", "--get", "core.fileMode", check=False).strip() == ""
