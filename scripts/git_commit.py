"""
Stages all non-ignored project files and creates a clean commit.
Uses dulwich for reliable pure-Python git management.
"""

import os
from pathlib import Path
from dulwich.repo import Repo
from dulwich.ignore import IgnoreFilterManager

repo_path = Path(".").resolve()
repo = Repo(str(repo_path))
ign = IgnoreFilterManager.from_repo(repo)

staged = []

for root, dirs, files in os.walk(repo_path):
    rel_root = os.path.relpath(root, repo_path)
    if rel_root == ".":
        rel_root = ""

    # Filter out ignored directories so we don't traverse them
    to_remove = []
    for d in dirs:
        if d == ".git":
            to_remove.append(d)
            continue
        sub_rel = os.path.normpath(os.path.join(rel_root, d)).replace("\\", "/") + "/"
        if ign.is_ignored(sub_rel):
            to_remove.append(d)
    for r in to_remove:
        dirs.remove(r)

    # Process files
    for f in files:
        sub_rel = os.path.normpath(os.path.join(rel_root, f)).replace("\\", "/")
        if sub_rel.startswith(".git/") or sub_rel == ".git":
            continue
        if not ign.is_ignored(sub_rel):
            staged.append(sub_rel)

print(f"Staging {len(staged)} files...")

# Stage files in repo index
index = repo.open_index()
for rel_path in staged:
    full_p = repo_path / rel_path
    if full_p.is_file():
        with open(full_p, "rb") as fp:
            data = fp.read()
        blob = repo.object_store.add_object(
            dulwich.objects.Blob.from_string(data) if hasattr(dulwich, "objects") else None
        )

# Use dulwich porcelain to stage properly
import dulwich.porcelain as dp
dp.add(repo, paths=staged)

# Commit
commit_msg = b"Initial commit: SmartBelt v2 complete ground-up pipeline with real hardware & UI"
commit_id = dp.commit(
    repo,
    message=commit_msg,
    committer=b"SmartBelt Developer <smartbelt@local.domain>",
    author=b"SmartBelt Developer <smartbelt@local.domain>",
)

print(f"Commit created: {commit_id.decode() if isinstance(commit_id, bytes) else commit_id}")
