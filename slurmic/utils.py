import os
import shlex
import shutil

import pathspec

from pathlib import Path
from typing import Generator, Callable, Optional


WANDB_DIRS = ("wandb", ".wandb")

# Patterns always ignored regardless of the project's .gitignore.
DEFAULT_GITIGNORE_PATTERNS = (".git/",)


def load_gitignore_spec(root: str) -> Optional[pathspec.GitIgnoreSpec]:
    """Build a gitignore matcher from the root ``.gitignore`` (if present).

    Returns ``None`` when the project has no ``.gitignore`` file, so callers can keep
    their original ``include_fn`` / ``exclude_dir_fn`` filtering untouched. When a
    ``.gitignore`` exists, VCS metadata is always ignored too (see
    ``DEFAULT_GITIGNORE_PATTERNS``).
    """
    gitignore_path = os.path.join(root, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return None

    with open(gitignore_path, "r", encoding="utf-8") as f:
        lines = list(DEFAULT_GITIGNORE_PATTERNS) + f.read().splitlines()
    return pathspec.GitIgnoreSpec.from_lines(lines)


def is_gitignored(
    spec: Optional[pathspec.GitIgnoreSpec],
    path: str,
    root: str,
    is_dir: bool = False,
) -> bool:
    """Return ``True`` if ``path`` is ignored by ``spec`` relative to ``root``."""
    if spec is None:
        return False

    rel = os.path.relpath(path, root)
    if rel == os.curdir:
        return False

    # gitignore patterns are always posix-style; directories match with a trailing slash.
    rel = Path(rel).as_posix()
    if is_dir:
        rel += "/"
    return spec.match_file(rel)


def _is_py_or_dockerfile(path: str, root: str) -> bool:
    file = os.path.basename(path)
    return file.endswith(".py") or file.startswith("Dockerfile")


def include_code_files(path: str, root: str, code_ext: list[str]):
    file = os.path.basename(path)
    return any(file.endswith(ext) for ext in code_ext) or file.startswith("Dockerfile")


def exclude_code_folders(path: str, root: str, code_folders: list[str]):
    rel = os.path.relpath(path, root)
    return any(
        rel == code_folder or rel.startswith(code_folder + os.sep) for code_folder in code_folders
    )


def exclude_wandb_fn(path: str, root: str) -> bool:
    return any(
        os.path.relpath(path, root).startswith(wandb_dir + os.sep) for wandb_dir in WANDB_DIRS
    )


def filtered_dir(
    root: str,
    include_fn: Callable[[str, str], bool],
    exclude_dir_fn: Callable[[str, str], bool],
) -> Generator[str, None, None]:
    """Simple generator to walk a directory."""

    for dirpath, _, files in os.walk(root):
        if exclude_dir_fn(dirpath, root):
            continue
        for fname in files:
            file_path = os.path.join(dirpath, fname)
            if include_fn(file_path, root):
                yield file_path


def pack_code_files(
    root: str,
    target_root: str,
    include_fn: Callable[[str, str], bool] = _is_py_or_dockerfile,
    exclude_dir_fn: Callable[[str, str], bool] = exclude_wandb_fn,
):
    root = os.path.abspath(root)
    code_root = Path(os.path.abspath(root))
    code_target = Path(os.path.abspath(target_root)) / "code"

    # Ensure target directory exists
    if not code_root.exists():
        raise ValueError(f"Code root {code_root} does not exist.")
    if not code_target.exists():
        code_target.mkdir(parents=True)

    # When a .gitignore exists it becomes the primary discovery mechanism: every file
    # that is not ignored gets packed, and the manual ``include_fn`` *adds back*
    # explicitly matched files even when they are gitignored. Without a .gitignore we
    # fall back to the original ``include_fn`` filtering. The manual ``exclude_dir_fn``
    # always hard-excludes whole folders regardless.
    spec = load_gitignore_spec(root)
    if spec is not None:
        base_include_fn = include_fn

        def include_fn(path: str, r: str) -> bool:
            return not is_gitignored(spec, path, r) or base_include_fn(path, r)

    for file_path in filtered_dir(root, include_fn, exclude_dir_fn):
        save_name = os.path.relpath(file_path, root)
        sub_file_path, file_name = os.path.split(save_name)
        sub_file_full_path = code_target / sub_file_path
        if not sub_file_full_path.exists():
            sub_file_full_path.mkdir(parents=True)
        shutil.copy(file_path, sub_file_full_path / file_name)
    return code_target


def reconstruct_command_line(argv):
    # Quote each argument that needs special handling (like spaces or shell characters)
    # and join them with spaces to form the command line
    return " ".join(shlex.quote(arg) for arg in argv)
