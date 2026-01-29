import os
import shlex
import shutil

from pathlib import Path
from typing import Generator, Callable


WANDB_DIRS = ("wandb", ".wandb")


def _is_py_or_dockerfile(path: str, root: str) -> bool:
    file = os.path.basename(path)
    return file.endswith(".py") or file.startswith("Dockerfile")


def include_code_files(path: str, root: str, code_ext: list[str]):
    file = os.path.basename(path)
    return any(file.endswith(ext) for ext in code_ext) or file.startswith("Dockerfile")


def exclude_code_folders(path: str, root: str, code_folders: list[str]):
    return any(
        os.path.relpath(path, root).startswith(code_folder + os.sep) for code_folder in code_folders
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
