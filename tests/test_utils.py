import os
from functools import partial
from slurmic.utils import (
    exclude_code_folders,
    include_code_files,
    filtered_dir,
    pack_code_files,
)


def _packed_relpaths(code_target):
    """Collect packed files as paths relative to the ``code`` target dir."""
    return {
        os.path.relpath(os.path.join(dirpath, fname), code_target)
        for dirpath, _, files in os.walk(code_target)
        for fname in files
    }


def test_filtered_dir(tmp_path):
    # Create a mock directory structure like:
    # /src
    #   /file1.py
    #   /file2.txt
    # /wandb
    #   /file3.txt
    # /Dockerfile
    # /output
    #    /src
    #       /file1.py
    #       /file2.txt
    (tmp_path / "src").mkdir()
    (tmp_path / "wandb").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "src" / "file1.py").write_text("print('Hello World')")
    (tmp_path / "src" / "file2.txt").write_text("This is a text file.")
    (tmp_path / "wandb" / "file3.txt").write_text("Wandb log file.")
    (tmp_path / "Dockerfile").write_text("FROM python:3.8-slim")
    (tmp_path / "output" / "src").mkdir()
    (tmp_path / "output" / "src" / "file1.py").write_text("print('Hello World')")
    (tmp_path / "output" / "src" / "file2.txt").write_text("This is a text file.")

    # code files to preserve
    code_suffix = [".py", ".sh", ".yaml", ".toml"]
    # folders to exclude
    exclude_folders = ["wandb", "output"]

    included_files = list(
        filtered_dir(
            str(tmp_path),
            partial(include_code_files, code_ext=code_suffix),
            partial(exclude_code_folders, code_folders=exclude_folders),
        )
    )

    expected_files = {
        str(tmp_path / "src" / "file1.py"),
        str(tmp_path / "Dockerfile"),
    }

    assert len(included_files) == len(expected_files)
    assert set(included_files) == expected_files


def test_pack_code_files_without_gitignore_honors_include_and_exclude(tmp_path):
    # No .gitignore present, so ``include_fn`` is the sole include filter and
    # ``exclude_dir_fn`` hard-excludes folders.
    #   /src/main.py        -> kept (matches suffix)
    #   /src/notes.txt      -> dropped (suffix not whitelisted)
    #   /config.yaml        -> kept
    #   /Dockerfile         -> kept (always whitelisted by include_code_files)
    #   /data/direct.py     -> dropped (direct child of excluded folder)
    #   /data/cache/big.py  -> dropped (nested under excluded folder)
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "data" / "cache").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('main')")
    (root / "src" / "notes.txt").write_text("not code")
    (root / "config.yaml").write_text("a: 1")
    (root / "Dockerfile").write_text("FROM python:3.12-slim")
    (root / "data" / "direct.py").write_text("print('direct child, excluded')")
    (root / "data" / "cache" / "big.py").write_text("print('excluded by folder')")

    code_target = pack_code_files(
        str(root),
        str(tmp_path / "out"),
        include_fn=partial(include_code_files, code_ext=[".py", ".yaml"]),
        exclude_dir_fn=partial(exclude_code_folders, code_folders=["data"]),
    )

    packed = _packed_relpaths(str(code_target))
    assert packed == {
        os.path.join("src", "main.py"),
        "config.yaml",
        "Dockerfile",
    }
    # include_fn honored: a non-matching suffix is dropped.
    assert os.path.join("src", "notes.txt") not in packed
    # exclude_dir_fn honored: .py files both directly in and nested under an
    # excluded folder are dropped.
    assert os.path.join("data", "direct.py") not in packed
    assert os.path.join("data", "cache", "big.py") not in packed


def test_pack_code_files_with_gitignore_is_honored(tmp_path):
    # With a .gitignore, every non-ignored file is packed and ignored files are
    # dropped. The default add-back filter only re-adds .py/Dockerfile, so the
    # ignored non-code files below stay excluded.
    root = tmp_path / "project"
    (root / "cache").mkdir(parents=True)
    (root / ".gitignore").write_text("*.log\ncache/\n")
    (root / "app.py").write_text("print('app')")
    (root / "notes.txt").write_text("keep me")
    (root / "run.log").write_text("log line")  # gitignored -> dropped
    (root / "cache" / "blob.bin").write_text("binary")  # gitignored dir -> dropped

    code_target = pack_code_files(str(root), str(tmp_path / "out"))

    packed = _packed_relpaths(str(code_target))
    assert packed == {".gitignore", "app.py", "notes.txt"}
    assert "run.log" not in packed
    assert os.path.join("cache", "blob.bin") not in packed


def test_pack_code_files_gitignore_add_back_reincludes_matching_suffix(tmp_path):
    # With a .gitignore, ``include_fn`` acts as an add-back whitelist: files that
    # are gitignored but match the suffix are re-included anyway.
    root = tmp_path / "project"
    (root / "generated").mkdir(parents=True)
    (root / ".gitignore").write_text("generated/\n*.gen.py\n")
    (root / "main.py").write_text("print('main')")
    (root / "skip.gen.py").write_text("# generated, but .py")
    (root / "generated" / "model.py").write_text("# generated module")

    code_target = pack_code_files(
        str(root),
        str(tmp_path / "out"),
        include_fn=partial(include_code_files, code_ext=[".py"]),
    )

    packed = _packed_relpaths(str(code_target))
    assert packed == {
        ".gitignore",
        "main.py",
        "skip.gen.py",
        os.path.join("generated", "model.py"),
    }
