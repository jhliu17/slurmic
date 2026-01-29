from functools import partial
from slurmic.utils import exclude_code_folders, include_code_files, filtered_dir


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
