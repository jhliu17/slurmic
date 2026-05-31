# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`slurmic` is a library for running Python functions on Slurm as if they were local calls. It wraps [submitit](https://github.com/facebookincubator/submitit) and adds a decorator-based API, job dependency chaining, distributed-training launch orchestration, and codebase snapshotting ("code packing").

## Commands

```bash
uv sync --locked          # install deps into .venv (matches CI)
uv run pytest             # run the full test suite
uv run pytest tests/test_slurm.py::test_sequential_jobs   # run a single test
uv tool run ruff check    # lint (CI gate)
uv tool run ruff format --check --diff   # format check (CI gate)
uv build                  # build wheel
```

Tests run on Python 3.10–3.13. Ruff targets `py312`, line length 100.

The tests do **not** require a real Slurm cluster — they use `submitit.core.test_core.MockedSubprocess` (see `mocked_slurm()` in [tests/test_slurm.py](tests/test_slurm.py)) and drive job execution manually via `submission.process_job(...)`. The distributed test is skipped because the mock can't model distributed tasks.

## Architecture

The public API is re-exported from [slurmic/__init__.py](slurmic/__init__.py). The key design split: there are **two** `SlurmFunction` classes.

- [slurmic/function.py](slurmic/function.py) — the **public** `SlurmFunction`, a thin facade. Every config-mutating method (`configure`, `__getitem__`, `on_condition`) returns a *copy* so configuration is non-destructive. It delegates all real work to `self.engine`.
- [slurmic/core/_slurm.py](slurmic/core/_slurm.py) — the **engine** `SlurmFunction` (`SlurmBackend`), which holds the actual submission logic, executor construction, and submitit interaction.

When editing behavior, change the engine; the public class mostly forwards.

### Decorator entry points ([slurmic/wrap.py](slurmic/wrap.py))

- `slurm_fn` / `slurm` (alias) — wraps a function into a `SlurmFunction`. Usage: `fn[slurm_config](args)`. `__getitem__` applies config and returns a configured copy; calling it submits.
- `slurm_launcher` — decorator for a program's main entry. Parses CLI args (via `tyro`) into a dataclass that must contain a `SlurmConfig` field (default key `"slurm"`), then configures the function with those args. Used together with `system_argv` for the distributed second launch.
- `slurm_function` — **deprecated**, kept for back-compat (`fn(slurm_config)(args)` style).

### Submission flow

`engine.__call__` → `_should_be_submitted_to_executor` decides whether to actually submit (modes `slurm`/`debug`/`local`) or just run the function inline (mode `run`). `prepare_executor` maps `SlurmConfig.mode` to a submitit cluster type via `cluster_dispatch` (`run`→`debug`, etc.). `slurm` mode is non-blocking and returns a `Job`; all other modes block.

### Modes ([slurmic/config.py](slurmic/config.py))

`SlurmConfig.mode`: `run` (call directly, no submitit), `debug` (submitit debug, pdb-capable), `local` (subprocess, no GPU allocation), `slurm` (real cluster). `partition` is required; validation happens in `__post_init__`. `SlurmArgs` is an alias for `SlurmConfig`.

### Distributed jobs (the tricky part)

When `use_distributed_env=True`, a job is launched **twice**:
1. First launch requests resources and sets up the distributed environment.
2. The env is exported to a `slurmic_distributed_env.sh` script, then the real command (`distributed_launch_command`) is run as a second launch.

This is implemented by:
- [slurmic/task.py](slurmic/task.py) — `Task` base class and `PyTorchDistributedTask`, which computes `num_processes`/`machine_rank`/master addr-port from the Slurm env and formats them into the launch command.
- [slurmic/core/_slurm_context.py](slurmic/core/_slurm_context.py) — `SubmititDistributedCommandContext`, a context manager that **monkey-patches** `SlurmExecutor._submitit_command_str` to inject the env-export + second-launch command into the sbatch script. Patch applies only in `slurm` mode and is reverted on exit.

The sentinel env var `SLURMIC_SLURM_HAS_BEEN_SET_UP` marks that setup is done, so the second launch runs the function directly instead of re-submitting. The `{num_processes}`, `{num_machines}`, `{machine_rank}`, `{main_process_ip}`, `{main_process_port}` placeholders in `distributed_launch_command` are filled from `DistributedTaskConfig`.

### Job dependencies

`on_condition(jobs, condition)` / `afterok` / `afterany` / `afternotok` build up a Slurm `dependency` string (e.g. `afterok:123:456`) in `slurm_params_kwargs`. Multiple calls append. `map_array` submits an array job (slurm mode, non-distributed only).

### Code packing ([slurmic/utils.py](slurmic/utils.py))

When `pack_code=True`, the codebase is snapshotted into `<output_path>/code` before submission and the job `chdir`s there (`SLURMIC_SLURM_PACKED_CODE` env var), decoupling the running job from later edits. File selection priority (see [docs/code_packing.md](docs/code_packing.md)):
- If `code_root` has a `.gitignore`: it's the primary filter (non-ignored files are packed); `code_file_suffixes` acts as an **add-back** whitelist that re-includes gitignored files matching those extensions.
- If no `.gitignore`: `code_file_suffixes` is the sole include filter.
- `exclude_code_folders` always hard-excludes whole folders regardless.

Uses `pathspec.GitIgnoreSpec` for matching. `.git/` is always ignored.

## Docs

Sphinx docs live in [docs/](docs/) (MyST markdown + rST). Build with the Sphinx Makefile. API pages auto-generate from docstrings, so keep docstrings accurate when changing the public API. [.github/workflows/docs.yml](.github/workflows/docs.yml) auto-deploys docs to the `gh-pages` branch on every push to `main` — no manual step.

## Releasing

The version is the single source of truth in [slurmic/version.py](slurmic/version.py) (`_MAJOR`/`_MINOR`/`_PATCH`/`_SUFFIX`); `pyproject.toml` reads `VERSION` from it dynamically. On `main` the version carries a `dev` suffix and the patch sits one ahead of the last released build (e.g. `0.2.0dev`). Publishing to PyPI is triggered **only** by creating a GitHub Release — [.github/workflows/wheel.yml](.github/workflows/wheel.yml) runs `uv build` and `twine upload dist/*` (both the sdist and the wheel) using the `PYPI_TOKEN` secret on the `release: created` event. Tag convention is `v<version>` (e.g. `v0.1.0`).

To cut a release:

1. **Pre-flight** on a clean, up-to-date `main`:

   ```bash
   git checkout main && git pull
   uv run pytest
   uv tool run ruff check && uv tool run ruff format --check --diff
   ```

2. **Set the release version** in [slurmic/version.py](slurmic/version.py): clear the suffix (`_SUFFIX = ""`) and confirm `_MAJOR/_MINOR/_PATCH`. Verify:

   ```bash
   uv build && uv run python -c "from slurmic.version import VERSION; print(VERSION)"
   ```

3. **Commit and push** the release version (`git commit -am "Release v0.2.0" && git push origin main`).
4. **Create the GitHub Release** — this creates the tag *and* triggers the PyPI upload:

   ```bash
   gh release create v0.2.0 --target main --title "v0.2.0" --generate-notes
   ```

   Then confirm the `wheel.yml` run succeeds and the new version appears on PyPI.
5. **Open the next dev cycle**: bump `_PATCH` by one and restore `_SUFFIX = "dev"` (→ `0.2.1dev`), commit as `Start v0.2.1 development`, and push.
