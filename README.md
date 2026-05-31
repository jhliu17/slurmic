# 🌩️ slurmic

[![Pytest](https://github.com/jhliu17/SLURMIC/actions/workflows/pytest.yml/badge.svg)](https://github.com/jhliu17/SLURMIC/actions/workflows/pytest.yml) [![Documentation](https://github.com/jhliu17/SLURMIC/actions/workflows/docs.yml/badge.svg)](https://github.com/jhliu17/SLURMIC/actions/workflows/docs.yml) [![PyPI](https://img.shields.io/pypi/v/slurmic.svg)](https://pypi.org/project/slurmic/)

**slurmic** lets you run Python functions on a [Slurm](https://slurm.schedmd.com/) cluster as if they were ordinary local calls. Decorate a function, attach a resource config, and call it — slurmic handles serialization, submission, and result retrieval for you. It builds on [submitit](https://github.com/facebookincubator/submitit) and adds a clean decorator-based API, scheduler-level job dependency chaining, distributed-training launch orchestration, and codebase snapshotting ("code packing").

```python
from slurmic import SlurmConfig, slurm_fn

@slurm_fn
def add(a, b):
    return a + b

cfg = SlurmConfig(mode="slurm", partition="gpu", cpus_per_task=8, mem="1GB")

job = add[cfg](1, b=2)   # submitted to Slurm, returns immediately
print(job.result())      # blocks until done => 3
```

## Why slurmic

- **No boilerplate.** No hand-written `sbatch` scripts, no manual `pickle`, no polling loops. The decorated function runs unchanged — slurmic only controls *where* and *how*.
- **One switch from laptop to cluster.** The `mode` field flips a function between running in-process, as a local subprocess, under `pdb`, or on the real cluster — without touching the function body.
- **Pipelines on the scheduler.** Chain jobs with `afterok` / `afterany` / `afternotok` so Slurm manages ordering; submit the whole DAG up front without keeping your Python process alive.
- **Array sweeps.** Fan one function out over many argument sets in a single array submission with `map_array`.
- **Distributed launches.** Set `use_distributed_env=True` and slurmic computes the rendezvous (world size, rank, master addr/port) and runs your `torchrun` / `accelerate` / `deepspeed` command across nodes.
- **Reproducible runs.** With `pack_code=True`, the source tree is snapshotted before submission so a queued or long-running job is decoupled from later edits.

## Installation

slurmic supports **Python 3.10–3.13** and is tested on Linux systems with Slurm installed.

```bash
pip install slurmic
```

> The library itself imports fine anywhere, but `mode="slurm"` requires the Slurm CLI (`sbatch`, `srun`, …) to be available on the host.

## Usage

### Run a Python function on Slurm

Decorate a plain function with `@slurm_fn`, bind a `SlurmConfig` using the `[config]` syntax, then call it like normal. The call returns a non-blocking job handle; `job.result()` waits for completion and returns the function's value (or re-raises whatever it raised on the cluster).

```python
from slurmic import SlurmConfig, slurm_fn

@slurm_fn
def run_on_slurm(a, b):
    return a + b

slurm_config = SlurmConfig(
    mode="slurm",
    partition="PARTITION",
    job_name="EXAMPLE",
    tasks_per_node=1,
    cpus_per_task=8,
    mem="1GB",
)

job = run_on_slurm[slurm_config](1, b=2)  # submitted to Slurm
result = job.result()                     # block and get the result => 3
```

The same function can be reused with different configs — `run_on_slurm[other_config](...)` — because every config-binding step returns a configured *copy*, leaving the original untouched.

### Run modes

`SlurmConfig.mode` controls *where* the function executes, so you can develop locally and scale out by changing a single field:

| Mode | Behavior | Blocking? |
| --- | --- | --- |
| `run` (default) | Calls the function directly in-process, no submitit involved. | yes |
| `debug` | Runs through submitit's debug executor; drops into `pdb` at a breakpoint. | yes |
| `local` | Runs as a local subprocess (no GPU allocation, `CUDA_VISIBLE_DEVICES` not set). | yes |
| `slurm` | Submits to the real Slurm cluster. | **no** — returns a `Job` |

Only `slurm` mode is non-blocking; every other mode runs and returns the value directly.

### Manage job dependencies

`.on_condition(job)` (and the shorthands `.afterok` / `.afterany` / `.afternotok`) returns a new configured function whose job is queued immediately but held by the scheduler until its dependency finishes. This builds the dependency chain on Slurm itself.

```python
jobs = []

# job1 is submitted to Slurm directly
job1 = run_on_slurm[slurm_config](10, 2)
jobs.append(job1)

# fn2 will only start once job1 has finished successfully
fn2 = run_on_slurm[slurm_config].on_condition(job1)   # same as .afterok(job1)
job2 = fn2(7, 12)
jobs.append(job2)

results = [job.result() for job in jobs]  # blocks until all jobs are done
assert results == [12, 19]
```

Condition shorthands:

- `.afterok(*jobs)` — start after the dependencies complete **successfully** (the default).
- `.afterany(*jobs)` — start after the dependencies finish, regardless of exit status.
- `.afternotok(*jobs)` — start after a dependency **fails** (useful for cleanup / fallback jobs).

Calls can be chained to combine conditions. **Use this** for multi-stage pipelines (preprocess → train → evaluate): submit everything up front and let Slurm start each stage the moment its prerequisites are met.

### Map an array of jobs

`.map_array(*arg_lists)` zips the argument lists together and submits one job per tuple as a single Slurm **job array**, returning a list of handles.

```python
# runs (1,3), (2,4), (8,8), (9,9)
jobs = run_on_slurm[slurm_config].map_array([1, 2, 8, 9], [3, 4, 8, 9])
results = [job.result() for job in jobs]
assert results == [4, 6, 16, 18]
```

**Use this** for hyperparameter sweeps, per-seed runs, per-shard data processing, or batch inference. (Array submission is `slurm` mode, non-distributed only.)

### Distributed jobs

For multi-process / multi-GPU training, set `use_distributed_env=True` and provide a `distributed_launch_command`. slurmic allocates the resources, sets up the distributed environment, and then runs your command — substituting the `{...}` placeholders with the values it computed from the Slurm allocation.

```python
# distributed launch with accelerate as an example
slurm_config = SlurmConfig(
    mode="slurm",
    cpus_per_task=8,
    gpus_per_node=4,
    use_distributed_env=True,
    distributed_launch_command=(
        "accelerate launch --config_file CONFIG_FILE "
        "--num_processes {num_processes} --num_machines {num_machines} "
        "--machine_rank {machine_rank} --main_process_ip {main_process_ip} "
        "--main_process_port {main_process_port} main.py"  # main.py is the distributed entry
    ),
)

main[slurm_config](config)
```

Available placeholders: `{num_processes}`, `{num_machines}`, `{machine_rank}`, `{main_process_ip}`, `{main_process_port}` — so you can plug in `torchrun`, `accelerate`, `deepspeed`, or any launcher. **Use this** when one GPU/process isn't enough and you don't want to hand-wire the rendezvous for every job.

### Launch a program from the CLI

`slurm_launcher` turns a program's `main` into a Slurm-aware entry point. It parses CLI arguments (via [tyro](https://github.com/brentyi/tyro)) into a dataclass that contains a `SlurmConfig` field (key `"slurm"` by default), then configures and runs the function with those args.

```python
from dataclasses import dataclass, field
from slurmic import SlurmConfig, slurm_launcher

@dataclass
class Args:
    lr: float = 1e-3
    slurm: SlurmConfig = field(default_factory=lambda: SlurmConfig(mode="slurm", partition="gpu"))

@slurm_launcher(Args)
def main(args: Args):
    train(lr=args.lr)

if __name__ == "__main__":
    main()
```

```bash
python train.py --lr 5e-4 --slurm.mode slurm --slurm.partition gpu
```

### Code packing

With `pack_code=True`, slurmic snapshots your source tree into the job's output folder before submission and runs the job against that frozen copy — so editing your working tree afterwards won't affect already-queued or long-running jobs. File selection is `.gitignore`-aware (with `code_file_suffixes` as an add-back whitelist) and `exclude_code_folders` hard-excludes whole directories. See the [Code Packing docs](docs/code_packing.md) for the full selection rules.

## Configuration reference

`SlurmConfig` (aliased as `SlurmArgs`) holds the resource request and run options. Common fields:

| Field | Default | Description |
| --- | --- | --- |
| `mode` | `"run"` | `run` / `debug` / `local` / `slurm` (see [Run modes](#run-modes)). |
| `partition` | `""` | Slurm partition — **required**. |
| `job_name` | `"Job"` | Job name shown in the queue. |
| `num_of_node` | `1` | Number of nodes to request. |
| `tasks_per_node` | `1` | Tasks per node. |
| `cpus_per_task` | `1` | CPUs per task. |
| `gpus_per_task` | `0` | GPUs per task. |
| `gpus_per_node` | `None` | GPUs per node (overrides `gpus_per_task` when set). |
| `mem` | `""` | Memory request, e.g. `"16GB"` (blank = node default). |
| `timeout_min` | no limit | Wall-clock limit in minutes. |
| `node_list` / `node_list_exclude` | `""` | Nodes to include / exclude. |
| `output_parent_path` / `output_folder` | `"./"` / `"slurm"` | Where Slurm logs and artifacts are written. |
| `setup` | `[]` | Shell commands run before the job (e.g. module loads, env vars). |
| `use_distributed_env` | `False` | Enable distributed launch orchestration. |
| `distributed_launch_command` | `""` | Launch command (required when distributed). |
| `pack_code` / `use_packed_code` | `False` | Snapshot the codebase / run from the snapshot. |
| `extra_params_kwargs` / `extra_submit_kwargs` / `extra_task_kwargs` | `{}` | Escape hatches for extra Slurm/submit/task options. |

`partition` is required and validated in `__post_init__`; distributed mode additionally requires `distributed_launch_command`. See the [full configuration docstring](slurmic/config.py) for every field.

## Documentation

Full docs (getting started, tutorials, API reference, code packing) are published from [`docs/`](docs/):
👉 **[jhliu17.github.io/slurmic](https://jhliu17.github.io/slurmic/)**

## Development

```bash
uv sync --locked        # install deps into .venv (matches CI)
uv run pytest           # run the test suite
uv tool run ruff check  # lint
uv build                # build the wheel
```

The test suite does **not** require a real Slurm cluster — it mocks `sbatch`/`srun` and drives execution manually. Tests run on Python 3.10–3.13; Ruff targets `py312` with a line length of 100.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
