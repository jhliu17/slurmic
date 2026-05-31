# Getting Started

## Installation

slurmic supports Python 3.10-3.13 and is tested on Linux systems with Slurm installed.

Install slurmic via pip:

```bash
pip install slurmic
```

## Quick Guide

The examples below walk through the core building blocks. Each one shows **what the
function does** and **when you would reach for it**.

### Execute Python functions on Slurm just like local functions

Decorate a plain Python function with `@slurm_fn`, attach a `SlurmConfig` with the
`[config]` syntax, then call it like a normal function. The call returns immediately
with a job handle; `job.result()` blocks until the job finishes and returns the
function's return value (or re-raises the exception it raised on the cluster).

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

   job = run_on_slurm[slurm_config](1, b=2) # job is submitted to slurm
   result = job.result() # block and get the result => 3
```

- **`@slurm_fn`** — marks a function as Slurm-runnable. The function body still runs
  unchanged; slurmic only takes over *where* and *how* it executes.
- **`SlurmConfig`** — declares the resources and run mode for a submission (partition,
  CPUs/GPUs, memory, etc.). The same function can be reused with different configs.
- **`run_on_slurm[slurm_config](...)`** — binds the config to the function and submits
  the job with the given arguments, returning a non-blocking handle.
- **`job.result()`** — waits for completion and returns the result.

**Use this when** you want a single function to run on the cluster while keeping your
code as if it were a local call — no boilerplate sbatch scripts, no manual
serialization, no polling.

:::{hint}
`mode` controls *where* it runs. Use `mode="slurm"` for the cluster, `mode="local"` to
run as a local subprocess, `mode="debug"` to drop into `pdb` on a breakpoint, and
`mode="run"` to call the function directly in-process. This lets you develop and debug
locally, then flip a single field to scale out.
:::

### Easily manage job dependencies

`.on_condition(job)` returns a new configured function whose job will only start once
the dependency job has finished. This builds a dependency chain on the scheduler itself,
so you can submit the whole pipeline up front without keeping your Python process
blocked between stages.

```python
   jobs = []

   # job1 is submitted to slurm directly
   job1 = run_on_slurm[slurm_config](10, 2)
   jobs.append(job1)

   # fn2 must be executed after job1 is finished
   fn2 = run_on_slurm[slurm_config].on_condition(job1)
   job2 = fn2(7, 12)
   jobs.append(job2)

   results = [job.result() for job in jobs]  # This will block until all jobs are finished
   assert results == [12, 19]
```

- **`.on_condition(job1)`** — declares a scheduler-level dependency: `job2` is queued
  immediately but held until `job1` completes.

**Use this when** you have a multi-stage pipeline (e.g. preprocess → train → evaluate)
where later stages depend on earlier ones. Submitting all stages at once lets Slurm
manage ordering and start each stage the moment its prerequisite is done — without your
script having to stay alive and orchestrate the hand-offs.

### Mapping sequential jobs

`.map_array(*arg_lists)` submits one job per set of arguments as a Slurm **job array**,
returning a list of handles. The argument lists are zipped together, so the call below
runs `run_on_slurm(1, 3)`, `run_on_slurm(2, 4)`, `run_on_slurm(8, 8)`, and
`run_on_slurm(9, 9)`.

```python
   jobs = run_on_slurm[slurm_config].map_array([1, 2, 8, 9], [3, 4, 8, 9])
   results = [job.result() for job in jobs]
   assert results == [4, 6, 16, 18]
```

- **`.map_array(...)`** — fans the same function out over many argument sets in a single
  array submission, which is lighter on the scheduler than submitting each job
  individually and lets the cluster run them in parallel.

**Use this when** you need to sweep the same computation over many inputs —
hyperparameter sweeps, per-seed runs, per-shard data processing, or batch inference.

### Distributed jobs

For multi-process / multi-GPU training, set `use_distributed_env=True` and provide a
`distributed_launch_command`. slurmic sets up the distributed environment (rank, world
size, master address/port) and then runs your launch command, substituting the
`{...}` placeholders with the values it computed.

```python
   # distributed launch command by accelerate as an example
   slurm_config = SlurmConfig(
         mode="slurm",
         cpus_per_task=8,
         gpus_per_node=4,
         use_distributed_env=True,
         distributed_launch_command="accelerate launch --config_file CONFIG_FILE --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} main.py",  # main.py is the entry of the distributed job
   )

   main[slurm_config](config)
```

- **`use_distributed_env=True`** — tells slurmic to allocate and export the distributed
  rendezvous information across the requested nodes/GPUs.
- **`distributed_launch_command`** — the command slurmic runs on each node once the
  environment is ready. The placeholders `{num_processes}`, `{num_machines}`,
  `{machine_rank}`, `{main_process_ip}`, and `{main_process_port}` are filled in
  automatically, so you can plug in launchers like `accelerate`, `torchrun`, or
  `deepspeed`.

**Use this when** a single process/GPU isn't enough and you need data- or
model-parallel training across multiple GPUs or nodes, but you don't want to hand-write
the rendezvous wiring for every job.

:::{seealso}
[Code Packing](code_packing.md) — snapshot your source tree so long-running or queued
jobs run against a frozen copy of the code.
:::
