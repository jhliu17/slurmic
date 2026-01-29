# 🚂 slurmic

[![Pytest](https://github.com/jhliu17/SLURMIC/actions/workflows/pytest.yml/badge.svg)](https://github.com/jhliu17/SLURMIC/actions/workflows/pytest.yml) [![Documentation](https://github.com/jhliu17/SLURMIC/actions/workflows/docs.yml/badge.svg)](https://github.com/jhliu17/SLURMIC/actions/workflows/docs.yml)

`slurmic` is a package designed to provide seamless Python function execution on Slurm.

## Features

### Execute Python functions on Slurm just like local functions

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

### Easily manage job dependencies

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

### Mapping sequential jobs

```python
   jobs = run_on_slurm[slurm_config].map_array([1, 2, 8, 9], [3, 4, 8, 9])
   results = [job.result() for job in jobs]
   assert results == [4, 6, 16, 18]
```

### Distributed jobs

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

## Installation

slurmic is tested and supported on the following systems:

* Python 3.10-3.13
* Linux systems with Slurm installed

Install slurmic via pip

```bash
   pip install slurmic
```

## Development

### Development Installation

```bash
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

### Build Wheel

```bash
uv build
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
