import os
import torch
import time
import accelerate
import contextlib
import typing as tp

from dataclasses import dataclass
from slurmic import SlurmConfig, slurm_fn
from submitit.core import test_core, submission
from submitit.slurm.slurm import SlurmJob


@contextlib.contextmanager
def mocked_slurm() -> tp.Iterator[test_core.MockedSubprocess]:
    mock = test_core.MockedSubprocess(known_cmds=["srun"])
    try:
        with mock.context():
            yield mock
    finally:
        # Clear the state of the shared watcher
        SlurmJob.watcher.clear()
        # Submitting marks these env vars on the current process; clear them so
        # they don't leak into the next test (the sentinel in particular flips
        # the distributed second-launch dispatch).
        os.environ.pop("SLURMIC_SLURM_HAS_BEEN_SET_UP", None)
        os.environ.pop("SLURMIC_SLURM_PACKED_CODE", None)


def get_slurm_config(output_path, is_distributed: bool = False):
    slurm_config = None
    if not is_distributed:
        slurm_config = SlurmConfig(
            mode="slurm",
            job_name="test_slurm",
            partition="zhanglab.p",
            node_list="galaxy",
            num_of_node=1,
            tasks_per_node=1,
            gpus_per_task=0,
            cpus_per_task=1,
            mem="2GB",
            timeout_min=10,
            pack_code=True,
            code_root="./",
            use_packed_code=True,
            exclude_code_folders=["wandb", "outputs", "tests", "datasets", ".venv"],
            use_distributed_env=False,
        )
    else:
        slurm_config = SlurmConfig(
            mode="slurm",
            job_name="test_slurm",
            partition="zhanglab.p",
            node_list="galaxy",
            num_of_node=1,
            tasks_per_node=1,
            gpus_per_task=2,
            cpus_per_task=1,
            mem="10G",
            timeout_min=10,
            pack_code=True,
            code_root="./",
            use_packed_code=True,
            exclude_code_folders=["wandb", "outputs", "tests", "datasets", ".venv"],
            use_distributed_env=True,
            processes_per_task=2,
            distributed_launch_command="accelerate launch --config_file tests/distributed.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m tests.test_slurm",
        )

    slurm_config = slurm_config.set_output_path(output_path)
    return slurm_config


@dataclass
class WorkerTest:
    name: str

    @slurm_fn
    def run(self, a: int, b: int):
        print("My name is:", self.name)
        time.sleep(a + b)
        return a + b


def worker_test(sleep_time: int = 30):
    print(torch.__file__)
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print("I am the main process")
        print(torch.cuda.device_count())
        a = torch.randn(1000, 1000).to(device)
        time.sleep(sleep_time)
    else:
        print("I am a worker process")
        print(torch.cuda.device_count())
        a = torch.randn(1000, 1000).to(device)
        time.sleep(sleep_time)

    del a


@slurm_fn
def work_fn(a, b):
    """a demo function to test slurm"""
    print(torch.__file__)
    print("PYTHONPATH", os.environ.get("PYTHONPATH"))
    time.sleep(5)
    return a + b


def test_job_array_slurm_function(tmp_path):
    with mocked_slurm() as mock:
        slurm_settings = get_slurm_config(f"{tmp_path}/outputs/", is_distributed=False)
        fn = work_fn[slurm_settings]

        job = fn(1, 2)
        with mock.job_context(job.job_id):
            submission.process_job(job.paths.folder)
        result = job.result()
        print(result)
        assert result == 3

        jobs = fn.map_array([1, 2, 8, 9], [3, 4, 8, 9])
        for job in jobs:
            with mock.job_context(job.job_id):
                submission.process_job(job.paths.folder)
        results = [job.result() for job in jobs]
        print(results)
        assert results == [4, 6, 16, 18]


def test_sequential_jobs(tmp_path):
    with mocked_slurm() as mock:
        slurm_settings = get_slurm_config(f"{tmp_path}/outputs/", is_distributed=False)

        jobs = []
        job1 = work_fn[slurm_settings](2, 2)
        jobs.append(job1)

        fn1 = work_fn[slurm_settings]
        fn1.on_condition(job1)
        job2 = fn1(7, 2)
        jobs.append(job2)

        fn2 = work_fn[slurm_settings]
        assert fn1 is not fn2

        fn2.afterany(job1, job2)
        job3 = fn2(2, 3)
        jobs.append(job3)

        for job in [job1, job2, job3]:
            with mock.job_context(job.job_id):
                submission.process_job(job.paths.folder)

        results = [job.result() for job in jobs]
        assert results == [4, 9, 5]


def test_class_slurm_function(tmp_path):
    with mocked_slurm() as mock:
        worker = WorkerTest("test_worker")
        slurm_settings = get_slurm_config(f"{tmp_path}/outputs/", is_distributed=False)
        job = worker.run[slurm_settings](worker, 2, 1)
        with mock.job_context(job.job_id):
            submission.process_job(job.paths.folder)
        result = job.result()
        assert result == 3


@slurm_fn
def distributed_fn(*args, **kwargs):
    """a demo function to test slurm

    :param args: argument settings
    """
    print(args, kwargs)
    worker_test(30)
    return args, kwargs


# NOTE: the distributed-job orchestration is covered in tests/test_distributed.py.
# A real distributed run cannot be exercised under the mock (no scheduler runs the
# generated sbatch script, no GPU), so those tests verify the slurmic-specific logic
# — sbatch script generation, the distributed-env math, and the sentinel dispatch.


if __name__ == "__main__":
    worker_test()
