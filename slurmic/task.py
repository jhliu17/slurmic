import os
import shlex
import submitit
import subprocess

from typing import Union
from dataclasses import dataclass
from .config import SlurmConfig
from .accelerator.utils import nvidia_smi_gpu_memory_stats_str
from .utils import reconstruct_command_line


class Task:
    """The base class for all tasks that will be run on Slurm. Especially useful for
    distributed tasks that need to set up the distributed environment variables.

    Args:
        argv (list[str]): the command line arguments to run the task. This will be passed to the command method to reconstruct the command line.
        slurm_config (SlurmConfig): the Slurm configuration to use for the task.
        verbose (bool, optional): whether to print verbose output. Defaults to False.
    """

    def __init__(self, argv: list[str], slurm_config: SlurmConfig, verbose: bool = False):
        self.argv = argv
        self.slurm_config = slurm_config
        self.verbose = verbose

    def log(self, msg: str):
        """Log a message to the console if verbose is enabled.

        Args:
            msg (str): the message to log.
        """
        if not self.verbose:
            return
        print(msg)

    def command(self) -> str:
        """Return the command to run the task. This method should be implemented by
        subclasses to return the actual command line to run the task.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Returns:
            str: the command to run the task.
        """
        raise NotImplementedError

    def checkpoint(self):
        """Return a checkpoint for the task. This is used to save the state of the task."""
        return submitit.helpers.DelayedSubmission(self)


@dataclass
class DistributedTaskConfig:
    """Configuration for distributed tasks. This is used to set up the distributed environment
    variables for PyTorch distributed training.

    Args:
        num_processes (int): The total number of processes to run across all machines.
        num_machines (int): The number of machines to run the task on.
        machine_rank (int): The rank of the current machine in the distributed setup.
        main_process_ip (str): The IP address of the main process (rank 0)
            in the distributed setup.
        main_process_port (int): The port of the main process (rank 0)
            in the distributed setup.
    """

    # The number of processes to run in total across all machines.
    num_processes: Union[int, str] = "$slurmic_num_processes"

    # The number of machines to run the task on.
    num_machines: Union[int, str] = "$slurmic_num_machines"

    # The rank of the current machine in the distributed setup.
    machine_rank: Union[int, str] = "$slurmic_machine_rank"

    # The IP address of the main process (rank 0) in the distributed setup.
    main_process_ip: str = "$slurmic_main_process_ip"

    # The port of the main process (rank 0) in the distributed setup.
    main_process_port: Union[int, str] = "$slurmic_main_process_port"

    def export_bash(self, output_folder: str):
        """Export the distributed environment variables to a bash script.
        This script can be sourced to set the environment variables for the distributed task.

        Args:
            output_folder (str): the folder to save the bash script to.
        """
        lines = ["#!/bin/bash"]
        for k, v in self.__dict__.items():
            lines.append(f"export slurmic_{k}={v}")
        with open(os.path.join(output_folder, "slurmic_distributed_env.sh"), "w") as f:
            f.write("\n".join(lines))


class PyTorchDistributedTask(Task):
    """A task that runs on Slurm and sets up the PyTorch distributed environment variables. It runs the command locally
    if in other modes.

    Args:
        launch_cmd (str): The command to launch the task.
        argv (list[str]): The command line arguments for the task.
        slurm_config (SlurmConfig): The Slurm configuration to use for the task.
        verbose (bool, optional): _description_. Defaults to False.

    References:
        https://github.com/huggingface/accelerate/issues/1239
        https://github.com/yuvalkirstain/PickScore/blob/main/trainer/slurm_scripts/slurm_train.py
        https://github.com/facebookincubator/submitit/pull/1703
    """

    def __init__(
        self,
        launch_cmd: str,
        argv: list[str],
        slurm_config: SlurmConfig,
        verbose: bool = False,
        **env_setup_kwargs,
    ):
        super().__init__(argv, slurm_config, verbose)
        self.launch_cmd = launch_cmd
        self.env_setup_kwargs = env_setup_kwargs

        # to be set up in the dist_set_up method
        self.dist_args: DistributedTaskConfig = DistributedTaskConfig()
        self.dist_env: Union[None, submitit.helpers.TorchDistributedEnvironment] = None

    def set_up_dist_env(self):
        """Set up the distributed environment variables for PyTorch distributed training."""

        self.log("running task on slurm")
        self.log("exporting PyTorch distributed environment variables")

        # prepare enviroment variables
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )

        # other setup
        env_setup = {}

        # set CUDA visible devices if slurm has scheduled GPUs otherwise use all GPUs (without setting
        # CUDA_VISIBLE_DEVICES)
        if self.slurm_config.mode == "slurm":
            env_setup.update(
                {"CUDA_VISIBLE_DEVICES": os.environ["SLURM_JOB_GPUS"]}
                if "SLURM_JOB_GPUS" in os.environ
                else {}
            )

        # other environment variables set by the user
        env_setup.update(self.env_setup_kwargs)
        self.log(f"Env setup: {env_setup}")

        # update environment variables
        os.environ.update(**env_setup)

        self.log(nvidia_smi_gpu_memory_stats_str())
        self.log(f"Master: {dist_env.master_addr}:{dist_env.master_port}")
        self.log(f"Rank: {dist_env.rank}")
        self.log(f"World size: {dist_env.world_size}")
        self.log(f"Local rank: {dist_env.local_rank}")
        self.log(f"Local world size: {dist_env.local_world_size}")
        self.log(
            f"Local rank {dist_env.local_rank}: CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
        )

        # set distributed arguments
        num_processes = (
            self.slurm_config.tasks_per_node
            * self.slurm_config.processes_per_task
            * self.slurm_config.num_of_node
        )
        machine_rank = dist_env.rank // self.slurm_config.tasks_per_node
        self.dist_args = DistributedTaskConfig(
            num_processes=num_processes,
            num_machines=self.slurm_config.num_of_node,
            machine_rank=machine_rank,
            main_process_ip=dist_env.master_addr,
            main_process_port=dist_env.master_port,
        )
        self.dist_env = dist_env

        return self.dist_args, self.dist_env

    def command(self) -> str:
        """Return the command to run the task. This method should be implemented by
        subclasses to return the actual command line to run the task.

        Returns:
            str: the command to run the task.
        """
        cmd = self.launch_cmd.format(**self.dist_args.__dict__)
        cmd += " " + reconstruct_command_line(self.argv)
        return cmd

    def __call__(self):
        # Set up distributed environment
        self.set_up_dist_env()

        # Job environment
        job_env = submitit.helpers.JobEnvironment()

        # Concrete run command
        cmd = self.command()

        # Export distributed environment variables only the global rank 0 process will run the command
        if self.dist_env.rank == 0:
            print(f"running command: {cmd}")
            if self.slurm_config.mode == "slurm":
                try:
                    # Export distributed environment variables to a bash script
                    # and the fn will be launched after the job is scheduled
                    self.dist_args.export_bash(shlex.quote(str(job_env.paths.folder)))
                except Exception as e:
                    print(f"failed to export distributed environment variables: {e}")
                    return -1
            elif self.slurm_config.mode == "local":
                cmd_list = shlex.split(cmd)
                return subprocess.Popen(cmd_list)
            else:
                # If not on slurm mode, we can just run the command directly
                # This is useful for local testing or when running on a single machine
                return os.system(cmd)

        return 0
