import os
import sys

from dataclasses import dataclass, field, replace
from typing import List, Literal, Dict, Optional


@dataclass
class SlurmConfig:
    """
    Configuration class for SLURM job submission and execution.

    Args:
        mode (Literal["run", "debug", "local", "slurm"]): Running mode for the job. Options include:
            "run" (default, directly run the function), "debug" (run debugging which will involve pdb if it reachs a breakpoint), "local" (run the job locally by subprocess, without gpu allocations and CUDA_VISIBLE_DEVICES cannot be set), or "slurm" (run the job on a SLURM cluster).

        job_name (str): The name of the SLURM job. Default is 'Job'.

        partition (str): The name of the SLURM partition to use. Default is ''.

        output_parent_path (str): The parent directory path for saving the slurm folder. Default is './'.

        output_folder (str): The folder name where SLURM output files will be stored. Default is 'slurm'.

        node_list (str): A string specifying the nodes to use. Leave blank to use all available nodes. Default is an empty string.

        node_list_exclude (str): A string specifying the nodes to exclude. Leave blank to use all nodes in the node list. Default is an empty string.

        num_of_node (int): The number of nodes to request. Default is 1.

        tasks_per_node (int): The number of tasks to run per node. Default is 1.

        gpus_per_task (int): The number of GPUs to request per task. Default is 0.

        cpus_per_task (int): The number of CPUs to request per task. Default is 1.

        gpus_per_node (int): The number of GPUs to request per node. If this is set, `gpus_per_task` will be ignored. Default is None.

        mem (str): The amount of memory (GB) to request. Leave blank to use the default memory configuration of the node. Default is an empty string.

        timeout_min (int): The time limit for the job in minutes. Default is `sys.maxsize` for effectively no limit.

        stderr_to_stdout (bool): Whether to redirect stderr to stdout. Default is False.

        setup (List[str]): A list of environment variable setup commands. Default is an empty list.

        pack_code (bool): Whether to pack the codebase before submission. Default is False.

        use_packed_code (bool): Whether to use the packed code for execution. Default is False.

        code_root (str): The root directory of the codebase, which will be used by the code packing. Default is the current directory (``.``).

        code_file_suffixes (List[str]): A list of file extensions for code files to be included when packing. Default includes ``.py``, ``.sh``, ``.yaml``, and ``.toml``.

        exclude_code_folders (List[str]): A list of folder names relative to `code_root` that will be excluded from packing. Default excludes 'wandb', 'outputs', and 'datasets'.

        use_distributed_env (bool): Whether to use a distributed environment for the job. Default is False.

        distributed_env_task (Literal["torch"]): The type of distributed environment task to use. Currently, only "torch" is supported. Default is "torch".

        processes_per_task (int): The number of processes to run per task. This value is not used by SLURM but is relevant for correctly set up distributed environments. Default is 1.

        distributed_launch_command (str): The command to launch distributed environment setup, using environment variables like ``{num_processes}``, ``{num_machines}``, ``{machine_rank}``, ``{main_process_ip}``, ``{main_process_port}``. Default is an empty string.

        extra_params_kwargs (Dict[str, str]): Additional parameters for the SLURM job as a dictionary of key-value pairs. Default is an empty dictionary.

        extra_submit_kwargs (Dict[str, str]): Additional submit parameters for the SLURM job as a dictionary of key-value pairs. Default is an empty dictionary.

        extra_task_kwargs (Dict[str, str]): Additional task parameters for the SLURM job as a dictionary of key-value pairs. Default is an empty dictionary.
    """

    # running mode
    mode: Literal["run", "debug", "local", "slurm"] = "run"

    # slurm job name
    job_name: str = "Job"

    # slurm partition name
    partition: str = ""

    # slurm output parent path
    output_parent_path: str = "./"

    # slurm output folder name
    output_folder: str = "slurm"

    # node list string (leave blank to use all nodes)
    node_list: str = ""

    # node list string to be excluded (leave blank to use all nodes in the node list)
    node_list_exclude: str = ""

    # number of nodes to request
    num_of_node: int = 1

    # tasks per node
    tasks_per_node: int = 1

    # number of gpus per task to request
    gpus_per_task: int = 0

    # number of cpus per task to request
    cpus_per_task: int = 1

    # number of gpus per node to request (if this is set, gpus_per_task will be ignored)
    gpus_per_node: Optional[int] = None

    # memory (GB) to request (leave black to use default memory configurations in the node)
    mem: str = ""

    # time out min
    timeout_min: int = sys.maxsize

    # whether to redirect stderr to stdout
    stderr_to_stdout: bool = False

    # environment variables setup command
    setup: List[str] = field(default_factory=list)

    # whether to pack code
    pack_code: bool = False

    # use packed code to run
    use_packed_code: bool = False

    # code root
    code_root: str = "."

    # code file extensions
    code_file_suffixes: list[str] = field(default_factory=lambda: [".py", ".sh", ".yaml", ".toml"])

    # exclude folders (relative to the code root)
    exclude_code_folders: list[str] = field(
        default_factory=lambda: ["wandb", "outputs", "datasets"]
    )

    # whether to use distributed environment
    use_distributed_env: bool = False

    # distributed enviroment task
    distributed_env_task: Literal["torch"] = "torch"

    # processes per task (this value is not used by slurm, but in the distributed environment)
    processes_per_task: int = 1

    # distributed launch command (this will be called after the distributed enviroment is set up)
    # the following environment variables are available:
    #   num_processes: int
    #   num_machines: int
    #   machine_rank: int
    #   main_process_ip: str
    #   main_process_port: int
    # use braces to access the environment variables, e.g. {num_processes}
    distributed_launch_command: str = ""

    # extra slurm job parameters
    extra_params_kwargs: Dict[str, str] = field(default_factory=dict)

    # extra slurm submit parameters
    extra_submit_kwargs: Dict[str, str] = field(default_factory=dict)

    # extra slurm task parameters
    extra_task_kwargs: Dict[str, str] = field(default_factory=dict)

    def _configuration_check(self):
        # check partition
        if self.partition == "":
            raise ValueError("partition must be set")

        # check distributed enviroment task
        if self.use_distributed_env and self.distributed_launch_command == "":
            raise ValueError(
                "distributed_launch_command must be set when use_distributed_env is True"
            )

    def __post_init__(self):
        # check configuration
        self._configuration_check()

        # normalize the output folder
        output_folder_suffix = ""
        if self.mode != "slurm":
            output_folder_suffix = f"_{self.mode}"
        if self.output_folder.endswith("slurm"):
            self.output_folder = f"{self.output_folder}{output_folder_suffix}"
        else:
            self.output_folder = os.path.join(self.output_folder, f"slurm{output_folder_suffix}")

        # output path
        self.output_path: str = os.path.join(self.output_parent_path, self.output_folder)

    def set_output_path(self, output_parent_path: str) -> "SlurmConfig":
        """Set output path and date for the slurm job.

        Args:
            output_parent_path (str): The parent path for the output.

        Returns:
            SlurmConfig: The updated SlurmConfig instance.
        """
        new_config = replace(
            self,
            output_parent_path=output_parent_path,
        )
        return new_config


SlurmArgs = SlurmConfig
