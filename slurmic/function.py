import submitit
import copy

from submitit import Job
from typing import Any, Callable, Literal, Tuple, Union, Dict, List, Optional
from .config import SlurmConfig
from .core import SlurmBackend


class SlurmFunction:
    """The function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass)."""

    def __init__(
        self,
        submit_fn: Callable[..., Any],
        default_submit_fn_args: Optional[Tuple[Any]] = None,
        default_submit_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

        Args:
            submit_fn: function to be submitted to Slurm, defaults to None
            default_submit_fn_args: default args for submit_fn, defaults to ()
            default_submit_fn_kwargs: default known word args for submit_fn, defaults to {}

        Returns:
            the wrapped submit function with configured slurm paramters
        """
        self.engine = SlurmBackend(submit_fn, default_submit_fn_args, default_submit_fn_kwargs)

    def __create_copy(self) -> "SlurmFunction":
        return copy.copy(self)

    def is_configured(self) -> bool:
        """Whether the slurm function has been configured.

        Returns:
            True if the slurm function has been configured, False otherwise
        """
        return self.engine.is_configured()

    def is_distributed(self) -> bool:
        """Whether the slurm function is distributed.

        Returns:
            True if the slurm function is distributed, False otherwise
        """
        return self.engine.is_distributed()

    def get_executor(
        self,
    ) -> submitit.AutoExecutor:
        return self.engine.get_executor()

    def configure(
        self,
        slurm_config: SlurmConfig,
        slurm_params_kwargs: Optional[Dict[str, str]] = None,
        slurm_submit_kwargs: Optional[Dict[str, str]] = None,
        slurm_task_kwargs: Optional[Dict[str, str]] = None,
        system_argv: Optional[List[str]] = None,
        pack_code_include_fn: Optional[Callable[[str, str], bool]] = None,
        pack_code_exclude_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> "SlurmFunction":
        """Update the slurm configuration for the slurm function. A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

        **Exported Distributed Enviroment Variables**

        - ``SLURMIC_SLURM_HAS_BEEN_SET_UP`` is a special environment variable to indicate that the slurm has been set up.
        - After the set up, the distributed job will be launched and the following variables are exported:
            - ``num_processes``: int
            - ``num_machines``: int
            - ``machine_rank``: int
            - ``main_process_ip``: str
            - ``main_process_port``: int

        Args:
            slurm_config: SlurmConfig, the slurm configuration dataclass, defaults to None
            slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
            slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
            slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
            system_argv: the system arguments for the second launch in the distributed task (by default it will use the current system arguments `sys.argv[1:]`), defaults to None

        Returns:
            a new copy with configured slurm parameters
        """
        configured_slurm_function = self.__create_copy()
        configured_slurm_function.engine = self.engine.configure(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv,
            pack_code_include_fn,
            pack_code_exclude_fn,
        )
        return configured_slurm_function

    def __getitem__(self, slurm_config: Union[Dict[str, Any], Tuple[Any], Any]) -> "SlurmFunction":
        """Instantiate the slurm configuration for the slurm function. A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

        **Exported Distributed Enviroment Variables**

        - ``SLURMIC_SLURM_HAS_BEEN_SET_UP`` is a special environment variable to indicate that the slurm has been set up.
        - After the set up, the distributed job will be launched and the following variables are exported:
            - ``num_processes``: int
            - ``num_machines``: int
            - ``machine_rank``: int
            - ``main_process_ip``: str
            - ``main_process_port``: int

        Args:
            slurm_config: SlurmConfig, the slurm configuration dataclass

        Returns:
            the wrapped submit function with configured slurm paramters
        """
        configured_slurm_function = self.__create_copy()
        configured_slurm_function.engine = self.engine[slurm_config]
        return configured_slurm_function

    def __call__(self, *submit_fn_args, **submit_fn_kwargs) -> Union[Job, Any]:
        """Run the submit_fn with the given arguments and keyword arguments. The function is non-blocking in the mode of `slurm`, while other modes cause blocking. If there is no given arguments or keyword arguments, the default arguments and keyword arguments will be used.

        Args:
            submit_fn_args: arguments for the submit_fn
            submit_fn_kwargs: keyword arguments for the submit_fn

        Returns:
            Slurm Job or the return value of the submit_fn, depends on the submit mode

        Raises:
            Exception: if the submit_fn is not set up
        """
        return self.engine(*submit_fn_args, **submit_fn_kwargs)

    def submit(self, *submit_fn_args, **submit_fn_kwargs) -> Union[Job, Any]:
        """An alias function to ``__call__``.

        Args:
            submit_fn_args: arguments for the submit_fn
            submit_fn_kwargs: keyword arguments for the submit_fn

        Raises:
            Exception: if the submit_fn is not set up

        Returns:
            Slurm Job or the return value of the submit_fn
        """
        return self(*submit_fn_args, **submit_fn_kwargs)

    def map_array(
        self, *submit_fn_args, **submit_fn_kwargs
    ) -> Union[Job[Any], List[Job[Any]], Any]:
        """Run the submit_fn with the given arguments and keyword arguments. The function is non-blocking in the mode of `slurm`, while other modes cause blocking. If there is no given arguments or keyword arguments, the default arguments and keyword arguments will be used.

        Args:
            submit_fn_args: arguments for the submit_fn
            submit_fn_kwargs: keyword arguments for the submit_fn

        Raises:
            Exception: if the submit_fn is not set up

        Returns:
            Slurm Job or the return value of the submit_fn
        """
        return self.engine.map_array(*submit_fn_args, **submit_fn_kwargs)

    def on_condition(
        self,
        jobs: Union[Job, List[Job], Tuple[Job]],
        condition: Literal["afterany", "afterok", "afternotok"] = "afterok",
    ) -> "SlurmFunction":
        """Mark this job should be executed after the provided slurm jobs have been done. This function allows combining different conditions by multiple calling.

        Args:
            jobs: dependent jobs
            condition: run condition, defaults to "afterok"

        Returns:
            the function itself
        """
        configured_slurm_function = self.__create_copy()
        configured_slurm_function.engine = self.engine.on_condition(jobs, condition)
        return configured_slurm_function

    def afterok(self, *jobs: Job) -> "SlurmFunction":
        """Mark the function should be executed after the provided slurm jobs have been done.

        Returns:
            the new slurm function with the condition
        """
        return self.on_condition(list(jobs), "afterok")

    def afterany(self, *jobs: Job) -> "SlurmFunction":
        """Mark the function should be executed after any one of the provided slurm jobs has been done.

        Returns:
            the new slurm function with the condition
        """
        return self.on_condition(list(jobs), "afterany")

    def afternotok(self, *jobs: Job) -> "SlurmFunction":
        """Mark the function should be executed after any one of the provided slurm jobs has been failed.

        Returns:
            the new slurm function with the condition
        """
        return self.on_condition(list(jobs), "afternotok")
