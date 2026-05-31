Distributed Training with PyTorch
#################################

The distributed training with slurmic has been extensively tested with the following features:


**Single Node**

- ✅ Single-process training
- ✅ DDP training
- ✅ FSDP training (e.g., FSDP2 with Accelerate)

We haven't tested the following features yet, but they are expected to work:

**Multiple Nodes**

- ❓ Multi-node DDP training
- ❓ Multi-node FSDP training


Below is an example shows how to submit a distributed training job with slurmic and use ``accelerate`` for distributed training.


Training function
=================

Here is an example of using ``accelerate`` to conduct a distributed training. Please refer to the ``accelerate`` `documentation <https://huggingface.co/docs/accelerate/index>`_ for more information.

.. code-block:: python
    :caption: main.py

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from slurmic import slurm_fn, SlurmConfig

    @slurm_fn
    def main(config: SlurmConfig) -> None:
        accelerator = Accelerator()
        set_seed(2024)
        model, optimizer, training_dataloader, scheduler = accelerator.prepare(
            model, optimizer, training_dataloader, scheduler
        )

        ...  # your training loop here



Distributed launch command
==========================

To launch a distributed job, it is necessary to set up the ``use_distributed_env`` and ``distributed_lanch_command`` in the ``SlurmConfig`` function.


Exported variables
------------------

The ``distributed_launch_command`` is a command that is used to launch the distributed job. There are several arguments exposed by the slurmic which are useful to set up the distributed job. The arguments are as follows:

- ``num_processes``: int
- ``num_machines``: int
- ``machine_rank``: int
- ``main_process_ip``: str
- ``main_process_port``: int

Set up the entry point
----------------------

Here is an example of how to use the ``distributed_launch_command`` in the ``SlurmConfig`` function. The command is used to launch the distributed job with ``accelerate``. The command is as follows:

.. code-block:: python
    :caption: distributed_launch_command in SlurmConfig

    accelerate launch --config_file CONFIG_FILE --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} main.py

The CONFIG_FILE is a ``accelerate`` configuration file that is used to set up the distributed type. It is worth noting that the ``main.py`` is the main file that is used to run the training. Based on the distributed training type, one can properly set up the required ``gpus_per_node`` and ``processes_per_task`` in the ``SlurmConfig`` function.

.. code-block:: python
    :caption: main.py
    :emphasize-lines: 10,11,14,15,18

    ...

    if __name__ == "__main__":
        slurm_config = SlurmConfig(
            mode="slurm",
            partition="PARITITION",
            job_name="JOB_NAME",
            tasks_per_node=1,
            cpus_per_task=8,
            gpus_per_node=4,
            processes_per_task=4,
            mem="192GB",
            node_list="NODE_LIST",
            use_distributed_env=True,
            distributed_launch_command="accelerate launch --config_file CONFIG_FILE --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} main.py",
        )

        main[slurm_config](config)

