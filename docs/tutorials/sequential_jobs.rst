Slurm Jobs with slurmic
######################

This example shows how to submit a series of jobs with `slurmic`.

Configuration of SlurmFunction
==============================

The ``slurm_fn`` decorator converts a Python function into a SlurmFunction. The SlurmFunction can be used to submit jobs to the Slurm cluster. Each configuration of a SlurmFunction will create a new copy of the function. This is useful when you want to run the same function with different configurations.

.. code-block:: python
    :caption: A worker function with slurm settings

    import time
    from slurmic.slurm import SlurmConfig, slurm_fn

    slurm_settings = SlurmConfig(
        mode="slurm",
        job_name="JOB_NAME",
        partition="PATITION",
        num_of_node=1,
        tasks_per_node=1,
        gpus_per_task=0,
        cpus_per_task=1,
        mem="2GB",
        timeout_min=10,
    )


    @slurm_fn
    def work_fn(a, b):
        time.sleep(a + b)
        return a + b


Submitting and blocking to get the result
=========================================

This example demonstrates how to submit a job to the Slurm cluster and block until the job is finished. The ``SlurmFunction`` is used to submit the job, and the ``result()`` method is used to get the result of the job.

.. code-block:: python
    :caption: A worker function return the result

    fn = work_fn[slurm_settings]

    job = fn(1, 2) # The job is submitted to the Slurm cluster
    result = job.result()  # This will block execution until the job is finished
    print(result)
    assert result == 3


.. important::

    A ``SlurmFunction`` executed on the Slurm cluster is non-blocking. The ``result()`` method is used to get the result of the job. The ``result()`` method will block until the job is finished.


Map array
=========

You can even map an array of values to a worker function. This will create a job for each element in the array and return a list of jobs. Each job can be executed in parallel on the Slurm cluster.

.. code-block:: python
    :caption: A worker function maps an array

    fn = work_fn[slurm_settings]

    # This will create a job for each element in the array
    # and return a list of jobs.
    jobs = fn.map_array([1, 2, 8, 9], [3, 4, 8, 9])
    results = [job.result() for job in jobs]
    print(results)
    assert results == [4, 6, 16, 18]


Dependency between jobs
=======================

You can create dependencies between jobs using the ``on_condition()``, ``afterok()``, ``afternotok``, and ``afterany()`` methods. This allows you to run jobs sequentially or in parallel based on the completion of other jobs. Please check the `SlurmFunction` documentation for more details on these methods.

.. code-block:: python
    :caption: A worker function runs sequentially

    jobs = []
    job1 = work_fn[slurm_settings](10, 2)
    jobs.append(job1)

    fn2 = work_fn[slurm_settings]
    fn2.on_condition(job1)
    job2 = fn2(7, 12)
    jobs.append(job2)

    fn3 = work_fn[slurm_settings]
    assert fn2 is not fn3  # Each configuration creates a new copy of the function

    fn3.afterany(job1, job2)
    job3 = fn3(2, 30)
    jobs.append(job3)

    results = [job.result() for job in jobs]  # This will block until all jobs are finished
    assert results == [12, 19, 32]
