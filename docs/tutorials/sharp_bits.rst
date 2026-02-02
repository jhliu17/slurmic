‼️ The Sharp Bits of slurmic
########################

Although the slurmic library is designed to be simple and easy to use, there are some sharp bits that you should be aware of. This section covers some of the sharp bits that you should be aware of when using the slurmic library.

‼️ Pure functions
=================

The submitted function should be a pure function. This means that the function should not have any side effects.

.. tip::

   To be a pure function, the function should satisfy the following conditions:

   - The function should not modify any global variables or modify any external state.
   - The function should only depend on the input arguments and should return the output based on the input arguments.


‼️ Stateless pending jobs
=========================

When you submit a job to the Slurm cluster, the job is pending until it is executed. If you make changes to the code after submitting the job, the job will be executed with the latest code version, not the code version that was submitted.

Let's say we submitted a job with code version 1.0 and the job is pending due to resource constraints. After that, we made some changes to the code and get the code version 1.1. Now, if the job is executed, it would be executed with the latest code version 1.1 not with the code version 1.0.

.. tip::
   To work around this issue, you can save the code in submission and execute the code based on the saved code. This way, you can ensure that the code is executed based on the code version you submitted. This can be achieved by setting ``pack_code=True`` and ``use_packed_code=True`` in the ``SlurmConfig`` function.
