"""Independent verification of the task-headroom protected-output campaign.

Own implementation of routing, metrics, AUC, bootstrap, readers and decision
rules. Imports nothing from task_headroom.* or task_conditioned_representation.*
except task_headroom.evaluation_queue.queue_config (queue binding infrastructure).
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
