"""Independent verification of the frozen-tree control-completion campaign.

Written from the lock texts and the stored file formats only. Imports numpy, scipy, sklearn and torch,
plus tree_completion.evaluation_queue.queue_config solely for queue binding and its verify() to open files.
No metric, inference, bootstrap, influence or power code from tree_completion.*, task_headroom.* or
task_conditioned_representation.* is imported.
"""
