"""Coordinator-run matched task-only control for family B's coarse-code construction (a separate module so the investigator's
amd/fam_b.py is not edited). It calls amd.fam_b.fit with the replication's TASK_REFIT arm (the identical leaf-correction loop
with the privacy terms disabled: same teacher routing, same 800 updates per recipient, same capacity) in place of the
privacy-trained JOINT_TREE / LOCAL_TREE arms:

    occupation = TASK_REFIT E20 occupation output;
    income     = h(c), c = 6 decision-aligned bins of the TASK_REFIT E20 income log-odds (label-free MODEL_FIT thresholds),
                 h = L2 logistic regression on one-hot c fitted on all of CALIBRATION (income labels only);
i.e. B3CTL_UNCOUPLED_MIX50_C6A with every privacy-trained component replaced by its task-only counterpart.

    python -m amd.ctl_taskonly --seed S
"""
from __future__ import annotations
import argparse

from . import fam_b

CONFIG = 'B3CTL_TASKONLY_TR_C6A'


def main(seed):
    fam_b.fit(seed, CONFIG, 'TASK_REFIT', 'none', 6, 'aligned', 1.0,
              'coordinator-run matched task-only control of the B3 coarse-code twin: TASK_REFIT occupation + uncoupled h(6 aligned bins of TASK_REFIT income log-odds)',
              stat='bins', stat_source='TASK_REFIT', alpha=0.0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
    main(ap.parse_args().seed)
