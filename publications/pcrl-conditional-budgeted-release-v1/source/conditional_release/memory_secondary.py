"""Explicit low-memory adapter for the unchanged secondary study runner.

Only materialization of task losses and actual-code aggregation changes. No
support, optimization objective, costs, guards, selection or solver changes.
The original secondary runner remains preserved and separately importable.
"""
from __future__ import annotations
import argparse
from . import secondary,memory_lp


def run(config):
    secondary.bank_action_costs=memory_lp.LazyTaskCosts
    secondary.bank_expected_rows=memory_lp.expected_rows
    secondary.solve_protection=memory_lp.solve_protection
    secondary.direct_bayes=memory_lp.direct_bayes
    return secondary.run(config)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();run(args.config)
