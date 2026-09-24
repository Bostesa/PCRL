"""Reproduce the aggregate-only, pre-outcome 2018 household-role census."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import data
from experiments.pcrl_adaptive_release_v1.roles import role_of, summarize_roles


def main() -> None:
    index = data.index(Path.cwd())
    prepared = {anchor: data.load_prepared(index, anchor) for anchor in (0, 1, 2)}
    summary = summarize_roles(prepared)
    historical_teacher_households: set[str] = set()
    for item in prepared.values():
        rep = np.asarray(item['ctx']['pools']['representation_fit']['households'])
        for name in ('teacher_fit', 'teacher_internal_validation'):
            historical_teacher_households.update(str(h) for h in rep[item['roles'][name]])
    outer_households = {
        str(h) for item in prepared.values() for pool in item['ctx']['pools'].values()
        for h in pool['households'] if role_of(h) == 'outer_assessment'
    }
    summary['historical_teacher_households_in_outer'] = len(
        outer_households & historical_teacher_households)
    summary['historical_teacher_overlap_limitation'] = (
        'Frozen historical teacher fitted some outer-assigned households upstream; '
        'outer is locked development assessment, not wholly unseen confirmation.')
    path = Path('results/pcrl_adaptive_release_v1/DATA_ROLE_COUNTS.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != summary:
            raise ValueError('existing role census differs; preserve and investigate')
    else:
        path.write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'outer_households': len(outer_households),
                      'historical_teacher_overlap': summary['historical_teacher_households_in_outer'],
                      'global_role_overlap': summary['global_role_overlap']}))


if __name__ == '__main__':
    main()
