"""Nomination frontier when no configuration qualifies (SELECTION_RULE section 6): score (minimum over the six required comparators
of the 8-cell mean pair benefit) against the minimum whole-release reserve ratio, for every admissible two-seed configuration, with
the Pareto-efficient set marked. Reads NOMINATION.json only (no label read).

    python -m ohn.frontier   -> results/<campaign>/NOMINATION_FRONTIER.csv
"""
from __future__ import annotations
import csv
import json

from . import common as C


def main():
    N = json.loads((C.RESULTS / 'NOMINATION.json').read_text())
    rows = []
    for r in N['configurations']:
        if r.get('score_min_required_mean_benefit') is None or not r.get('reserve'):
            continue
        pc = r['per_comparator']
        binding = min(pc, key=lambda x: pc[x]['benefit'].get('mean', float('inf')))
        rows.append({'configuration': r['config_id'], 'kind': r.get('kind'), 'label': r['label'],
                     'score': round(r['score_min_required_mean_benefit'], 5), 'score_binding_comparator': binding,
                     'min_whole_release_ratio': round(r['reserve']['min_ratio'], 3), 'reserve_status': r['reserve']['status'],
                     'reserve_binding': r['reserve'].get('binding_condition'),
                     'occupation_recipient_max': round(max(pc[x]['recipient']['occupation']['max'] for x in pc), 4),
                     'income_recipient_max': round(max(pc[x]['recipient']['income']['max'] for x in pc), 4),
                     'reader_stability': r['outcomes']['reader_stability'], 'material_target': r['outcomes']['material_target']})
    for a in rows:                                   # Pareto: no other row at least as good on both and better on one
        a['pareto_efficient'] = not any(b['score'] >= a['score'] and b['min_whole_release_ratio'] >= a['min_whole_release_ratio']
                                        and (b['score'] > a['score'] or b['min_whole_release_ratio'] > a['min_whole_release_ratio'])
                                        for b in rows)
    rows.sort(key=lambda x: -x['score'])
    with open(C.RESULTS / 'NOMINATION_FRONTIER.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for x in rows:
        print(f"{x['configuration']:26s} score {x['score']:+.4f} ({x['score_binding_comparator']}) ratio {x['min_whole_release_ratio']:.3f} "
              f"{'PARETO' if x['pareto_efficient'] else ''}")


if __name__ == '__main__':
    main()
