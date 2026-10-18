"""Index of the reader-robustness machine-readable outputs (label-free; reads only this role's JSON outputs and the queue ledger).

    python3 rr_index.py --final-tag RR_FINAL --probe RR_PROBE_A --probe RR_PROBE_B1 ... --split RR_SPLIT3_S0 --lcfix RR_LCFIX
Writes roles/reader_robustness/RR_INDEX.json: every output file with its purpose and SHA-256, the queue units of this role with their
charged CPU, and a few headline numbers copied from the outputs (for cross-checking the text report; no new computation).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from ohn import common as C          # noqa: E402

OUT = C.RESULTS / 'roles' / 'reader_robustness'
PURPOSE = {
    '__STABILITY': 'task 2: frozen stability rule per (candidate, comparator), spread components, sign changes, conditional intervals, split sharing',
    '__WINNERS': 'task 3: VALIDATION winners per release, view and cell; near ties; recipe families; selection optimism',
    '__PAIR_FAMILY': 'task 3: recipe family / access of candidate and comparator pair winners per cell',
    '__INCOME_NOISE': 'task 3: near-chance income-view noise and the per-cell recipient gate false-failure analysis',
    '__LEARNING_CURVES': 'task 1: mean SCORE AUC and mean pair benefit by FIT fraction with the frozen shrink flag',
    '__PROBE_SUMMARY': 'task 4: diagnostic probe readers (MLP_BIG, GB_LONG, KNN_LOGP) and fixed frozen recipes; orderings',
    '__PROBE_PAIRS': 'task 4: paired benefits / recipient differences per reading scheme (CSV)',
    '__LC_FIXED_RECIPE': 'task 1: learning curves per fixed frozen recipe from the cached grid readers (no fit)',
    '__SPLIT3_SUMMARY': 'coordinator request: third household split R3 (tag OHN1_devgrid_R3_diag) next to R1 / R2',
    '__SPLIT3_PAIRS': 'coordinator request: R1 / R2 / R3 split means per pair and scheme (CSV)',
    'GRID_SUMMARY_': 'task 1: official ohn.grid_summary of a learning-curve bank (labels and frozen shrink flags)',
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--final-tag', required=True)
    a = ap.parse_args(argv)
    files = {}
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name != 'RR_INDEX.json':
            rel = str(p.relative_to(OUT))
            files[rel] = {'sha256': C.sha(p), 'bytes': p.stat().st_size,
                          'purpose': next((v for k, v in PURPOSE.items() if k in p.name), 'code / spec / check')}
    units = []
    for line in (C.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e.get('event') == 'unit_end' and str(e.get('unit_id', '')).startswith('OHN_RR_'):
            units.append({'unit_id': e['unit_id'], 'charged_cpu_s': e['charged_cpu_s'], 'returncode': e['returncode'], 'utc': e['utc']})
    st = json.loads((OUT / f'{a.final_tag}__STABILITY.json').read_text())
    inc = json.loads((OUT / f'{a.final_tag}__INCOME_NOISE.json').read_text())
    rec = {'schema': 'OHN1-rr-index-v1', 'utc': C.now(), 'files': files, 'queue_units': units,
           'queue_cpu_s_total': round(sum(u['charged_cpu_s'] for u in units), 1),
           'headline_copies': {'pooled_stability': st['pooled_all_pairs'], 'income_gate': inc['gate_false_failure'],
                               'income_pooled_components': inc['pooled_difference_components'],
                               'income_near_identical_empirical': {k: v for k, v in inc['gate_failures_near_identical_income_views'].items() if k != 'orientations'}},
           'note': 'development evidence on MODEL_SELECT (adaptively used); diagnostic; never selection; reader-relative'}
    C.write_json(OUT / 'RR_INDEX.json', rec)
    print(json.dumps({'files': len(files), 'units': len(units), 'queue_cpu_s_total': rec['queue_cpu_s_total']}))


if __name__ == '__main__':
    main()
