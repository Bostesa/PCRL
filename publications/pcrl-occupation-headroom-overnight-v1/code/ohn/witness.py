"""Stage-1 close: task-only occupation witness selection with the registered rule (ohn.rules.select_witnesses) and the
income-branch trigger (ohn.rules.income_branch_trigger). Reads only the stored MODEL_SELECT evaluations (EVAL_MS_H3.json) and the
ledger; no new label read.

    python -m ohn.witness select [--families A B C]     -> results/<campaign>/WITNESS_SELECTION.json (write-once) + ledger rows
"""
from __future__ import annotations
import argparse
import json

from . import common as C
from . import devrun as D
from . import ledger as L
from . import rules as RU


def candidates(families):
    rows = L.rows()
    first_entry = {}
    for r in rows:
        if r['kind'] in ('task_model', 'control') and r['seed'] == '0' and r['anchor'] == 'H3':
            first_entry.setdefault((r['family'], r['config_id']), int(r['entry']))
    out, skipped = {}, {}
    for fam in families:
        root = C.RUNS / fam
        if not root.is_dir():
            continue
        for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
            evals, recs = {}, {}
            for s in C.SEEDS:
                d = cdir / f'S{s}'
                if (d / 'EVAL_MS_H3.json').is_file() and (d / 'OCC.json').is_file():
                    evals[s] = json.loads((d / 'EVAL_MS_H3.json').read_text())
                    recs[s] = json.loads((d / 'OCC.json').read_text())
            if not evals:
                continue
            kinds = {r.get('kind') for r in recs.values()}
            task_only = kinds <= {'task_model', 'control'} and not any('SEX' in str(r.get('roles_read', '')).upper()
                                                                      and 'NO SEX' not in str(r.get('roles_read', '')).upper()
                                                                      for r in recs.values())
            key = f'{fam}/{cdir.name}'
            out[key] = {'evals': evals, 'task_only': task_only,
                        'n_model_fits': sum(int(r.get('n_model_fits') or 0) for r in recs.values()),
                        'n_calibrator_fits': sum(int(r.get('n_calibrator_fits') or 0) for r in recs.values()),
                        'entry': first_entry.get((fam, cdir.name), 10 ** 9), 'kinds': sorted(k for k in kinds if k)}
    return out, skipped


def select(families):
    cands, _ = candidates(families)
    sel = RU.select_witnesses(cands)
    top = sel['selected'][0] if sel['selected'] else None
    complete_reserve = RU.reserve(cands[top]['evals']) if top else None
    trig = RU.income_branch_trigger(sel, complete_reserve)
    rec = {'schema': 'OHN1-witness-selection-v1', 'utc': C.now(), 'rule_sha256': RU.load_rules().get('_sha256'),
           'families': list(families), 'n_candidates': len(cands),
           'candidates': {k: {kk: vv for kk, vv in v.items() if kk != 'evals'} | {'seeds': sorted(v['evals'])} for k, v in cands.items()},
           'selection': sel, 'top_witness_complete_reserve': complete_reserve, 'income_branch_trigger': trig,
           'note': 'utility evidence only (MODEL_SELECT, adaptively used); says nothing about SEX recovery'}
    C.write_json_x(C.RESULTS / 'WITNESS_SELECTION.json', rec)
    L.append({'family': 'SELECT', 'version': 'witness', 'config_id': 'WITNESS_SELECTION', 'kind': 'decision', 'budget_class': 'NONE',
              'counts': 0, 'decision': json.dumps({'selected': sel['selected'], 'ranking': sel['ranking']})[:400],
              'notes': f"income branch trigger fires={trig['fires']} (leg A {trig['leg_A_model_select']}, leg B {trig['leg_B_published_dev']})"})
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select'); s.add_argument('--families', nargs='+', default=['A', 'B', 'C'])
    a = ap.parse_args(argv)
    rec = select(a.families)
    sel = rec['selection']
    print(json.dumps({'selected': sel['selected'], 'ranking': sel['ranking'],
                      'labels': {k: (v['label'], v.get('selection')) for k, v in sel['records'].items()},
                      'income_branch_trigger': {k: v for k, v in rec['income_branch_trigger'].items() if k != 'leg_B_detail'}},
                     indent=1, default=str))


if __name__ == '__main__':
    main()
