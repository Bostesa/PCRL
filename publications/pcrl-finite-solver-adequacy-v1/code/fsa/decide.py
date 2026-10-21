"""Coordinator decision records of the PCRL finite-output solver-adequacy (`pcrl_finite_solver_adequacy_v1`; label-free: this module
reads saved records only and applies the frozen FSA1 rules of `frc.rules`).

    python3 -m fsa.decide qual      -> results/<campaign>/decisions/QUAL_DECISION.json and QUAL_DECISION.md

Inputs (write-once records of earlier units):
- T0 `fa/REPRO_V<v>.json` (saved twins reproduced bitwise);
- T1 `fa/NEST_<cap>_V<v>.json` (nested fixed-routing comparison; GATE and decoder reproduction);
- T3 `runs/FA/TWB_<cap>/EVAL_MS.json` (B1 utility and MODEL_SELECT paired costs B1 - A) and
  `runs/SOURCE/FA/<source twin>/EVAL_MS.json` (A utility under this campaign's evaluator).
Applies fsa.rules.qpe_label, construction_choice, qual, capacity_gate, capacity_decision and d1, and writes the decision that the
protected stage requires (fsa.fa.route.protection_decision). Commit the decision before any protected fit.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from . import common as C
from . import rules as RU

CAPS = {'C64': '16x64', 'C128': '24x128', 'C256': '32x256'}
SIZES = {'C64': [16, 64], 'C128': [24, 128], 'C256': [32, 256]}
SOURCE_TWIN = {'C64': 'PRC_TWIN', 'C128': 'PRC_TWIN_C128'}
OUT = C.RESULTS / 'decisions'


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def nested_input(cid):
    """fsa.rules nested input {v: {'B1-A', 'B0-B1', 'spend_over_cont', 'fold_means'}} from the T1 records (None if missing)."""
    out = {}
    for v in C.VERSIONS:
        r = _json(C.RESULTS / 'fa' / f'NEST_{cid}_V{v}.json')
        if r is None:
            return None, None
        pdif = r['paired_differences']
        out[v] = {'B1-A': pdif['B1_minus_A'],
                  'B0-B1': {k: [-m, s] for k, (m, s) in pdif['B1_minus_B0'].items()},
                  'spend_over_cont': r['spend_over_cont'],
                  'fold_means': {k: x['fold_means'] for k, x in r['paired_differences_fold_level']['B1_minus_A'].items()}}
    return out, {v: _json(C.RESULTS / 'fa' / f'NEST_{cid}_V{v}.json') for v in C.VERSIONS}


def reproduced(cid, nest_recs):
    if cid not in SOURCE_TWIN:
        return None
    reps = [_json(C.RESULTS / 'fa' / f'REPRO_V{v}.json') for v in C.VERSIONS]
    if any(r is None for r in reps) or any(r is None for r in nest_recs.values()):
        return False
    ok = all(r['reproduced'] and r['twins'][cid]['book_refit_bitwise'] for r in reps)
    for v, r in nest_recs.items():
        sr = r.get('source_reproduction', {})
        ok &= bool(sr.get('gate', {}).get('reproduced')) and bool(sr.get('decoders_from_labels_bitwise'))
    return bool(ok)


def eval_record(cid, construction):
    if construction == 'B1':
        return _json(C.RUNS / 'FA' / f'TWB_{cid}' / 'EVAL_MS.json')
    if cid in SOURCE_TWIN:
        return _json(C.RUNS / 'SOURCE' / 'FA' / SOURCE_TWIN[cid] / 'EVAL_MS.json')
    return _json(C.RUNS / 'FA' / f'TWA_{cid}' / 'EVAL_MS.json')


def capacity_input(cid):
    nested, recs = nested_input(cid)
    if nested is None:
        return None
    eb, ea = eval_record(cid, 'B1'), eval_record(cid, 'A')
    comp = f'NRM:FA/{SOURCE_TWIN[cid]}' if cid in SOURCE_TWIN else f'FA/TWA_{cid}'
    ms = {v: eb['versions'][str(v)]['paired_costs'][comp]['costs'] for v in C.VERSIONS} if eb and \
        all(comp in eb['versions'][str(v)].get('paired_costs', {}) for v in C.VERSIONS) else None
    evals = {}
    if eb:
        evals['B1'] = {v: eb['versions'][str(v)]['utility'] for v in C.VERSIONS}
    if ea:
        evals['A'] = {v: ea['versions'][str(v)]['utility'] for v in C.VERSIONS}
    rec = {'nested': nested, 'ms': ms, 'evals': evals}
    rp = reproduced(cid, recs)
    if rp is not None:
        rec['reproduced'] = rp
    return rec


def qual_decision():
    by = {}
    for cid, key in CAPS.items():
        x = capacity_input(cid)
        if x is not None:
            by[key] = x
    cap = RU.capacity_decision(by)
    report = {}
    for cid, key in CAPS.items():
        x = by.get(key)
        if x is None:
            continue
        report[cid] = {'qpe': RU.qpe_label(x['nested'], x['ms']), 'construction_choice': RU.construction_choice(x['nested']),
                       'qual': {c: RU.qual(x['evals'][c]) for c in x['evals']},
                       'original_capacity_gate': {c: RU.capacity_gate({v: x['nested'][v]['spend_over_cont'][c]
                                                                       for v in C.VERSIONS}) for c in ('A', 'B0', 'B1')},
                       'reproduced': x.get('reproduced')}
    d1 = RU.d1(cap)
    outcome = 'QUALIFIED' if cap.get('state') == 'PROTECT' else cap.get('state')
    rec = {'schema': 'FSA1-qual-decision-v1', 'utc': C.now(), 'outcome': outcome, 'capacity_decision': cap, 'd1': d1,
           'per_capacity': report, 'rules_sha256': C.sha(RU.RULES_PATH),
           'note': 'MODEL_FIT nested evidence chooses the construction; MODEL_SELECT utility only qualifies (QUAL); the reserve and '
                   'the original capacity gate are reported, never gating; adaptive development evidence'}
    if outcome == 'QUALIFIED':
        inv = {v: k for k, v in CAPS.items()}
        cid = inv.get(cap['capacity'])
        cons = cap['construction']
        rec.update({'construction': cons, 'capacity_id': cid, 'capacity': SIZES[cid],
                    'twin': ({'source': 'fsa', 'config': f'TWB_{cid}'} if cons == 'B1' else
                             {'source': 'nrm', 'config': SOURCE_TWIN[cid]} if cid in SOURCE_TWIN else
                             {'source': 'fsa', 'config': f'TWA_{cid}'})})
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'QUAL_DECISION.json'
    if p.exists():
        raise PermissionError('QUAL_DECISION.json exists (write-once; a new decision gets a new file name)')
    C.write_json_x(p, rec)
    lines = [f'# Qualification decision ({rec["utc"]})', '', f'- Outcome: **{outcome}**; D1: **{d1.get("label")}**']
    if outcome == 'QUALIFIED':
        lines.append(f'- Protection capacity {rec["capacity_id"]} {rec["capacity"]}, construction {rec["construction"]}, twin '
                     f'{rec["twin"]}')
    for cid, r in report.items():
        lines.append(f'- {cid}: Q-PE {r["qpe"]["label"]}; choice {r["construction_choice"].get("choice")}; QUAL '
                     + ', '.join(f'{c} {q["label"]}' for c, q in r['qual'].items()) + '; original gate '
                     + ', '.join(f'{c} {g.get("label")}' for c, g in r['original_capacity_gate'].items())
                     + f'; reproduced {r["reproduced"]}')
    (OUT / 'QUAL_DECISION.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'outcome': outcome, 'd1': d1.get('label'), 'capacity': rec.get('capacity_id'),
                      'construction': rec.get('construction')}, indent=1))
    return rec


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv and argv[0] == 'qual':
        qual_decision()
        return 0
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
