"""Verifier check of the RESEARCH_DECISION.md statements against verified records (label-free; reads campaign records and the
verifier's own replay records only). Checks wording and scope; it does not re-derive numbers already replayed elsewhere.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).

    python3 .../v_claims.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

REC = V.VER_DIR / 'records'
RUNS = V.ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    items, att = {}, []
    # 1. binding men's-guard reserve slack of the refitted witnesses (decision: +0.0005, -0.0008, +0.0001, -0.0003)
    rho = {}
    for P in ('D0', 'D1', 'R0', 'R1'):
        c = json.loads((REC / f'V_COSTS__{P}__V1.json').read_text())
        rho[P] = [x for x in c['checks'] if 'WITNESS_BASELINE' in x['id']][0]['numbers']['conditions']['guard|sex0']['rho']
    stated = {'D0': 0.0005, 'D1': -0.0008, 'R0': 0.0001, 'R1': -0.0003}
    items['mens_guard_reserve_slack'] = {'verified': rho, 'stated': stated,
                                         'consistent': all(abs(rho[P] - stated[P]) < 0.00005 for P in rho)}
    # 2. nonzero strengths chosen on the panels: crossings on C0 (complete release against W), by deployment
    rows = {'full': [0, 0], 'ens': [0, 0]}
    for d in ('REFIT_PAIR', 'REFIT_LOC', 'REFIT_CONT', 'REFIT_PAIR_TR'):
        r = json.loads((RUNS / 'R0' / 'SELECT' / f'{d}__REALIZED.json').read_text())['realized']
        for key, v in r['CHECK'].items():
            if v['selected'] > 0:
                dep = key.split('|')[1]
                oc = v['outcome'] if dep == 'full' else v['outcome_complete']
                rows[dep][0] += 1; rows[dep][1] += int(oc['reserve_crossing'])
    c = json.loads((REC / 'V_COSTS__R0__V1.json').read_text())
    ens_identity = [x for x in c['checks'] if x['id'].startswith('C4.R0.REFIT_PAIR.REALIZED')][0]['numbers']['complete_release_outcomes_mine']['CHECK/UNCERTAINTY|ens']
    items['panel_selections_overspent'] = {'R0_nonzero_full': {'n': rows['full'][0], 'reserve_crossings_on_C0': rows['full'][1]},
                                          'R0_nonzero_ens': {'n': rows['ens'][0], 'reserve_crossings_on_C0': rows['ens'][1]},
                                          'R1_nonzero': 0,
                                          'uncorrected_ensemble_e0_vs_W_on_C0': ens_identity}
    if ens_identity['reserve_crossing'] and rows['ens'][1] == rows['ens'][0]:
        att.append('"the nonzero strengths chosen on the panels mostly overspent the reserve": on C0 %d of %d nonzero selections cross '
                   'the reserve, but every ens selection crosses (%d / %d) and so does the UNCORRECTED ensemble e_0 against W (men\'s '
                   'cost %.5f, slack use %.1f): the ens crossings belong to the ensemble base, not to the chosen strengths; full '
                   'deployment: %d of %d' % (rows['full'][1] + rows['ens'][1], rows['full'][0] + rows['ens'][0], rows['ens'][1],
                                             rows['ens'][0], ens_identity['men_brier_vs_W'], ens_identity['max_slack_use'], rows['full'][1],
                                             rows['full'][0]))
    # 3. question A label of the coalition-aware direction on all four partitions (registered scheme)
    qa = {}
    for f in sorted(REC.glob('V_QUESTIONA__*__REG__V1.json')):
        x = json.loads(f.read_text())
        P, did, dep, oof = f.stem.split('__')[1:5]
        qa[f'{P}|{did}|{dep}|{oof}'] = x['numbers']['mine']['label']
    pair_refit = {k: v for k, v in qa.items() if k.split('|')[1] == 'REFIT_PAIR' and k.endswith('OOF')}
    others_improved = sorted(k for k, v in qa.items() if v == 'PREDICTION_IMPROVED')
    items['question_A'] = {'REFIT_PAIR_labels': pair_refit, 'all_REFIT_PAIR_not_improved': all(v == 'NOT_IMPROVED' for v in pair_refit.values()),
                           'prediction_improved_registered': others_improved}
    if any(k.split('|')[1] == 'FROZEN_PAIR' for k in others_improved):
        att.append('"NOT_IMPROVED for the coalition-aware direction on all four training partitions" holds for REFIT_PAIR (the only PAIR '
                   'direction fitted on all four); FROZEN_PAIR (D partitions only) is PREDICTION_IMPROVED on D1 ens (registered scheme) '
                   'and should be named in the scope')
    # 4. coalition-aware correction vs matched zero-critic continuation at common strengths
    st = {}
    for t, f in (('D', 'V_SCREENS__READER_STABILITY_D_FRONTIER_KEY__V1.json'), ('R', 'V_SCREENS__READER_STABILITY_R_FRONTIER__V1.json')):
        x = json.loads((REC / f).read_text())['numbers']['stability_labels']
        st[t] = {k: v for k, v in x.items() if k.startswith('REFIT_PAIR@') and '|REFIT_CONT@' in k and k.split('|')[0].split('@')[1] ==
                 k.split('|')[1].split('@')[1]}
    items['pair_vs_cont_common_strengths'] = st
    status = 'ATTENTION' if att else 'PASS'
    chk = {'id': f'C6.6.DECISION_STATEMENTS__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': 'RESEARCH_DECISION.md decision-in-brief statements against the verified records (reserve slacks, panel overspending, '
                   'question A scope, PAIR vs CONT at common strengths); wording and scope only',
           'method': 'verifier replay records (V_COSTS, V_QUESTIONA REG, V_SCREENS) and the campaign REALIZED records (replayed in C4)',
           'status': status, 'numbers': {'items': items, 'wording_or_scope_notes': att}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_CLAIMS__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: status, 'notes': att, 'rho': rho, 'stability': st}, indent=1))


if __name__ == '__main__':
    main()
