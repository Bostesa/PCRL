"""Claims plans of the PCRL finite-routing comparison (coordinator; label-free). Writes, from the committed P1 / R1 selection records,
the saved release ledger and the label-free reader budget, the plans that frc.claims applies to the grid (commit them before the
claims are computed):

    python3 -m frc.plan write --bank FRC1_MAIN     -> results/<campaign>/claims/PLAN_MAIN.json (registered reading)
                                                       results/<campaign>/claims/PLAN_R1.json   (revision R1 reading)

Registered reading: untargeted path [SHUF_S2]. R1 reading (card FA_FRC_R1): [SHUF_S2, UNT_BR]. Every other input is identical.
"""
from __future__ import annotations
import json
import sys

from . import common as C
from . import readerbudget as RB

REFS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
FAMILY_MEMBERS = {'TWIN': 'TWB_C64', 'PRC_S2': 'PRC_S2', 'HQ_S2': 'HQ_S2', 'SHUF_S2': 'SHUF_S2', 'UNT_BR': 'UNT_BR', 'ORD_S2': 'ORD_S2',
                  'ORD_S3': 'ORD_S3', 'TWB_C128': 'TWB_C128'}
SOURCE_MEMBERS = {'A_TWIN': 'PRC_TWIN', 'A_TWIN_C128': 'PRC_TWIN_C128'}
NESTED_ENTRY = {'TWIN': 'PRC_TWIN', 'HQ_S2': ('HQ', 'S2'), 'SHUF_S2': ('SHUF', 'S2'), 'ORD_S2': ('ORD', 'S2'), 'ORD_S3': ('ORD', 'S3')}


def _sel(v):
    return json.loads((C.RESULTS / 'fa' / f'SELECTION_V{v}.json').read_text())


def nested_lookup():
    """{member: {v: AUC_member - AUC_prc}} of the nested held-out pair LOOKUP (P1; UNT_BR from the R1 record, same folds and twin)."""
    out = {}
    for v in C.VERSIONS:
        s = _sel(v)
        look = s['nested_lookup_pair']
        prc = look[s['selection']['PRC']['S2']]['auc']
        for m, e in NESTED_ENTRY.items():
            key = e if isinstance(e, str) else s['selection'][e[0]][e[1]]
            out.setdefault(m, {})[str(v)] = look[key]['auc'] - prc
        u = json.loads((C.RESULTS / 'fa' / f'UNT_SELECTION_V{v}.json').read_text())
        out.setdefault('UNT_BR', {})[str(v)] = u['nested_lookup_pair'][u['bracket']]['auc'] - prc
    return out


def reader_budget_flags(budget_id='MAIN', cand='PRC_S2', others=REFS + tuple(FAMILY_MEMBERS) + tuple(SOURCE_MEMBERS)):
    b = json.loads((C.RESULTS / 'readers' / f'BUDGET_{budget_id}.json').read_text())['members']
    flags, detail = {}, {}
    for o in others:
        if o == cand or o not in b:
            continue
        cs = {f'V{v}{rep}': RB.contrast(b[cand]['versions'][str(v)], b[o]['versions'][str(v)], view='pair', rep=rep)
              for v in C.VERSIONS for rep in ('R1', 'R2')}
        flags[o] = 'READER_BUDGET_ASYMMETRIC' if any(c['label'] == 'READER_BUDGET_ASYMMETRIC' for c in cs.values()) \
            else 'READER_BUDGET_COMPARABLE'
        detail[o] = {k: {'label': c['label'], 'group_size_ratio': c['group_size_ratio'],
                         'score_unseen_difference': c['score_unseen_difference']} for k, c in cs.items()}
    return flags, detail


def write(bank):
    f4 = {'version': {}}
    for v in C.VERSIONS:
        x = _sel(v)['F4']
        f4['version'][str(v)] = {'diff': x['diff_twin_minus_prc'], 'se': x['se_boot'], 'entry': x['entry']}
    members = {m: {'eval': f'frc:FA/{c}'} for m, c in FAMILY_MEMBERS.items()}
    members.update({m: {'eval': f'nrm:FA/{c}'} for m, c in SOURCE_MEMBERS.items()})
    members.update({r: {'eval': f'ref:{r}'} for r in REFS})
    flags, detail = reader_budget_flags()
    base = {'bank': bank, 'prc_s2': 'PRC_S2', 'members': members, 'alias_of': {},
            'every_nonzero_selection_alias_of_twin': False, 'f4': f4, 'nested_lookup': nested_lookup(),
            'reader_budget': flags, 'reader_budget_detail': detail,
            'minimum_set': ['PRC_S2', 'TWIN', 'HQ_S2', 'SHUF_S2', 'UNT_BR', 'ORD_S2', 'ORD_S3'],
            'notes': {'PRC_S1': 'version 1 only (version 0 NOT_FEASIBLE) and an exact alias of PRC_S2 in version 1 (identical wires): '
                                'not a bank member; PRC_S2 is the D2 configuration in both versions',
                      'PRC_S3': 'ALIAS_OF_PRC_S2 in both versions',
                      'HQ_BR': 'version 1 only (version 0 CONTROL_DID_NOT_REACH_SPEND): not a bank member; the pooled HQ path is [HQ_S2] '
                               '(HQ_S2 spends more than PRC_S2 on MODEL_SELECT in both versions)',
                      'SHUF_BR': 'CONTROL_DID_NOT_REACH_SPEND in both versions', 'ORD_S1': 'NOT_FEASIBLE in both versions',
                      'ORD_BR': 'ALIAS_OF_ORD_S3 in both versions',
                      'nested_lookup': 'P1 nested held-out pair LOOKUP AUC of each member minus that of the S2 PRC entry; UNT_BR from the '
                                       'R1 record (same folds, teachers and twin as P1, checked equal within 1e-12)'},
            'written_utc': C.now()}
    out = C.RESULTS / 'claims'
    out.mkdir(parents=True, exist_ok=True)
    plans = {'PLAN_MAIN': {**base, 'plan_id': 'MAIN', 'reading': 'registered', 'paths': {'HQ': ['HQ_S2'], 'SHUF': ['SHUF_S2'],
                                                                                          'ORD': ['TWIN', 'ORD_S2', 'ORD_S3']}},
             'PLAN_R1': {**base, 'plan_id': 'R1', 'reading': 'revision R1 (card FA_FRC_R1)',
                         'paths': {'HQ': ['HQ_S2'], 'SHUF': ['SHUF_S2', 'UNT_BR'], 'ORD': ['TWIN', 'ORD_S2', 'ORD_S3']}}}
    for name, plan in plans.items():
        p = out / f'{name}.json'
        if p.exists():
            raise PermissionError(f'{p.name} exists (write-once)')
        C.write_json_x(p, plan)
    print(json.dumps({'written': sorted(plans), 'reader_budget_flags': flags}, indent=1))


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) == 3 and argv[0] == 'write' and argv[1] == '--bank':
        write(argv[2])
        return 0
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
