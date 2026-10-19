"""Reader bank specification of the conditional stage (label-free; written after the frozen strength selection, before any reader).

    python3 -m tsz.banks conditional        writes results/pcrl_controlled_training_size_v1/grid_specs/COND.json

Members (each a complete release: frozen H3 income of the version + an occupation output), per version v the ACTUAL selected outputs:
  W              W100_v (tsz.sizefit A1__S100__V<v>)
  LOC_SEL        LOC at its selected strength s*_LOC,v (identity = W exactly when s* = 0)
  PAIR_SEL       PAIR at its selected strength s*_PAIR,v
  CONT_AT_LOC    the matched zero-critic continuation at s*_LOC,v
  CONT_AT_PAIR   the matched zero-critic continuation at s*_PAIR,v
  LOC_AT_PAIR    LOC at s*_PAIR,v (descriptive only; RULES.md section 14 point 4; never enters a label)
  TR100, TT100   the equally exposed full-pool task-only controls (TASK_REFIT and its T_TREE first stage on S100) with H3 income,
                 read so that the equal-reserve comparison covers every method
  NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ   the strong inherited references (frozen source releases)
Selected strengths come only from the selection records (SELECTION_V<v>.json); nearby strengths are never substituted.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from . import common as C

REFERENCES = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
OUT = C.RESULTS / 'grid_specs'


def _selection(v):
    from . import select as SEL
    p = Path(SEL.SEL_DIR) / f'SELECTION_V{v}.json'
    if not C.committed(p):
        raise PermissionError(f'{p.name} must be committed before the reader bank is built')
    rec = json.loads(p.read_text())
    return rec, C.sha(p)


def _selected(rec, arm):
    for key in ('selected', 'selections'):
        if key in rec and arm in rec[key]:
            x = rec[key][arm]
            return float(x['selected_strength'] if isinstance(x, dict) else x)
    raise KeyError(f'no selected strength for {arm}')


def conditional():
    from . import correct as CR
    members, prov = {}, {}
    files = {}
    for v in C.VERSIONS:
        rec, sha = _selection(v)
        s_loc, s_pair = _selected(rec, 'LOC'), _selected(rec, 'PAIR')
        prov[str(v)] = {'selection_sha256': sha, 's_LOC': s_loc, 's_PAIR': s_pair}
        rel_path = CR.cdir(v) / 'RELEASES.npz'
        files[v] = {'path': str(rel_path.relative_to(C.ROOT)), 'sha256': C.sha(rel_path)}
        wpath = C.RUNS / 'SIZE' / f'A1__S100__V{v}' / 'RELEASE.npz'
        spec_v = {
            'W': {'path': str(wpath.relative_to(C.ROOT)), 'key': 'MODEL_SELECT', 'sha256': C.sha(wpath)},
            'LOC_SEL': ('LOC', s_loc), 'PAIR_SEL': ('PAIR', s_pair), 'CONT_AT_LOC': ('CONT', s_loc), 'CONT_AT_PAIR': ('CONT', s_pair),
            'LOC_AT_PAIR': ('LOC', s_pair)}
        cdir = C.RUNS / 'CONTROL' / f'S100__V{v}'
        for cname, sub, fname, key in (('TR100', 'STAGE_B', 'STAGE_B_OUTPUTS_FINAL.npz', 'TASK_REFIT__MODEL_SELECT__occupation'),
                                       ('TT100', 'TEACHERS', 'TEACHER_OUTPUTS.npz', 'T_TREE__MODEL_SELECT__occupation')):
            p = cdir / sub / fname
            spec_v[cname] = {'path': str(p.relative_to(C.ROOT)), 'key': key, 'sha256': C.sha(p), 'row_id_key': 'row_id__MODEL_SELECT'}
        for name, x in spec_v.items():
            if isinstance(x, tuple):
                arm, s = x
                occ = {**files[v], 'key': f'{arm}__s{CR.skey(s)}__MODEL_SELECT'} if s != 0.0 else \
                    {'path': str(wpath.relative_to(C.ROOT)), 'key': 'MODEL_SELECT', 'sha256': C.sha(wpath)}
            else:
                occ = x
            members.setdefault(name, {'type': 'arrays', 'seeds': {}})['seeds'][str(v)] = {'income': {'frozen_anchor': 'H3'}, 'occupation': occ}
    for r in REFERENCES:
        members[r] = {'type': 'reference', 'name': r}
    cands = ['W', 'LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'LOC_AT_PAIR']
    comps = ['W', 'LOC_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'LOC_AT_PAIR', 'TR100', 'TT100'] + list(REFERENCES)
    spec = {'bank_id': 'COND', 'purpose': 'conditional stage: the actual selected corrected releases of each version (identity where '
                                          'selected), the unchanged witness, the matched continuations and the strong inherited references',
            'members': members, 'candidates': cands, 'comparators': comps, 'fractions': [1.0], 'reps': ['R1', 'R2'],
            'selection_provenance': prov,
            'note': 'an arm whose selected strength is 0 in a version is W exactly in that version (exact alias); CONT_AT_x is matched '
                    'to the strength selected for arm x in the same version; LOC_AT_PAIR is descriptive only'}
    C.write_json_x(OUT / 'COND.json', spec)
    print(json.dumps(prov))
    return spec


if __name__ == '__main__':
    if sys.argv[1:] != ['conditional']:
        raise SystemExit(__doc__)
    conditional()
