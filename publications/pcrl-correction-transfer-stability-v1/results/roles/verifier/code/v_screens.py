"""Verifier check 6: recipient screens and stability labels of the reader tables, recomputed from the grids' paired cells with the
verifier's own implementation of RULES.md section 10 (old per-cell gate; aggregate paired rule AGG1 with phi = max(1, phi_prior(view),
phi_emp); exact aliases from the wire hashes) and of the source stability screen (rung 3; z = 2). Label-free (reads GRID.json records).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).

    python3 .../v_screens.py --table READER_STABILITY_D_FRONTIER_KEY --bank D_FRONTIER --tag V1
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

GRID = V.ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'GRID'
TABLES = V.RESULTS / 'tables'
PHI_PRIOR = {'income': 5.636511599640258, 'occupation': 0.7781545862915747, 'pair': 1.0867113014695036}   # RULES.md section 10
ALLOW = 0.010
CELL = re.compile(r'^S(\d+)R(\d+)k(\d+)$')


def gate(cells):
    vals = list(cells.values())
    if not vals or not all(math.isfinite(v) for v in vals):
        return 'INCOMPLETE', None
    return ('GATE_PASS' if all(v <= ALLOW for v in vals) else 'GATE_FAIL'), max(vals)


def agg1(cells, ses, view):
    parsed = {c: tuple(int(x) for x in CELL.match(c).groups()) for c in cells}
    reps = sorted({p[1] for p in parsed.values()})
    dbar = float(np.mean(list(cells.values())))
    y = {r: float(np.mean([cells[c] for c, p in parsed.items() if p[1] == r])) for r in reps}
    se = {r: float(np.mean([ses[c] for c, p in parsed.items() if p[1] == r])) for r in reps}
    n = len(reps)
    if n < 2:
        return 'INCOMPLETE', None
    v_samp = sum(se[r] ** 2 for r in reps) / n ** 2
    mse = float(np.mean([se[r] ** 2 for r in reps]))
    v_btw = sum((y[r] - dbar) ** 2 for r in reps) / (n - 1)
    if mse <= 0:
        return 'ZERO_SE', None
    phi = max(1.0, PHI_PRIOR[view], v_btw / mse)
    U = dbar + V.Z_REGISTERED * math.sqrt(phi * v_samp)
    return ('AGG_PASS' if U <= ALLOW else 'AGG_FAIL'), U


def stability(cells, ses, z=2.0):
    parsed = {c: tuple(int(x) for x in CELL.match(c).groups()) for c in cells}
    seeds = sorted({p[0] for p in parsed.values()})
    by = {s: [cells[c] for c, p in parsed.items() if p[0] == s] for s in seeds}
    d = {s: float(np.mean(v)) for s, v in by.items()}
    v_rep = sum((x - d[s]) ** 2 for s in seeds for x in by[s]) / (len(cells) - len(seeds))
    v_samp = float(np.mean([ses[c] ** 2 for c in cells]))
    sd = math.sqrt(v_rep + v_samp)
    allpos = all(v > 0 for v in cells.values())
    stable = allpos and all(d[s] - z * sd > 0 for s in seeds)
    mean = float(np.mean(list(cells.values())))
    label = 'STABLE_POSITIVE' if stable else 'POSITIVE_NOT_STABLE' if allpos else 'SIGN_CHANGES' if mean > 0 else 'NOT_POSITIVE'
    return label, sd, min(d[s] - z * sd for s in seeds), mean, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True); ap.add_argument('--bank', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    tab = json.loads((TABLES / f'{a.table}.json').read_text())
    grids = {int(p.name[1:]): json.loads((p / 'GRID.json').read_text()) for p in sorted((GRID / a.bank).glob('S*'))}
    mism, rows = [], {}
    for key, rec in tab['records'].items():
        cand, comp, view = rec['candidate'], rec['comparator'], rec['view']
        cells, ses = {}, {}
        for s, g in grids.items():
            pv = g['paired'][cand][comp][view]
            cells.update(pv['cells']); ses.update(pv['cell_se'])
        alias = all(g['members'][cand]['wire_sha256'].get(view) == g['members'][comp]['wire_sha256'].get(view)
                    for g in grids.values()) if view != 'pair' else all(
            g['members'][cand]['wire_sha256'] == g['members'][comp]['wire_sha256'] for g in grids.values())
        r = {'n_cells': len(cells), 'alias': alias}
        if len(cells) != 8 or len(grids) != 2:
            # RULES.md section 10 / rung 3 define the screens and the stability label over the 8 registered cells (both release
            # seeds x 2 repetitions x 2 reader seeds); a one-seed bank gets descriptive means only and must carry no screen label
            mean = float(np.mean(list(cells.values())))
            r.update({'screenable': False, 'mean': mean})
            if abs(mean - rec['mean']) > 1e-12 or any(rec.get(k) is not None for k in ('old_gate', 'agg1', 'stability')):
                mism.append(f'{key}: one-seed record mean / labels differ')
            rows[key] = r
            continue
        if view == 'pair':
            lab, sd, lo, mean, d = stability(cells, ses)
            r.update({'stability': lab, 'sd_total': sd, 'min_lower': lo, 'mean': mean})
            if lab != rec.get('stability') or abs(sd - rec['sd_total']) > 1e-12 or abs(lo - rec['min_lower']) > 1e-12 \
                    or abs(mean - rec['mean']) > 1e-12:
                mism.append(f'{key}: stability {lab} vs {rec.get("stability")}')
        else:
            gl, gmax = gate(cells)
            if alias:
                if any(v != 0.0 for v in cells.values()):
                    mism.append(f'{key}: exact alias with nonzero cells')
                al, U = 'EXACT_ALIAS', 0.0
            else:
                al, U = agg1(cells, ses, view)
            r.update({'old_gate': gl, 'old_gate_max': gmax, 'agg1': al, 'agg1_U': U})
            if gl != rec.get('old_gate') or (gmax is not None and abs(gmax - rec['old_gate_max']) > 1e-12):
                mism.append(f'{key}: gate {gl} vs {rec.get("old_gate")}')
            if al != rec.get('agg1') or (U is not None and rec.get('agg1_U') is not None and abs(U - rec['agg1_U']) > 1e-12):
                mism.append(f'{key}: AGG1 {al} {U} vs {rec.get("agg1")} {rec.get("agg1_U")}')
        rows[key] = r
    both = {k: (v['old_gate'], v['agg1']) for k, v in rows.items() if 'old_gate' in v}
    differ = {k: v for k, v in both.items() if (v[0] == 'GATE_PASS') != (v[1] in ('AGG_PASS', 'EXACT_ALIAS'))}
    chk = {'id': f'C6.3.SCREENS.{a.table}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{a.table}: old per-cell recipient gate and AGG1 (single views) reported side by side, and the stability label of the '
                   'pair benefit, for every recorded (candidate, comparator, view)',
           'method': 'own implementation of RULES.md section 10 (gate: every cell <= +0.010; AGG1: dbar + z sqrt(phi v_samp), phi = max(1, '
                     'phi_prior(view), phi_emp), z = 1.644853626951472, exact alias from the wire hashes) and of the source stability '
                     'screen (z = 2; pooled within-seed variance + mean squared cell SE) on the grids\' paired cells; tolerance 1e-12',
           'status': 'PASS' if not mism else 'FAIL',
           'numbers': {'records': len(rows), 'mismatches': mism[:30], 'gate_and_agg1_disagree': differ,
                       'stability_labels': {k: v['stability'] for k, v in rows.items() if 'stability' in v},
                       'seeds_in_grid': sorted(grids)}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_SCREENS__{a.table}__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'mismatches': mism[:10], 'n': len(rows), 'gate_vs_agg1_differ': len(differ)}))


if __name__ == '__main__':
    main()
