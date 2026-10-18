"""Independent verifier, check 5 summary (no label read): two-seed mean pair proxy and the per-cell pair benefits / recipient
differences against NEURAL_RAW and H_LJ from my recomputed cells (PROXY_CELLS_S0/S1.json), compared with NOMINATION.json."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RES = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'
V = ROOT / '.private' / 'pcrl_income_headroom_successor_v1' / 'verifier'
CONFIGS = ('H3_STRONG_LJDIR_PB', 'HC_STRONG', 'HC_TTREE_LJDIR_PB')
REFS = ('NEURAL_RAW', 'H_LJ')
CELLS = ('A0', 'A1', 'B0', 'B1')


def main():
    cells = {s: json.loads((V / f'PROXY_CELLS_S{s}.json').read_text()) for s in (0, 1)}
    nom = {c['config_id']: c for c in json.loads((RES / 'NOMINATION.json').read_text())['configurations']}
    out = {'per_seed_totals': {s: {k: cells[s][k] for k in ('n_compared', 'n_exact', 'max_abs_diff', 'dir_compared', 'dir_max_abs_diff', 'benefit_max_abs_diff')} for s in cells}}
    worst = 0.0
    for cfg in CONFIGS:
        vals = [cells[s]['releases'][cfg]['cells'][c]['pair']['max_incl_single']['recomputed'] for s in (0, 1) for c in CELLS]
        score = sum(vals) / len(vals)
        rec = {'score_recomputed': score, 'score_stored': nom[cfg]['score_two_seed_mean_pair_proxy'], 'abs_diff': abs(score - nom[cfg]['score_two_seed_mean_pair_proxy']),
               'label_stored': nom[cfg]['label'], 'c1_stored': nom[cfg]['c1_pair_superior'], 'c2_stored': nom[cfg]['c2_recipient_screen']}
        c1, c2, diffs = True, True, []
        for r in REFS:
            for s in (0, 1):
                st = nom[cfg]['reference_cells'][f'{r}|s{s}']
                for c in CELLS:
                    b = cells[s]['releases'][r]['cells'][c]['pair']['max_incl_single']['recomputed'] - cells[s]['releases'][cfg]['cells'][c]['pair']['max_incl_single']['recomputed']
                    iv = cells[s]['releases'][cfg]['cells'][c]['income']['view_max']['recomputed'] - cells[s]['releases'][r]['cells'][c]['income']['view_max']['recomputed']
                    ov = cells[s]['releases'][cfg]['cells'][c]['occupation']['view_max']['recomputed'] - cells[s]['releases'][r]['cells'][c]['occupation']['view_max']['recomputed']
                    diffs += [abs(b - st['pair_benefit_cells'][c]), abs(iv - st['income_recipient_cells'][c]), abs(ov - st['occupation_recipient_cells'][c])]
                    c1 &= b > 0; c2 &= (iv <= 0.010) and (ov <= 0.010)
        rec.update({'c1_recomputed': c1, 'c2_recomputed': c2, 'max_abs_diff_reference_cells_vs_nomination': max(diffs)})
        worst = max(worst, rec['abs_diff'], rec['max_abs_diff_reference_cells_vs_nomination'])
        out[cfg] = rec
    out['worst_abs_diff_vs_nomination'] = worst
    (V / 'PROXY_SUMMARY.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
