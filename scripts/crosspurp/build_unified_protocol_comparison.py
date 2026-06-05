"""Build apples-to-apples comparison: submission Δ vs cross-purpose Δ, same protocol.

Submission values: hard-coded from paper-body/tables/cross_purpose_attack_extra.tex
(mean ± std over 3 PCRL seeds, gain_pp = concat_acc - best_single_purpose_acc).

Cross-purpose values: read from results/rebuttal/cross_purpose/unified_protocol_results.json
(aggregate field, same protocol).
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = REPO_ROOT / 'results/rebuttal/cross_purpose/unified_protocol_results.json'

SUBMISSION = {
    # ds, attr, arch -> (mean, std)
    ('adult', 'age_group',     'LR'):  (1.62, 0.59),
    ('adult', 'age_group',     'MLP'): (7.66, 3.48),
    ('adult', 'age_group',     'XGB'): (9.28, 2.62),
    ('adult', 'income',        'LR'):  (-0.05, 0.22),
    ('adult', 'income',        'MLP'): (0.14, 0.13),
    ('adult', 'income',        'XGB'): (0.30, 0.54),
    ('adult', 'marital_status','LR'):  (4.08, 1.98),
    ('adult', 'marital_status','MLP'): (6.87, 1.77),
    ('adult', 'marital_status','XGB'): (10.87, 1.12),
    ('adult', 'race',          'LR'):  (-0.09, 0.10),
    ('adult', 'race',          'MLP'): (1.30, 1.52),
    ('adult', 'race',          'XGB'): (2.79, 0.87),
    ('adult', 'sex',           'LR'):  (0.38, 0.62),
    ('adult', 'sex',           'MLP'): (4.09, 1.87),
    ('adult', 'sex',           'XGB'): (6.70, 1.57),
    ('hmda',  'ethnicity',     'LR'):  (2.51, 1.65),
    ('hmda',  'ethnicity',     'MLP'): (6.04, 6.29),
    ('hmda',  'ethnicity',     'XGB'): (6.45, 5.34),
    ('hmda',  'race',          'LR'):  (3.79, 5.04),
    ('hmda',  'race',          'MLP'): (5.55, 0.90),
    ('hmda',  'race',          'XGB'): (5.14, 1.32),
    ('hmda',  'sex',           'LR'):  (4.78, 7.21),
    ('hmda',  'sex',           'MLP'): (5.24, 2.46),
    ('hmda',  'sex',           'XGB'): (5.07, 2.47),
    ('diabetes','age_bucket',  'LR'):  (9.94, 5.88),
    ('diabetes','age_bucket',  'MLP'): (9.53, 2.87),
    ('diabetes','age_bucket',  'XGB'): (11.96, 2.60),
    ('diabetes','gender',      'LR'):  (0.20, 0.34),
    ('diabetes','gender',      'MLP'): (0.31, 0.08),
    ('diabetes','gender',      'XGB'): (-0.50, 0.26),
    ('diabetes','race',        'LR'):  (-0.01, 0.02),
    ('diabetes','race',        'MLP'): (0.00, 0.00),
    ('diabetes','race',        'XGB'): (-0.25, 0.08),
}

def main():
    cp = json.load(open(DEFAULT_JSON))
    agg = cp.get('aggregate', cp)

    print(f"{'='*100}")
    print(f"{'cell':<28} {'submission':>14} {'cross-purpose':>17} {'Δ':>12} {'verdict':>15}")
    print(f"{'-'*100}")

    # Category counters
    improved_count = 0; flat_count = 0; regressed_count = 0
    cells = []
    for (ds, attr, arch), (sub_mean, sub_std) in SUBMISSION.items():
        cell_key = f'{attr}/{arch}'
        cp_data = agg.get(ds, {}).get(cell_key)
        if cp_data is None:
            print(f"{ds+'/'+cell_key:<28} {sub_mean:+6.2f}±{sub_std:4.2f}{'':>4} (cross-purpose: NOT YET COMPUTED)")
            continue
        cp_mean = cp_data['gain_mean_pp']
        cp_std  = cp_data['gain_std_pp']
        delta = cp_mean - sub_mean
        # Categorize using std-aware comparison
        if abs(delta) <= max(sub_std, cp_std):
            verdict = "FLAT"; flat_count += 1
        elif delta < 0:
            verdict = "IMPROVED"; improved_count += 1
        else:
            verdict = "REGRESSED"; regressed_count += 1
        cells.append((ds, attr, arch, sub_mean, sub_std, cp_mean, cp_std, delta, verdict))
        print(f"{ds+'/'+cell_key:<28} {sub_mean:+6.2f}±{sub_std:4.2f}{'':<4} {cp_mean:+6.2f}±{cp_std:4.2f}{'':<5} {delta:+7.2f}pp {verdict:>15}")

    print(f"{'='*100}")
    print(f"  IMPROVED (cp_mean < sub_mean by > max-std): {improved_count}")
    print(f"  FLAT     (|cp_mean - sub_mean| ≤ max-std):  {flat_count}")
    print(f"  REGRESSED (cp_mean > sub_mean by > max-std): {regressed_count}")

    # Worst regressions
    regressions = [(c, c[7]) for c in cells if c[8] == "REGRESSED"]
    regressions.sort(key=lambda x: -x[1])
    if regressions:
        print(f"\n  Worst regressions:")
        for (ds, attr, arch, sm, ss, cm, cs, dl, _), _ in regressions[:5]:
            print(f"    {ds}/{attr}/{arch}: {sm:+.2f}±{ss:.2f} → {cm:+.2f}±{cs:.2f}  (Δ={dl:+.2f}pp)")

    # Best improvements
    improvements = [(c, c[7]) for c in cells if c[8] == "IMPROVED"]
    improvements.sort(key=lambda x: x[1])
    if improvements:
        print(f"\n  Best improvements:")
        for (ds, attr, arch, sm, ss, cm, cs, dl, _), _ in improvements[:5]:
            print(f"    {ds}/{attr}/{arch}: {sm:+.2f}±{ss:.2f} → {cm:+.2f}±{cs:.2f}  (Δ={dl:+.2f}pp)")

    # Flag-status changes (>1pp threshold on each side)
    flag_changes = []
    for c in cells:
        ds, attr, arch, sm, ss, cm, cs, dl, _ = c
        sub_flag = sm > 1.0
        cp_flag = cm > 1.0
        if sub_flag != cp_flag:
            flag_changes.append((ds, attr, arch, sm, cm, sub_flag, cp_flag))
    print(f"\n  Flag-status changes (>1pp on each side, same protocol):")
    if flag_changes:
        for ds, attr, arch, sm, cm, sf, cf in flag_changes:
            arrow = "→"
            print(f"    {ds}/{attr}/{arch}: {sm:+.2f} ({'F' if sf else 'ok'}) {arrow} {cm:+.2f} ({'F' if cf else 'ok'})")
    else:
        print(f"    (none)")

    sub_flagged = sum(1 for c in cells if c[3] > 1.0)
    cp_flagged  = sum(1 for c in cells if c[5] > 1.0)
    print(f"\n  Total flag count under unified protocol:")
    print(f"    submission Δ>1pp: {sub_flagged}/{len(cells)}")
    print(f"    cross-purpose Δ>1pp: {cp_flagged}/{len(cells)}")

if __name__ == '__main__':
    main()
