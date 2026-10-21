"""Coordinator's label-free summary of the family fit records (reads results/<campaign>/da|db/FIT_V<v>.json only).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/inspect_fit.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]


def sp(d, keys=('occ_nll', 'inc_nll', 'tot_nll', 'occ_acc', 'inc_acc', 'occ_brier_men', 'occ_brier_women')):
    return {k: (round(d[k]['estimate'], 4), round(d[k]['se'], 4)) for k in keys if k in d}


for fam in ('da', 'db'):
    for v in (0, 1):
        p = R / fam / f'FIT_V{v}.json'
        if not p.is_file():
            print(fam, v, 'missing'); continue
        r = json.loads(p.read_text())
        s = r['selection']
        key = 'lambda_star' if fam == 'da' else 'alpha_star'
        print(f'== {fam.upper()} V{v}: strength*={s[key]} w={s.get("w")} cpu={r.get("cpu_s", 0):.1f}s sbc={r.get("sbc_attribute") or r.get("sbc_target")}')
        for g, d in s['path'].items():
            print('   path', g, sp(d))
        for t in r.get('sbc_tries', []):
            b = t.get('bracket') or {}
            print('   sbc try', t.get('attribute') or t.get('target'), 'deg' if t.get('degenerate') else '',
                  {g: round(x['spend'], 5) for g, x in (b.get('grid') or {}).items()}, 'cand', round(b.get('cand_spend', 0), 5),
                  'lo/hi', b.get('lo'), b.get('hi'), b.get('flag'), 'scale', t.get('scale_m'))
        pi = r.get('pseudo_info') or {}
        if pi:
            print('   pseudo2 seed', pi.get('chosen_seed'), 'max|corr|', round(pi.get('max_abs_corr', 0), 4), 'accepted', pi.get('accepted'))
        for k, x in (r.get('attribute_profiles') or {}).items():
            print('   profile', k, {kk: round(vv, 4) for kk, vv in x.items() if isinstance(vv, float)})
        b = r.get('spend_path_bracket') or {}
        print('   spend', {g: round(x['spend'], 5) for g, x in (b.get('grid') or {}).items()}, 'lo/hi', b.get('lo'), b.get('hi'), b.get('flag'))
        for cfg, x in (r.get('saved') or {}).items():
            k = 'oof_spend_over_twin' if 'oof_spend_over_twin' in x else 'nested_spend_over_twin'
            print('   saved', cfg, {kk: vv for kk, vv in x.items() if kk in ('lambda', 'alpha', 'delta', 'attribute', 'target', 'bracket_flag', 'reached_cand_spend')},
                  'tot', round(x[k]['tot_nll']['estimate'], 5), 'occ', round(x[k]['occ_nll']['estimate'], 5), 'inc', round(x[k]['inc_nll']['estimate'], 5))
        mf = (r.get('mf_reader_check') or {}).get('auc') or {}
        for cfg, a in mf.items():
            print('   mfcheck', cfg, {k: round(x, 4) for k, x in a.items()})
        if fam == 'da':
            for k, x in (r.get('und_descriptive') or {}).items():
                print('   und', k, x['t_inc'], x['t_occ'], 'tot', round(x['spend']['tot_nll']['estimate'], 5))
