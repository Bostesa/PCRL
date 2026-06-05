"""Re-run attack matrix under SUBMISSION protocol (Δ over best-single-purpose-auditor).

For each cross-purpose checkpoint:
  1. Load encoder, extract h_per[purpose] and h_concat on TRAIN + TEST splits
  2. For each (attr, arch) ∈ {LR, MLP, XGB}:
     - concat_acc = max over AUDITOR_SEEDS of attack_acc(h_concat → attr)  [REUSE from results.json]
     - For each purpose p: single_acc[p] = max over AUDITOR_SEEDS of attack_acc(h_p → attr)
     - best_single_acc = max over purposes of single_acc[p]
     - gain_pp = (concat_acc - best_single_acc) * 100
  3. Aggregate mean ± std across 3 PCRL seeds per (ds, attr, arch).

Submission baseline values are read from paper-body/tables/cross_purpose_attack_extra.tex.

Checkpoint source: s3://pcrl-bios-overnight-20260504/archive/cross_purpose_{ab,diabetes}/checkpoints/
CKPT_DIR env var below points at where best.pt files are staged locally (default /tmp/cp_analysis).
"""
import os, sys, json, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import numpy as np
import torch

CKPT_DIR = Path(os.environ.get('CKPT_DIR', '/tmp/cp_analysis'))
OUT_DIR  = Path(os.environ.get('OUT_DIR',  str(REPO_ROOT / 'results/rebuttal/cross_purpose')))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / 'unified_protocol_results.json'

from scripts.crosspurp.run_eval_multi import (
    build_loaders, load_encoder, extract_reps, extract_attr,
    attack_acc as _attack_acc,
)

AUDITOR_SEEDS = [11, 22, 33]
ARCHS = ["LR", "MLP", "XGB"]

DATASETS = [
    ('adult',    'CROSS_PURPOSE_AB',       3, ['race', 'sex', 'age_group', 'marital_status', 'income']),
    ('hmda',     'CROSS_PURPOSE_AB',       3, ['ethnicity', 'race', 'sex']),
    ('diabetes', 'CROSS_PURPOSE_DIABETES', 3, ['race', 'gender', 'age_bucket']),
]

def best_attack(X_tr, y_tr, X_te, y_te, arch):
    return max(_attack_acc(X_tr, y_tr, X_te, y_te, arch, s) for s in AUDITOR_SEEDS)

results = {}
total_start = time.time()

for ds, tag, n_purposes, attrs in DATASETS:
    print(f"\n{'='*70}\n[{ds.upper()}]")
    results[ds] = {'per_seed': []}
    purposes, train_ds, test_ds, train_loader, test_loader, cfg = build_loaders(ds)
    purpose_names = [p.name for p in purposes]
    input_dim = train_ds.info.num_features
    print(f"  purposes: {purpose_names}, input_dim={input_dim}")

    # Pre-extract attrs once
    train_attrs = {a: extract_attr(train_loader, a) for a in attrs}
    test_attrs  = {a: extract_attr(test_loader, a)  for a in attrs}

    # Load existing cross-purpose run results (concat_acc)
    cp_results = json.load(open(str(REPO_ROOT / f'results/v2_{ds}_{tag}/results.json')))

    for seed in [0, 1, 2]:
        ckpt = str(CKPT_DIR / f'{ds}_s{seed}_best.pt')
        t_seed_start = time.time()
        encoder = load_encoder(Path(ckpt), input_dim, n_purposes, cfg)

        H_tr = {p: extract_reps(encoder, train_loader, i) for i, p in enumerate(purpose_names)}
        H_te = {p: extract_reps(encoder, test_loader, i) for i, p in enumerate(purpose_names)}

        seed_rec = {'seed': seed, 'rows': []}
        cp_seed = cp_results['per_seed'][seed]

        for attr in attrs:
            y_tr = train_attrs[attr]
            y_te = test_attrs[attr]
            for arch in ARCHS:
                t0 = time.time()
                # Single-purpose accs
                single_per_purpose = {}
                for p in purpose_names:
                    acc = best_attack(H_tr[p], y_tr, H_te[p], y_te, arch)
                    single_per_purpose[p] = acc
                best_single = max(single_per_purpose.values())
                best_single_purpose = max(single_per_purpose, key=single_per_purpose.get)
                # Concat acc (reused from results.json — same protocol: best over AUDITOR_SEEDS)
                concat_acc = cp_seed['attack'][arch][attr]['attack_acc']
                majority = cp_seed['attack'][arch][attr]['majority']
                gain_pp = (concat_acc - best_single) * 100
                concat_delta_pp = (concat_acc - majority) * 100
                single_delta_pp = (best_single - majority) * 100
                elapsed = time.time() - t0

                row = {
                    'attribute': attr, 'arch': arch, 'majority': majority,
                    'concat_acc': concat_acc, 'best_single_acc': best_single,
                    'best_single_purpose': best_single_purpose,
                    'single_per_purpose': single_per_purpose,
                    'gain_pp': gain_pp,
                    'concat_delta_pp': concat_delta_pp,
                    'single_delta_pp': single_delta_pp,
                }
                seed_rec['rows'].append(row)
                print(f"  [s{seed}] {attr:>14} {arch:>3}: concat={concat_acc:.3f}(Δ{concat_delta_pp:+.2f}) "
                      f"best_single={best_single:.3f}(Δ{single_delta_pp:+.2f}, via {best_single_purpose}) "
                      f"gain={gain_pp:+.2f}pp [{elapsed:.1f}s]")
                # incremental save
                with open(str(OUT_JSON), 'w') as f:
                    json.dump(results, f, indent=2, default=float)
        results[ds]['per_seed'].append(seed_rec)
        with open(str(OUT_JSON), 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  [s{seed}] seed done in {time.time()-t_seed_start:.1f}s")

# Aggregate
print(f"\n{'='*70}\nAGGREGATE (mean ± std across 3 PCRL seeds)\n{'='*70}")
agg = {}
for ds in results:
    by_key = {}
    for sr in results[ds]['per_seed']:
        for r in sr['rows']:
            k = (r['attribute'], r['arch'])
            by_key.setdefault(k, []).append(r['gain_pp'])
    agg[ds] = {}
    for (attr, arch), gains in sorted(by_key.items()):
        g = np.array(gains)
        agg[ds][f'{attr}/{arch}'] = {
            'gain_mean_pp': float(g.mean()),
            'gain_std_pp':  float(g.std(ddof=0)),
            'gains_per_seed': gains,
            'flag_above_1pp': float(g.mean()) > 1.0,
        }
        print(f"  {ds:>9} {attr:>14} {arch:>3}: {g.mean():+6.2f} ± {g.std(ddof=0):4.2f} pp  {'FLAG' if g.mean()>1.0 else 'ok'}")
with open(str(OUT_JSON), 'w') as f:
    json.dump({'per_seed': results, 'aggregate': agg}, f, indent=2, default=float)

print(f"\nTotal wall: {time.time()-total_start:.1f}s")
