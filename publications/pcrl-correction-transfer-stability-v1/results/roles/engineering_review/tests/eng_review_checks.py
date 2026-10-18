"""Engineering-review invented-data checks (CTS1). No label, feature or person-level file is read: every array is invented.

    python3 results/pcrl_correction_transfer_stability_v1/roles/engineering_review/tests/eng_review_checks.py   (from the worktree root)

Each check prints its id, whether the code under review behaves as the review claims ('demonstrated'), and the numbers.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402

from cts import common as C, rules as RL, trackf as TF, compression as CP   # noqa: E402

OUT = {}


def rec(cid, demonstrated, **kw):
    OUT[cid] = {'demonstrated': bool(demonstrated), **kw}
    print(json.dumps({cid: OUT[cid]}, default=float))


def baseline(rho_guard=0.0005, margin_guard=0.0015, rho_other=0.004, margin_other=0.008):
    b = {}
    for c in RL.OCC_CONDITIONS:
        g = c.startswith('guard|')
        b[c] = {'slack': rho_guard if g else rho_other, 'margin': margin_guard if g else margin_other, 'se_used': 0.0005}
    return b


def t_ens_base_mismatch():
    """ENS deployment: select.realized / dstage.gates measure e_s - e_0 against the FULL witness's margin, so G3 ('the complete
    release passes at the point on MODEL_SELECT') is not checked for the deployed ensemble when e_0 != W."""
    rng = np.random.default_rng(1)
    n = 6000
    y = rng.integers(0, 5, n); sex = rng.integers(0, 2, n); unit = np.arange(n) // 2
    logits = rng.normal(size=(n, 5)); logits[np.arange(n), y] += 1.2
    W = TF.probs(logits)                                              # deployed full-FIT witness
    # fold pipelines: noisier for men (their ensemble is worse than W on the men's guard)
    folds = []
    for k in range(5):
        z = logits + rng.normal(scale=0.35, size=(n, 5)) * (sex == 0)[:, None]
        folds.append(TF.probs(z))
    pf = np.stack(folds)
    dl = rng.normal(scale=0.05, size=(n, 5)); df = np.stack([dl] * 5)
    s = 0.3
    e0 = TF.ensemble_at(pf, df, 0.0)
    es = TF.ensemble_at(pf, df, s)
    base = baseline()
    code = TF.cost_summary(es, e0, y, sex, unit)                       # what select.realized computes for 'RULE|ens'
    complete = TF.cost_summary(es, W, y, sex, unit)                    # increment of the released ensemble over the witness
    o_code = RL.rule_outcome(s, code['stats'], base)
    o_comp = RL.rule_outcome(s, complete['stats'], base)
    shift = TF.cost_summary(e0, W, y, sex, unit)['stats']['men_brier']
    rec('ENS_BASE_G3', o_code['point_cap_crossing'] != o_comp['point_cap_crossing'],
        e0_minus_W_men_brier=shift, code_men_cost=code['stats']['men_brier'], complete_men_cost=complete['stats']['men_brier'],
        margin_guard=base['guard|sex0']['margin'], code_point_cap_crossing=o_code['point_cap_crossing'],
        complete_point_cap_crossing=o_comp['point_cap_crossing'],
        note='increments are additive: complete = (e_s - e_0) + (e_0 - W); the gate drops the second term')
    add = abs(complete['stats']['men_brier'] - (code['stats']['men_brier'] + shift))
    rec('ENS_BASE_ADDITIVE', add < 1e-12, abs_diff=add)


def t_full_complete_equivalence():
    """FULL deployment: witness excess + (r_s - W) increment equals the corrected release's excess vs a reference (G3 exact)."""
    rng = np.random.default_rng(2)
    n = 3000
    y = rng.integers(0, 5, n); sex = rng.integers(0, 2, n); unit = np.arange(n) // 3
    W = TF.probs(rng.normal(size=(n, 5))); U0 = TF.probs(rng.normal(size=(n, 5)))
    rs = TF.release_at(W, rng.normal(scale=0.1, size=(n, 5)), 0.6)
    tv = lambda p: C.task_vectors(y, p, 5)                            # noqa: E731
    exc_w = {m: float((tv(W)[m] - tv(U0)[m]).mean()) for m in ('brier', 'nll')}
    exc_r = {m: float((tv(rs)[m] - tv(U0)[m]).mean()) for m in ('brier', 'nll')}
    inc = TF.cost_summary(rs, W, y, sex, unit)['stats']
    d = max(abs(exc_r['brier'] - exc_w['brier'] - inc['brier']), abs(exc_r['nll'] - exc_w['nll'] - inc['nll']))
    rec('FULL_G3_EXACT', d < 1e-12, max_abs_diff=d)


def t_comp_infinite_target():
    """compression.fit: a statistic with nonpositive reserve slack (D1's men's guard: -0.0008 in the coordinator log) and a
    candidate with positive cost there give target = inf; the bisection then returns the SMALLEST K (2) flagged 'MATCHED'."""
    sslack = {'men_brier': -0.0008, 'women_brier': 0.003, 'brier': 0.004, 'nll': 0.006, 'acc_loss': 0.005}
    cand = {'men_brier': 0.0003, 'women_brier': 0.0002, 'brier': 0.0002, 'nll': 0.0004, 'acc_loss': 0.0}
    target = CP.utilization_of(cand, sslack)
    ladder = list(range(28))
    util = {i: 5.0 / (1 + i) for i in ladder}                        # any finite utilizations (K=2 code costs a lot)
    lo, hi = 0, len(ladder) - 1                                       # the exact search of cts/compression.py lines 70-80
    if util[hi] > target:
        best, flag = hi, 'TARGET_NOT_REACHED_LARGEST_K'
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if util[mid] <= target:
                hi = mid
            else:
                lo = mid + 1
        best, flag = lo, 'MATCHED'
    rec('COMP_INF_TARGET', (not math.isfinite(target)) and best == 0 and flag == 'MATCHED',
        target=str(target), best_index=best, K_if_source_ladder=2, flag=flag, utilization_of_K2=util[0])


def t_entry_point_granularity():
    """_check_entry authorizes by MODULE: any function of cts.trackf (e.g. fit_direction, ladder) passes the 'source_replay'
    entry check, which returns whole-MODEL_FIT labels including the C0 / C1 panels before the freeze."""
    g = {'__name__': 'cts.trackf', 'C': C}
    exec('def fit_direction():\n    return C._check_entry("source_replay")\n', g)
    ok_trackf = g['fit_direction']()
    g2 = {'__name__': 'cts.rstage', 'C': C}
    exec('def lane():\n    return C._check_entry("source_replay")\n', g2)
    try:
        g2['lane'](); refused = False
    except PermissionError:
        refused = True
    rec('ENTRY_MODULE_LEVEL', ok_trackf == 'cts.trackf:fit_direction' and refused, trackf_function_entry=ok_trackf,
        other_module_refused=refused)


def t_d0b_suffix_latent():
    """D0B is declared 'same_rows_as D0' but takes TAG_SUFFIX['D0B'], not D0's: a support-check suffix on D0 would silently
    give D0B different households (latent: PARTITION_SUPPORT.json has no failure)."""
    rng = np.random.default_rng(3)
    unit = np.repeat(np.arange(4000), rng.integers(1, 3, 4000))
    m = C.partition_masks(unit)
    same_now = all(np.array_equal(m['D0'][r], m['D0B'][r]) for r in C.PROLES)
    saved = dict(C.TAG_SUFFIX)
    try:
        C.TAG_SUFFIX['D0'] = 'b'
        m2 = C.partition_masks(unit)
        same_after = all(np.array_equal(m2['D0'][r], m2['D0B'][r]) for r in C.PROLES)
    finally:
        C.TAG_SUFFIX.clear(); C.TAG_SUFFIX.update(saved)
    rec('D0B_SUFFIX_LATENT', same_now and not same_after, same_rows_now=same_now, same_rows_if_D0_suffixed=same_after)


def t_magnitude_definition():
    """trackf.cost_summary's mean_abs_logit_change sums over the 5 classes; RULES section 5 / cts.rules.correction_magnitude
    use the mean over people AND classes: the ladder records are 5x the registered definition."""
    rng = np.random.default_rng(4)
    p = TF.probs(rng.normal(size=(500, 5))); d = rng.normal(size=(500, 5)); s = 0.4
    a = TF.cost_summary(TF.release_at(p, d, s), p, rng.integers(0, 5, 500), rng.integers(0, 2, 500), np.arange(500))
    b = RL.correction_magnitude(TF.logp(p), d, s, p0=p)
    ratio = a['magnitude']['mean_abs_logit_change'] / b['mean_abs_logit_change']
    rec('MAGNITUDE_DEF', abs(ratio - 5.0) < 1e-6, ratio=ratio,
        flip_rate_equal=abs(a['magnitude']['flip_rate'] - b['flip_rate']) < 1e-12)


def t_identity_base():
    """GUARD / CHECK / MS 'full' estimators use the raw witness as base while s = 0 is softmax(logp(W)): the identity cost is
    ~1e-17 rather than exactly 0 (harmless; recorded)."""
    rng = np.random.default_rng(5)
    p = TF.probs(rng.normal(size=(2000, 5))); y = rng.integers(0, 5, 2000)
    cs = TF.cost_summary(TF.release_at(p, rng.normal(size=(2000, 5)), 0.0), p, y, rng.integers(0, 2, 2000), np.arange(2000))
    mx = max(abs(v) for v in cs['stats'].values())
    rec('IDENTITY_BASE', mx < 1e-12, max_abs_identity_cost=mx)


if __name__ == '__main__':
    for f in (t_ens_base_mismatch, t_full_complete_equivalence, t_comp_infinite_target, t_entry_point_granularity,
              t_d0b_suffix_latent, t_magnitude_definition, t_identity_base):
        f()
    p = Path(__file__).with_name('eng_review_checks_RESULT.json')
    p.write_text(json.dumps(OUT, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_demonstrated': all(v['demonstrated'] for v in OUT.values()), 'n': len(OUT)}))
