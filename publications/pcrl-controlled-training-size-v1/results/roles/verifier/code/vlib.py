"""Independent verifier library of the PCRL controlled training-size campaign (TSZ1; EXPLORATORY DEVELOPMENT on previously used
records, not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records (housing-unit residents aged 25-64,
civilian employed), development roles of the protected-tree replication. Income (2 classes) and occupation (5 groups) are predicted
from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP); each recipient receives only its own probability vector and argmax decision.
SEX (sex 0 = SEX code 1 = men, sex 1 = SEX code 2 = women) is a training label on MODEL_FIT and an evaluation label on MODEL_SELECT,
never a runtime input.

Written by the independent verifier from the TEXT of PROTOCOL.md, SUBSETS.json / READER_SPLITS.json definitions and
roles/statistical_reviewer/RULES.md. Nothing here imports the campaign's subset, metric, reserve, contrast, trigger, selection or
stability functions (tsz.common helpers, tsz.rules, tsz.evaluate, ohn.rules, ihs.evaluate). This module reads no data and no label;
every label read happens in the verifier's own __main__ scripts through tsz.common.load(operation='verify').

Contents
  hashing      file_sha, arr_sha (SHA-256 of dtype + shape + C bytes), hh_set_sha, git_blob_sha
  structure    order_households, n_households, subset_masks, folds (SHA-256('<tag>|<household>') order, dealt round-robin)
  encoding     encode_raw (from a package encoder spec), apply_dictionary, fit_only_dictionary, compact_po, softmax
  metrics      person_vectors, cluster_se, subgroup_se, clauses_and_guards, reserve
  contrasts    paired_change, summarize_orderings, version_label, across_versions
  rules        trigger, select_strength, stability
  records      append_checks, add_check_change, write_record, note
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
VER_DIR = CODE_DIR.parent
RESULTS = VER_DIR.parent.parent
ROOT = RESULTS.parent.parent
# stage V6 (restored deployment): the code and data under test may be a restored copy (TSZ_VERIFIER_ROOT) while the records are
# written to the original verifier folder (TSZ_VERIFIER_RECORD_DIR)
if os.environ.get('TSZ_VERIFIER_ROOT'):
    ROOT = Path(os.environ['TSZ_VERIFIER_ROOT']).resolve()
    RESULTS = ROOT / 'results' / 'pcrl_controlled_training_size_v1'
if os.environ.get('TSZ_VERIFIER_RECORD_DIR'):
    VER_DIR = Path(os.environ['TSZ_VERIFIER_RECORD_DIR']).resolve()
CHECKS = VER_DIR / 'CHECKS.json'
CHECK_CHANGES = VER_DIR / 'CHECK_CHANGES.json'
RECORDS = VER_DIR / 'records'
HISTORY = RECORDS / 'history'
NOTES = RECORDS / 'NOTES.jsonl'

K_OCC = 5
NLL_FLOOR = 1e-12
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD_CAP = 0.005
K_RESERVE = 2.0
FLOOR_HH = 5306
OCC_SIDE = ('occupation|U0|accuracy_loss', 'occupation|U0|nll_excess', 'occupation|U0|brier_excess',
            'occupation|Ucal|accuracy_loss', 'occupation|Ucal|nll_excess', 'occupation|Ucal|brier_excess',
            'guard|sex0', 'guard|sex1')
INCOME_SIDE = tuple(f'income|{r}|{m}' for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess'))
ALL_CONDITIONS = INCOME_SIDE + OCC_SIDE
METRICS5 = ('acc_loss', 'nll', 'brier', 'men_brier', 'women_brier')
Z = 2.0
STRENGTHS = (0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
STATS5 = ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')
ORDERINGS = ('O0', 'O1', 'O2', 'O3')
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')
FIELDS6 = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
OHN1_RULES = 'results/pcrl_occupation_headroom_overnight_v1/roles/statistical_reviewer/RULES.json'
OHN1_RULES_SHA = 'e18f1c68f4646ce9d1395960ab1a364a80c12541b815a55e0ba06b65074cbbb3'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


# ================================================================ hashing
def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arr_sha(a):
    """SHA-256 of str(dtype) + str(shape) + C-order bytes (the campaign's record definition of an array / row-set hash)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def hh_set_sha(households):
    """SHA-256 of the comma-joined ascending household ids (SUBSETS.json / READER_SPLITS.json household_set_sha256)."""
    hh = sorted(int(h) for h in set(np.asarray(households).tolist()))
    return hashlib.sha256(','.join(str(h) for h in hh).encode()).hexdigest()


def git_blob_sha(data: bytes):
    """git's blob id: SHA-1 of b'blob <len>\\0' + content (independent of the git binary)."""
    return hashlib.sha1(b'blob %d\x00' % len(data) + data).hexdigest()


def payload_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ================================================================ structure (label-free)
def _rank(groups, tag):
    """Households sorted by SHA-256('<tag>|<household id>') (hex digest, ascending)."""
    return sorted(groups, key=lambda g: hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest())


def order_households(unit, o):
    groups = sorted(set(int(g) for g in np.asarray(unit).tolist()))
    return _rank(groups, f'TSZ1_order_{o}')


def n_households(G, f):
    """ceil(f / 100 x G) in exact integer arithmetic; f = 100 -> G."""
    return G if f == 100 else -(-(f * G) // 100)


def subset_masks(unit):
    """{sid: boolean mask over MODEL_FIT people} for S50_O<o>, S70_O<o>, S100 (whole households, nested per ordering)."""
    unit = np.asarray(unit)
    G = len(set(unit.tolist()))
    out = {'S100': np.ones(len(unit), dtype=bool)}
    for o in ORDERINGS:
        order = order_households(unit, o)
        for f in (50, 70):
            keep = set(order[:n_households(G, f)])
            out[f'S{f}_{o}'] = np.fromiter((int(u) in keep for u in unit.tolist()), dtype=bool, count=len(unit))
    return out


def folds(unit, k, tag):
    """Fold id per person: households ranked by SHA-256('<tag>|<household>') and dealt round-robin into k folds."""
    unit = np.asarray(unit)
    groups = sorted(set(int(g) for g in unit.tolist()))
    ranked = _rank(groups, tag)
    fold_of = {g: i % k for i, g in enumerate(ranked)}
    return np.fromiter((fold_of[int(u)] for u in unit.tolist()), dtype=np.int64, count=len(unit))


def reader_roles(unit, rep_tag):
    """READER_SPLITS / RULES.md section 9: 10 parts; 0-4 FIT, 5-6 VALIDATION, 7-9 SCORE."""
    part = folds(unit, 10, rep_tag)
    return {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


# ================================================================ encoding (from a package encoder spec; own code)
def encode_raw(raw, enc, n=None):
    """Own vectorised raw-field encoder: numeric (v - mean) / sd in float64; categorical one-hot of str(value).strip() with unknown
    levels -> <block>=__OTHER__; returned as float32 (the encoder contract)."""
    names = [str(x) for x in enc['feature_names']]
    col = {nm: i for i, nm in enumerate(names)}
    n = len(raw['AGEP']) if n is None else n
    X = np.zeros((n, len(names)), dtype=np.float64)
    for c in ('AGEP', 'WKHP'):
        v = np.array([float(x) for x in raw[c]], dtype=np.float64)
        if not np.isfinite(v).all():
            raise ValueError(f'non-finite {c}')
        X[:, col[c]] = (v - float(enc['numeric'][c]['mean'])) / float(enc['numeric'][c]['sd'])
    for c in BLOCKS:
        other = col[f'{c}=__OTHER__']
        idx = np.array([col.get(f'{c}={str(x).strip()}', other) for x in raw[c]], dtype=np.int64)
        X[np.arange(n), idx] = 1.0
    return X.astype(np.float32), names


def fit_only_dictionary(X_train, names, min_count=10):
    """PROTOCOL section 5: a COW / SCHL / MAR / POBP level with fewer than `min_count` training people moves to its block's
    __OTHER__ column. Returns {block: [collapsed level names]} (levels other than __OTHER__ only)."""
    names = [str(x) for x in names]
    X_train = np.asarray(X_train)
    out = {}
    for b in BLOCKS:
        out[b] = [nm for i, nm in enumerate(names) if nm.startswith(b + '=') and nm != f'{b}=__OTHER__'
                  and float(np.sum(X_train[:, i], dtype=np.float64)) < min_count]
    return out


def apply_dictionary(X, names, collapsed):
    names = [str(x) for x in names]
    X = np.array(X, dtype=np.float64, copy=True)
    for b in BLOCKS:
        other = names.index(f'{b}=__OTHER__')
        for lvl in collapsed.get(b, []):
            i = names.index(lvl)
            on = X[:, i] == 1
            X[on, i] = 0.0
            X[on, other] = 1.0
    return X


def compact_po(X, names):
    """compact_po from the recipe text: AGEP, WKHP numeric; COW, MAR native category codes (index inside the one-hot block);
    SCHL ordinal admitted code (__OTHER__ = 3); POBP numeric place-of-birth code (__OTHER__ = 999). Columns in that order."""
    names = [str(x) for x in names]
    X = np.asarray(X, dtype=np.float64)
    cols = [X[:, names.index('AGEP')], X[:, names.index('WKHP')]]
    for b in BLOCKS:
        idx = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        Zb = X[:, idx]
        if not (np.all((Zb == 0) | (Zb == 1)) and np.all(Zb.sum(1) == 1)):
            raise ValueError(f'{b}: incomplete one-hot block')
        code = np.argmax(Zb, axis=1)
        if b in ('COW', 'MAR'):
            cols.append(code.astype(np.float64))
        else:
            other = 3.0 if b == 'SCHL' else 999.0
            vals = np.array([other if names[i].endswith('__OTHER__') else float(names[i].split('=', 1)[1]) for i in idx])
            cols.append(vals[code])
    return np.column_stack(cols)


def softmax(raw):
    z = np.asarray(raw, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ================================================================ metrics (RULES.md sections 0-3, from the text)
def person_vectors(y, p, k):
    """accuracy 1[argmax p = y] (first maximum), NLL = -log(max(p_y, 1e-12)), summed Brier = sum_j (p_j - 1[y = j])^2."""
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    n = len(y)
    if p.shape != (n, k) or not np.isfinite(p).all() or (p < 0).any() or np.max(np.abs(p.sum(1) - 1.0)) > 1e-9:
        raise ValueError('not a probability-vector release of the right shape')
    idx = np.arange(n)
    onehot = np.zeros((n, k)); onehot[idx, y] = 1.0
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[idx, y], NLL_FLOOR)),
            'brier': np.sum((p - onehot) ** 2, axis=1)}


def _cluster_sums(values, unit):
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    inv = inv.reshape(-1)
    G = int(inv.max()) + 1
    s = np.zeros(G, dtype=np.float64)
    np.add.at(s, inv, np.asarray(values, dtype=np.float64))
    return s, G


def cluster_se(d, unit):
    """Household-clustered SE of a mean: sqrt(G / (G - 1) x sum_g (sum_{i in g} (d_i - dbar))^2) / n."""
    d = np.asarray(d, dtype=np.float64)
    s, G = _cluster_sums(d - d.mean(), unit)
    return math.sqrt(G / (G - 1) * float(np.dot(s, s))) / d.size


def subgroup_se(d, mask, unit):
    """Linearized household-clustered SE of a subgroup mean (ratio of two means), computed on the whole set:
    z_i = 1[i in group] (d_i - m_group) / share_group; SE = sqrt(G / (G - 1) x sum_g (sum_{i in g} z_i)^2) / n."""
    d = np.asarray(d, dtype=np.float64); mask = np.asarray(mask, dtype=bool)
    m = float(d[mask].mean()); share = float(mask.mean())
    zi = np.where(mask, (d - m) / share, 0.0)
    s, G = _cluster_sums(zi, unit)
    return math.sqrt(G / (G - 1) * float(np.dot(s, s))) / d.size


def clauses_and_guards(labels, own, refs, sex, unit):
    """One version's 12 clauses and 2 guards. labels {'income','occupation'}: int arrays; own {'income': (n,2), 'occupation': (n,5)};
    refs {'U0': {...}, 'Ucal': {...}}. Clause value = mean of the per-person paired difference (equals the difference of means up to
    rounding); SE = cluster_se of that difference. Guards: within-SEX mean of occupation Brier(own) - Brier(U0), subgroup_se."""
    K = {'income': 2, 'occupation': 5}
    out = {}
    pv = {t: person_vectors(labels[t], own[t], K[t]) for t in K}
    for t in ('income', 'occupation'):
        for r in ('U0', 'Ucal'):
            rv = person_vectors(labels[t], refs[r][t], K[t])
            diffs = {'accuracy_loss': rv['accuracy'] - pv[t]['accuracy'], 'nll_excess': pv[t]['nll'] - rv['nll'],
                     'brier_excess': pv[t]['brier'] - rv['brier']}
            for m, d in diffs.items():
                out[f'{t}|{r}|{m}'] = {'value': float(d.mean()), 'se': cluster_se(d, unit), 'cap': CAPS[m],
                                       'value_as_difference_of_means': float({'accuracy_loss': rv['accuracy'].mean() - pv[t]['accuracy'].mean(),
                                                                              'nll_excess': pv[t]['nll'].mean() - rv['nll'].mean(),
                                                                              'brier_excess': pv[t]['brier'].mean() - rv['brier'].mean()}[m])}
    u0 = person_vectors(labels['occupation'], refs['U0']['occupation'], 5)
    d = pv['occupation']['brier'] - u0['brier']
    sex = np.asarray(sex)
    for g, nm in ((0, 'sex0'), (1, 'sex1')):
        msk = sex == g
        out[f'guard|{nm}'] = {'value': float(d[msk].mean()), 'se': subgroup_se(d, msk, unit), 'cap': GUARD_CAP}
    for c in out.values():
        c['point_pass'] = bool(math.isfinite(c['value']) and c['value'] <= c['cap'])
    return out


def reserve(conds, floor_v, households, names=OCC_SIDE, k=K_RESERVE):
    """RULES.md section 1 from the text: margin = cap - value; se_used = max(own SE, floor x sqrt(5306 / G)); met iff margin >= k x
    se_used (inclusive); ratio = margin / se_used; rho = margin - k x se_used. Own SE 0 or below the floor -> floor; missing /
    non-finite / non-positive floor -> error; negative own SE -> error; non-finite value or own SE -> the condition fails."""
    scale = math.sqrt(FLOOR_HH / float(households))
    out = {}
    for c in names:
        x = conds[c]
        fl = floor_v.get(c)
        if fl is None or not math.isfinite(float(fl)) or float(fl) <= 0:
            raise ValueError(f'{c}: invalid floor {fl!r}')
        fl = float(fl) * scale
        own = float(x['se']) if x.get('se') is not None else math.nan
        val = float(x['value'])
        if math.isfinite(own) and own < 0:
            raise ValueError(f'{c}: negative SE')
        rec = {'value': val, 'cap': float(x['cap']), 'own_se': own, 'floor_se': fl, 'point_pass': bool(math.isfinite(val) and val <= x['cap'])}
        if not (math.isfinite(own) and math.isfinite(val)):
            rec.update({'margin': None, 'se_used': None, 'ratio': None, 'rho': None, 'met': False, 'se_source': None})
        else:
            margin = float(x['cap']) - val
            su = max(own, fl)
            rec.update({'margin': margin, 'se_used': su, 'ratio': margin / su, 'rho': margin - k * su, 'met': bool(margin >= k * su),
                        'se_source': 'own' if own >= fl else 'floor'})
        out[c] = rec
    ratios = {c: (r['ratio'] if r['ratio'] is not None else -math.inf) for c, r in out.items()}
    binding = min(ratios, key=lambda c: ratios[c])
    return {'conditions': out, 'met': all(r['met'] for r in out.values()), 'failed': [c for c, r in out.items() if not r['met']],
            'point_pass': all(r['point_pass'] for r in out.values()), 'point_failed': [c for c, r in out.items() if not r['point_pass']],
            'min_ratio': ratios[binding], 'binding': binding, 'floor_scale': scale}


# ================================================================ paired contrasts (RULES.md section 3)
def loss_vectors(y, p, sex):
    """Per-person losses on the common people: error indicator, NLL, summed Brier; SEX masks."""
    v = person_vectors(y, p, K_OCC)
    sex = np.asarray(sex)
    return {'acc_loss': 1.0 - v['accuracy'], 'nll': v['nll'], 'brier': v['brier'], 'men': sex == 0, 'women': sex == 1}


def paired_change(la, lb, unit):
    """change(a -> b) = loss(b) - loss(a) per metric (negative = b better), with household-clustered SEs (subgroup SE for the SEX
    groups). Returns {metric: {'value', 'se'}} and the per-person difference vectors (for double differences)."""
    out, vec = {}, {}
    for m in ('acc_loss', 'nll', 'brier'):
        d = lb[m] - la[m]
        out[m] = {'value': float(d.mean()), 'se': cluster_se(d, unit)}; vec[m] = d
    d = lb['brier'] - la['brier']
    out['men_brier'] = {'value': float(d[la['men']].mean()), 'se': subgroup_se(d, la['men'], unit)}
    out['women_brier'] = {'value': float(d[la['women']].mean()), 'se': subgroup_se(d, la['women'], unit)}
    vec['men_brier'] = d; vec['women_brier'] = d
    return out, vec


def diff_stats(d, men, women, unit):
    """Mean / SE of a per-person (double) difference vector for the five metrics given per-metric vectors d[m]."""
    out = {}
    for m in ('acc_loss', 'nll', 'brier'):
        out[m] = {'value': float(d[m].mean()), 'se': cluster_se(d[m], unit)}
    out['men_brier'] = {'value': float(d['men_brier'][men].mean()), 'se': subgroup_se(d['men_brier'], men, unit)}
    out['women_brier'] = {'value': float(d['women_brier'][women].mean()), 'se': subgroup_se(d['women_brier'], women, unit)}
    return out


def summarize_orderings(xs, ses):
    """RULES.md 3.3: mean, SD_ord (df 3), range, sign reversals vs the mean's sign, se_eval = sqrt(mean s^2), se_comb =
    sqrt(se_eval^2 + SD_ord^2 / 4); one value -> se_comb = its own SE. Label (improvement orientation) per version."""
    xs = [None if x is None else float(x) for x in xs]
    ses = [None if s is None else float(s) for s in ses]
    if not xs or any(x is None or not math.isfinite(x) for x in xs) or any(s is None or not math.isfinite(s) for s in ses):
        return {'label': 'INCOMPLETE', 'values': xs, 'ses': ses}
    n = len(xs)
    xb = sum(xs) / n
    if n == 1:
        sd, se_eval, se_comb = 0.0, ses[0], ses[0]
    else:
        sd = math.sqrt(sum((x - xb) ** 2 for x in xs) / (n - 1))
        se_eval = math.sqrt(sum(s * s for s in ses) / n)
        se_comb = math.sqrt(se_eval ** 2 + sd ** 2 / n)
    sign_mean = (xb > 0) - (xb < 0)
    reversals = sum(1 for x in xs if ((x > 0) - (x < 0)) != sign_mean)
    if xb - Z * se_comb > 0 and all(x > 0 for x in xs):
        lab = 'RESOLVED_POSITIVE'
    elif xb + Z * se_comb < 0 and all(x < 0 for x in xs):
        lab = 'RESOLVED_NEGATIVE'
    else:
        lab = 'UNRESOLVED'
    flags = ['SIGN_REVERSAL'] if (any(x > 0 for x in xs) and any(x < 0 for x in xs)) else []
    var_share = (sd ** 2 / (sd ** 2 + n * se_eval ** 2)) if n > 1 and (sd > 0 or se_eval > 0) else None
    return {'label': lab, 'flags': flags, 'mean': xb, 'sd_ord': sd, 'range': [min(xs), max(xs)], 'sign_reversals': reversals,
            'se_eval': se_eval, 'se_comb': se_comb, 'values': xs, 'ses': ses, 'variance_share_ordering': var_share}


def across_versions(l0, l1):
    labs = (l0, l1)
    if 'INCOMPLETE' in labs:
        return 'INCOMPLETE'
    if labs == ('RESOLVED_POSITIVE', 'RESOLVED_POSITIVE'):
        return 'POSITIVE_BOTH_VERSIONS'
    if labs == ('RESOLVED_NEGATIVE', 'RESOLVED_NEGATIVE'):
        return 'NEGATIVE_BOTH_VERSIONS'
    resolved = [x for x in labs if x.startswith('RESOLVED')]
    if resolved:
        return 'VERSION_DEPENDENT'
    return 'NOT_RESOLVED'


# ================================================================ trigger (RULES.md section 7)
def trigger(res_by_v, t0_ok_by_v, deploy_by_v):
    """res_by_v[v] = own reserve() over OCC_SIDE; t0_ok_by_v[v] = (all 14 conditions finite and H3 hash ok); deploy_by_v[v] =
    {'bitwise': bool, 'refusals_all_pass': bool} (None if the record is absent)."""
    parts = {}
    parts['T0'] = all(t0_ok_by_v.get(v) is True for v in (0, 1))
    parts['T1'] = parts['T0'] and all(res_by_v[v]['point_pass'] for v in (0, 1))
    parts['T2'] = parts['T0'] and all(res_by_v[v]['met'] for v in (0, 1))
    parts['T3'] = all(isinstance(deploy_by_v.get(v), dict) and deploy_by_v[v].get('bitwise') is True
                      and deploy_by_v[v].get('refusals_all_pass') is True for v in (0, 1))
    fires = all(parts.values())
    return {'decision': 'TRIGGER_FIRES' if fires else 'TRIGGER_DOES_NOT_FIRE', 'parts': parts,
            'failing_parts': [k for k, ok in parts.items() if not ok]}


# ================================================================ strength selection (RULES.md section 8.3)
def select_strength(est, se, allowance, strengths=STRENGTHS, z=Z):
    """est[s][j], se[s][j] for nonzero s; pass at s iff for all 5 statistics est + z se <= allowance (inclusive), non-finite fails;
    s* = largest s with every nonzero strength <= s passing (fixed ascending sequence); PASS_AFTER_FAIL flag for later passes."""
    trail, selected, first_fail, binding = {}, 0.0, None, None
    passes = {}
    for s in strengths:
        if s == 0:
            continue
        ok_all, row = True, {}
        for j in STATS5:
            e, sj = est[s][j], se[s][j]
            if e is None or sj is None or not (math.isfinite(e) and math.isfinite(sj)):
                u, ok = None, False
            else:
                u = e + z * sj
                ok = u <= allowance[j]
            row[j] = {'estimate': e, 'se': sj, 'U': u, 'allowance': allowance[j],
                      'U_over_allowance': (u / allowance[j]) if u is not None else None, 'pass': bool(ok)}
            ok_all = ok_all and ok
        trail[s] = row
        passes[s] = ok_all
    for s in strengths:
        if s == 0:
            continue
        if passes[s]:
            selected = s
        else:
            first_fail = s
            worst = max(STATS5, key=lambda j: (trail[s][j]['U_over_allowance'] if trail[s][j]['U_over_allowance'] is not None else math.inf))
            binding = worst
            break
    later = [s for s in strengths if s != 0 and first_fail is not None and s > first_fail and passes[s]]
    return {'selected': selected, 'first_failing_strength': first_fail, 'binding_statistic': binding,
            'flags': ['PASS_AFTER_FAIL'] if later else [], 'pass_after_fail_strengths': later, 'passes': passes, 'trail': trail}


# ================================================================ stability (RULES.md section 9)
def stability(cells, ses, threshold=0.0):
    """RULES.md section 9 (final text). cells, ses: {cell name: value} for the 8 cells S<v>R<r>k<k>.
    k_differs per (v, R) = (b(k0) != b(k1)) exactly; n_eff = 4 + number of differing pairs. d_v = mean of version v's 4 cells;
    SS = sum over the DISTINCT cells of (b - d_v)^2 (an equal pair enters once); v_rep = SS / (n_eff - 2); v_samp = mean of the 8
    squared bootstrap SEs; sd_total = sqrt(v_rep + v_samp). Labels (threshold 0 or 0.020): STABLE_POSITIVE (all 8 > threshold and
    d_v - 2 sd_total > threshold for both versions), POSITIVE_NOT_STABLE, SIGN_CHANGES (8-cell mean > threshold, some cell <=),
    NOT_POSITIVE, INCOMPLETE. Qualifier: READER_SEEDS_EFFECTIVE_ALL_PAIRS (8) / READER_SEEDS_PARTLY_EFFECTIVE (5-7) /
    READER_SEED_REPLICATION_ABSENT (4)."""
    names = [f'S{v}R{r}k{k}' for v in (0, 1) for r in (1, 2) for k in (0, 1)]
    vals = [cells.get(n) for n in names]
    sv = [ses.get(n) for n in names]
    if any(x is None or not math.isfinite(x) for x in vals) or any(s is None or not math.isfinite(s) for s in sv):
        return {'label': 'INCOMPLETE'}
    b = dict(zip(names, vals))
    d = {v: sum(b[n] for n in names if n.startswith(f'S{v}')) / 4.0 for v in (0, 1)}
    k_differs = {f'S{v}R{r}': bool(b[f'S{v}R{r}k0'] != b[f'S{v}R{r}k1']) for v in (0, 1) for r in (1, 2)}
    n_eff = 4 + sum(k_differs.values())
    ss = 0.0
    for v in (0, 1):
        for r in (1, 2):
            if k_differs[f'S{v}R{r}']:
                ss += (b[f'S{v}R{r}k0'] - d[v]) ** 2 + (b[f'S{v}R{r}k1'] - d[v]) ** 2
            else:
                ss += (b[f'S{v}R{r}k0'] - d[v]) ** 2
    v_rep = ss / (n_eff - 2)
    v_samp = sum(s * s for s in sv) / 8.0
    sd = math.sqrt(v_rep + v_samp)
    mean8 = sum(vals) / 8.0
    allpos = all(x > threshold for x in vals)
    bound = {v: d[v] - Z * sd for v in (0, 1)}
    if allpos and all(bound[v] > threshold for v in (0, 1)):
        lab = 'STABLE_POSITIVE'
    elif allpos:
        lab = 'POSITIVE_NOT_STABLE'
    elif mean8 > threshold:
        lab = 'SIGN_CHANGES'
    else:
        lab = 'NOT_POSITIVE'
    qual = 'READER_SEEDS_EFFECTIVE_ALL_PAIRS' if n_eff == 8 else ('READER_SEED_REPLICATION_ABSENT' if n_eff == 4 else 'READER_SEEDS_PARTLY_EFFECTIVE')
    return {'label': lab, 'qualifier': qual, 'mean': mean8, 'min': min(vals), 'max': max(vals), 'd_v': d, 'v_rep': v_rep,
            'v_samp': v_samp, 'sd_total': sd, 'bound': bound, 'n_not_above': sum(1 for x in vals if x <= threshold),
            'n_eff': n_eff, 'k_differs': k_differs, 'cells': b}


def target_label(cells, ses):
    """C6: the same computation at threshold 0.020: DEV_GRID_TARGET_STABLE / _POINT_ONLY (8-cell mean > 0.020) / _NOT_MET."""
    s = stability(cells, ses, threshold=0.020)
    if s['label'] == 'INCOMPLETE':
        return 'INCOMPLETE', s
    if s['label'] == 'STABLE_POSITIVE':
        return 'DEV_GRID_TARGET_STABLE', s
    if s['mean'] > 0.020:
        return 'DEV_GRID_TARGET_POINT_ONLY', s
    return 'DEV_GRID_TARGET_NOT_MET', s


# ================================================================ size-stage labels (RULES.md sections 3.5, 4, 5, 6; from the text)
def margin_pattern(met100_by_v, met50, met70):
    """met100_by_v {v: bool}; met50 / met70 {(o, v): bool} (8 cells each)."""
    vals = list(met100_by_v.values()) + list(met50.values()) + list(met70.values())
    if len(met100_by_v) != 2 or len(met50) != 8 or len(met70) != 8 or not all(isinstance(x, bool) for x in vals):
        return {'pattern': 'INCOMPLETE', 'flags': []}
    met100 = all(met100_by_v.values())
    n50, n70 = sum(met50.values()), sum(met70.values())
    flags = []
    if not met100:
        pat = 'FULL_NOT_REPRODUCED'
    elif n50 < 8:
        pat = 'FULL_ONLY'
        if n70 == 8:
            flags.append('FULL_ONLY_70_RETAINS')
    elif n70 == 8:
        pat = 'ALL_SIZES'
    else:
        pat = 'FULL_ONLY'
        flags.append('NON_NESTED_PATTERN_50_RETAINS_70_LOSES')
    return {'pattern': pat, 'flags': flags, 'met100': met100, 'n50': n50, 'n70': n70}


def size_effect(across_label):
    return {'POSITIVE_BOTH_VERSIONS': 'SIZE_GAIN_RESOLVED', 'NEGATIVE_BOTH_VERSIONS': 'SIZE_LOSS_RESOLVED',
            'INCOMPLETE': 'INCOMPLETE'}.get(across_label, 'SIZE_EFFECT_UNRESOLVED')


def size_explanation(pattern, effect):
    """Text order: SUPPORTED (FULL_ONLY and GAIN); NOT_SUPPORTED (ALL_SIZES or LOSS); NOT_APPLICABLE (FULL_NOT_REPRODUCED);
    UNRESOLVED otherwise. The only overlap the text leaves open (FULL_NOT_REPRODUCED with LOSS) is resolved to NOT_APPLICABLE
    (a full-data margin that does not reproduce leaves no size explanation to support or reject); flagged if it occurs."""
    if pattern == 'INCOMPLETE' or effect == 'INCOMPLETE':
        return 'INCOMPLETE', []
    flags = ['TEXT_OVERLAP_FULL_NOT_REPRODUCED_AND_LOSS'] if (pattern == 'FULL_NOT_REPRODUCED' and effect == 'SIZE_LOSS_RESOLVED') else []
    if pattern == 'FULL_NOT_REPRODUCED':
        return 'NOT_APPLICABLE_FULL_NOT_REPRODUCED', flags
    if pattern == 'FULL_ONLY' and effect == 'SIZE_GAIN_RESOLVED':
        return 'SUPPORTED_ON_THIS_POOL', flags
    if pattern == 'ALL_SIZES' or effect == 'SIZE_LOSS_RESOLVED':
        return 'NOT_SUPPORTED', flags
    return 'UNRESOLVED', flags


def control_level_label(across_label):
    return {'POSITIVE_BOTH_VERSIONS': 'RECIPE_BETTER_RESOLVED', 'NEGATIVE_BOTH_VERSIONS': 'RECIPE_WORSE_RESOLVED',
            'VERSION_DEPENDENT': 'VERSION_DEPENDENT', 'NOT_RESOLVED': 'NOT_RESOLVED'}.get(across_label, 'INCOMPLETE')


def control_explains(recipe_gain, control_gain, did):
    """Per metric, 50 -> 100, across-version labels of the recipe's gain, the control's gain and the DiD."""
    if 'INCOMPLETE' in (recipe_gain, control_gain, did):
        return 'INCOMPLETE'
    if recipe_gain != 'POSITIVE_BOTH_VERSIONS':
        return 'NO_RECIPE_SIZE_GAIN'
    if did == 'POSITIVE_BOTH_VERSIONS':
        return 'RECIPE_SPECIFIC_SIZE_GAIN'
    if control_gain == 'POSITIVE_BOTH_VERSIONS':
        return 'CONTROL_EXPLAINS_GAIN'
    return 'RECIPE_GAIN_CONTROL_UNRESOLVED'


def control_also_meets(recipe_met, control_met):
    """{(o, v): bool} each; RULES.md section 4 text: True iff the control meets the reserve in every cell at which A1_PO_16 meets
    it (vacuously True when A1_PO_16 meets it nowhere; the vacuous case is reported separately by the caller)."""
    return all(control_met.get(cell) is True for cell, m in recipe_met.items() if m)


def reproduction_label(per_version):
    """per_version[v] = {'bitwise': bool (every role), 'max_abs': float, 'decisions_identical': bool, 'tstar_equal': bool}."""
    if len(per_version) != 2:
        return 'INCOMPLETE'
    if all(x['bitwise'] and x['tstar_equal'] for x in per_version.values()):
        return 'REPRODUCED_BITWISE'
    if all(x['max_abs'] <= 1e-12 and x['decisions_identical'] and x['tstar_equal'] for x in per_version.values()):
        return 'REPRODUCED_WITHIN_TOLERANCE'
    return 'NOT_REPRODUCED'


def encoding_label(version_label):
    return {'RESOLVED_POSITIVE': 'ENCODING_EFFECT_RESOLVED', 'RESOLVED_NEGATIVE': 'ENCODING_EFFECT_RESOLVED_NEGATIVE',
            'UNRESOLVED': 'ENCODING_EFFECT_UNRESOLVED'}.get(version_label, 'INCOMPLETE')


# ================================================================ append-only records
def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        raise TypeError('person-level arrays are never written to verifier records')
    if isinstance(o, Path):
        return str(o)
    return str(o)


def inputs_sha(paths):
    """{repository-relative path: SHA-256} for the files a check read (absolute paths outside the repository kept as given)."""
    out = {}
    for p in paths:
        p = Path(p)
        try:
            key = str(p.resolve().relative_to(ROOT))
        except ValueError:
            key = str(p)
        out[key] = file_sha(p) if p.is_file() else None
    return out


def append_checks(new_checks):
    """Append checks to CHECKS.json (never edits or removes an earlier check) and snapshot the new version to records/history/."""
    RECORDS.mkdir(parents=True, exist_ok=True); HISTORY.mkdir(parents=True, exist_ok=True)
    lock = RECORDS / '.checks.lock'
    with open(lock, 'a+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if CHECKS.is_file():
                doc = json.loads(CHECKS.read_text())
            else:
                doc = {'schema': 'TSZ1-verifier-checks-v1', 'campaign': 'pcrl_controlled_training_size_v1',
                       'role': 'independent verifier (own code under roles/verifier/code/; labels only via tsz.common.load operation verify)',
                       'rules': ['append-only: a check entry is never edited or removed; every version is snapshotted to records/history/',
                                 'kind NUMERICAL_REPLAY (own code recomputes a recorded number) and PROCEDURAL_COMPLIANCE (process, '
                                 'access and record evidence) are kept separate',
                                 'an acceptance rule is never widened; a corrected check gets a new id and an entry in CHECK_CHANGES.json '
                                 '(before, after, cause); the earlier result is kept',
                                 'statuses: PASS, FAIL, ATTENTION (no registered criterion failed, but a discrepancy or caveat the '
                                 'coordinator must see), NOT_RUN'],
                       'checks': []}
            ids = {c['id'] for c in doc['checks']}
            for c in new_checks:
                if c['id'] in ids:
                    raise ValueError(f'check id {c["id"]} already used (append-only)')
                if c.get('status') not in ('PASS', 'FAIL', 'ATTENTION', 'NOT_RUN'):
                    raise ValueError(f'bad status {c.get("status")}')
                if c.get('kind') not in ('NUMERICAL_REPLAY', 'PROCEDURAL_COMPLIANCE'):
                    raise ValueError(f'bad kind {c.get("kind")}')
                for key in ('what', 'inputs', 'expected', 'observed'):
                    c.setdefault(key, None)
                c.setdefault('utc', utc())
                c.setdefault('unit_id', os.environ.get('TSZ_UNIT_ID'))
                doc['checks'].append(json.loads(json.dumps(c, default=_json_default)))
                ids.add(c['id'])
            counts = {kind: {st: sum(1 for c in doc['checks'] if c['kind'] == kind and c['status'] == st)
                             for st in ('PASS', 'FAIL', 'ATTENTION', 'NOT_RUN')}
                      for kind in ('NUMERICAL_REPLAY', 'PROCEDURAL_COMPLIANCE')}
            doc['summary'] = {'n_checks': len(doc['checks']), 'by_kind_and_status': counts, 'updated_utc': utc()}
            tmp = CHECKS.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(doc, indent=1, default=_json_default) + '\n')
            os.replace(tmp, CHECKS)
            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copyfile(CHECKS, HISTORY / f'CHECKS__{stamp}.json')
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return doc['summary']


def add_check_change(entry):
    """Append one correction record {check_id_before, check_id_after, before, after, cause, ...} to CHECK_CHANGES.json."""
    RECORDS.mkdir(parents=True, exist_ok=True); HISTORY.mkdir(parents=True, exist_ok=True)
    doc = json.loads(CHECK_CHANGES.read_text()) if CHECK_CHANGES.is_file() else {
        'schema': 'TSZ1-verifier-check-changes-v1',
        'rules': 'every correction of a verifier check: the earlier check stays in CHECKS.json unchanged; before, after and cause are '
                 'recorded here for the coordinator; acceptance rules are never widened', 'changes': []}
    entry = dict(entry); entry.setdefault('utc', utc())
    entry['n'] = len(doc['changes']) + 1
    doc['changes'].append(json.loads(json.dumps(entry, default=_json_default)))
    tmp = CHECK_CHANGES.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(doc, indent=1, default=_json_default) + '\n')
    os.replace(tmp, CHECK_CHANGES)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    shutil.copyfile(CHECK_CHANGES, HISTORY / f'CHECK_CHANGES__{stamp}.json')
    return entry['n']


def write_record(name, obj):
    """Write-once verifier record under records/ (aggregates only; person-level arrays refused)."""
    p = RECORDS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_json_default)
    return p


def note(text, **kw):
    RECORDS.mkdir(parents=True, exist_ok=True)
    with open(NOTES, 'a') as f:
        f.write(json.dumps({'utc': utc(), 'note': text, **kw}, default=_json_default) + '\n')


def code_hashes(*paths):
    return {Path(p).name: file_sha(p) for p in paths}


# ================================================================ invented-data self-test (no real data)
def selftest():
    rng = np.random.default_rng(11)
    ok = {}
    unit = rng.integers(10 ** 6, 10 ** 6 + 400, 1200)
    f5 = folds(unit, 5, 'T')
    ok['folds_whole_households'] = all(len(set(f5[unit == u].tolist())) == 1 for u in np.unique(unit))
    per = np.bincount(np.array([f5[unit == u][0] for u in np.unique(unit)]), minlength=5)
    ok['folds_balanced_households'] = int(per.max() - per.min()) <= 1
    sm = subset_masks(unit)
    G = len(np.unique(unit))
    ok['subset_sizes'] = all(len(np.unique(unit[sm[f'S{f}_{o}']])) == n_households(G, f) for f in (50, 70) for o in ORDERINGS)
    ok['subset_nested'] = all(not (sm[f'S50_{o}'] & ~sm[f'S70_{o}']).any() for o in ORDERINGS)
    ok['n_households_ceil'] = (n_households(13337, 50), n_households(13337, 70), n_households(13337, 100)) == (6669, 9336, 13337)
    d = rng.normal(size=300)
    iid = math.sqrt(np.sum((d - d.mean()) ** 2) * 300 / 299) / 300
    ok['cluster_se_singletons'] = abs(cluster_se(d, np.arange(300)) - iid) < 1e-15
    ok['subgroup_se_all_equals_cluster'] = abs(subgroup_se(d, np.ones(300, bool), np.arange(300) // 3) - cluster_se(d, np.arange(300) // 3)) < 1e-15
    floor = {c: 0.001 for c in OCC_SIDE}
    conds = {c: {'value': 0.0, 'se': 0.0, 'cap': (GUARD_CAP if c.startswith('guard') else CAPS[c.split('|')[2]])} for c in OCC_SIDE}
    r = reserve(conds, floor, FLOOR_HH)
    ok['reserve_zero_se_uses_floor'] = all(x['se_source'] == 'floor' and x['se_used'] == 0.001 for x in r['conditions'].values())
    conds2 = dict(conds); conds2['guard|sex0'] = {'value': 0.003, 'se': 0.001, 'cap': 0.005}
    r2 = reserve(conds2, floor, FLOOR_HH)
    ok['reserve_inclusive_boundary'] = r2['conditions']['guard|sex0']['met'] is True and abs(r2['conditions']['guard|sex0']['ratio'] - 2.0) < 1e-12
    conds3 = dict(conds); conds3['guard|sex1'] = {'value': 0.0031, 'se': 0.001, 'cap': 0.005}
    ok['reserve_fail_just_below'] = reserve(conds3, floor, FLOOR_HH)['met'] is False
    try:
        reserve(conds, {**floor, 'guard|sex0': 0.0}, FLOOR_HH); ok['reserve_zero_floor_refused'] = False
    except ValueError:
        ok['reserve_zero_floor_refused'] = True
    s = summarize_orderings([0.003, 0.002, 0.004, 0.0035], [0.0005] * 4)
    ok['summary_positive'] = s['label'] == 'RESOLVED_POSITIVE'
    s2 = summarize_orderings([0.003, -0.001, 0.004, 0.0035], [0.0005] * 4)
    ok['summary_sign_reversal'] = s2['label'] == 'UNRESOLVED' and s2['flags'] == ['SIGN_REVERSAL'] and s2['sign_reversals'] == 1
    ok['summary_single'] = summarize_orderings([0.003], [0.001])['label'] == 'RESOLVED_POSITIVE' and \
        summarize_orderings([0.0019], [0.001])['label'] == 'UNRESOLVED'
    ok['across'] = (across_versions('RESOLVED_POSITIVE', 'RESOLVED_POSITIVE') == 'POSITIVE_BOTH_VERSIONS' and
                    across_versions('RESOLVED_POSITIVE', 'UNRESOLVED') == 'VERSION_DEPENDENT' and
                    across_versions('UNRESOLVED', 'UNRESOLVED') == 'NOT_RESOLVED' and
                    across_versions('RESOLVED_POSITIVE', 'RESOLVED_NEGATIVE') == 'VERSION_DEPENDENT')
    allow = {j: 0.002 for j in STATS5}
    est = {s_: {j: s_ / 1000 for j in STATS5} for s_ in STRENGTHS}
    se_ = {s_: {j: 0.0 for j in STATS5} for s_ in STRENGTHS}
    sel = select_strength(est, se_, allow)
    ok['select_monotone'] = sel['selected'] == 1.0 and sel['first_failing_strength'] is None
    est2 = {s_: {j: (0.003 if s_ == 0.2 else 0.0) for j in STATS5} for s_ in STRENGTHS}
    sel2 = select_strength(est2, se_, allow)
    ok['select_pass_after_fail'] = sel2['selected'] == 0.1 and sel2['flags'] == ['PASS_AFTER_FAIL'] and sel2['first_failing_strength'] == 0.2
    est3 = {s_: {j: (0.003 if s_ == 0.01 else 0.0) for j in STATS5} for s_ in STRENGTHS}
    ok['select_identity'] = select_strength(est3, se_, allow)['selected'] == 0.0
    cells = {f'S{v}R{r}k{k}': 0.01 for v in (0, 1) for r in (1, 2) for k in (0, 1)}
    ses = {c: 0.001 for c in cells}
    ok['stability_stable'] = stability(cells, ses)['label'] == 'STABLE_POSITIVE'
    cells2 = dict(cells); cells2['S1R2k1'] = -0.001
    ok['stability_sign_changes'] = stability(cells2, ses)['label'] == 'SIGN_CHANGES'
    ok['softmax_rows'] = bool(np.allclose(softmax(rng.normal(size=(5, 5))).sum(1), 1.0))
    names = ['AGEP', 'WKHP', 'COW=1', 'COW=2', 'COW=__OTHER__', 'SCHL=01', 'SCHL=16', 'SCHL=__OTHER__', 'MAR=1', 'MAR=__OTHER__',
             'POBP=012', 'POBP=__OTHER__']
    enc = {'feature_names': names, 'numeric': {'AGEP': {'mean': 40.0, 'sd': 10.0}, 'WKHP': {'mean': 40.0, 'sd': 8.0}}}
    raw = {'AGEP': ['30', '50'], 'WKHP': ['40', '20'], 'COW': ['2', '9'], 'SCHL': ['16', '02'], 'MAR': ['1', '3'], 'POBP': ['012', '999']}
    X, nm = encode_raw(raw, enc)
    Zc = compact_po(X, nm)
    ok['encode_compact'] = bool(np.allclose(Zc, [[-1.0, 0.0, 1.0, 16.0, 0.0, 12.0], [1.0, -2.5, 2.0, 3.0, 1.0, 999.0]]))
    ok['git_blob_empty'] = git_blob_sha(b'') == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    ok = {k: bool(v) for k, v in ok.items()}
    return all(ok.values()), ok


if __name__ == '__main__':
    good, detail = selftest()
    print(json.dumps({'selftest': 'PASS' if good else 'FAIL', 'checks': detail}))
    raise SystemExit(0 if good else 1)
