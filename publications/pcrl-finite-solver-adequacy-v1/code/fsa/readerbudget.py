"""Label-free reader-budget diagnostics of the PCRL finite-output solver-adequacy campaign (`pcrl_finite_solver_adequacy_v1`; derived from nrm.readerbudget; reader-
robustness investigator). EXPLORATORY DEVELOPMENT records on adaptively used MODEL_SELECT people; never confirmation.

Why: a fixed auxiliary sample under-reads fine partitions. In the SCM benchmark the shared-cell release kept 16-17 ATTACK_FIT people
per released pair vector against 53-64 for its ordinary control, so a LOOKUP-type reader was weaker against the finer release for
budget reasons alone. This module measures, for any release and WITHOUT any label, how densely the development grid's readers can
read it: per grid repetition (the frozen FSA1_devgrid_R1..R4 reader splits of MODEL_SELECT), per FIT fraction (the grid's learning-
curve subsets 0.25 / 0.5 / 1.0) and per view (income vector, occupation vector, linked pair = both vectors of one person):

- distinct released vectors (exact float64 bytes, the LOOKUP reader's key; also after rounding to 6 / 3 / 2 decimals, so near-
  aliases created by floating-point noise and coarse structure of continuous outputs are visible);
- reader-FIT occupancy: people per distinct vector (per vector: mean, min, max, quantiles; per person: the size of the person's own
  FIT vector group, share in singletons, share in groups of >= 15 people, the ~15-per-cell figure of the SCM records being a
  heuristic, not a gate);
- coverage of VALIDATION and SCORE people: share whose vector never occurs in FIT (LOOKUP returns the FIT prior for them), share
  whose vector has 1-4 FIT people, share with >= 15, quantiles of the FIT count, and the mean LOOKUP data weight n / (n + 2) (the
  frozen pseudo-count 2 of scm_bench.readers; 0 = prior only, 1 = fully data-driven);
- for the pair view, the joint structure: distinct income vectors x distinct occupation vectors (product bound) against the joint
  values actually realized;
- randomized releases (R, n, k stacks): every draw separately (one draw per person, as the grid reads them), the draw each grid
  repetition reads (fsa.devreaders.draw_for: R1 0, R2 1, R3 6, R4 7, Monte Carlo offsets +2 / +4 on R1 / R2 only; a refused
  combination is recorded as None) and a cross-draw summary.

Nothing here reads a label: releases come from fsa.devreaders.load_member (the grid's own loader: fsa.release.load for this
campaign's releases, the admitted frozen copies for references), households from fsa.common.features('MODEL_SELECT').

    python3 -m fsa.readerbudget selftest                         invented data only (no real data, no label)
    python3 -m fsa.readerbudget splits                           label-free: rebuild the R1-R4 masks, compare with READER_SPLITS.json
    python3 -m fsa.readerbudget run --id ID [--spec BANK.json] [--member NAME=SPEC ...] [--offsets 0 2 4]
        SPEC: F/C (a saved release of this campaign) | REF:<name> (ohn.devrun.reference_release) | W (the witness) |
              BANK:<name> (an fsa.banks reference or diagnostic member, e.g. BANK:TR100)
        writes results/<campaign>/readers/BUDGET_<ID>.json (write-once) and BUDGET_<ID>.csv
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C

SCHEMA = 'FSA1-readerbudget-v1'
ALL_REPS = ('R1', 'R2', 'R3', 'R4')
FRACTIONS = (0.25, 0.5, 1.0)
VIEWS = ('income', 'occupation', 'pair')
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
DENSE = 15                       # the SCM records' "about 15 people per cell" heuristic (diagnostic only, never a gate)
SPARSE_MAX = 4                   # "very sparse": 1-4 FIT people share the vector
LOOKUP_PSEUDO = 2.0              # scm_bench.readers.LOOKUP_PSEUDO (checked in the selftest)
ROUND_DECIMALS = (6, 3, 2)
MC_OFFSETS = (0, 2, 4)           # the registered instance (offset 0) and the two Monte Carlo replicate offsets of the plan
OUT = C.RESULTS / 'readers'
SPLITS_PATH = C.RESULTS / 'READER_SPLITS.json'
# asymmetry flags for a (candidate, control) budget contrast (declared before any candidate outcome; diagnostic only)
ASYM_RATIO = 2.0                 # person-weighted median FIT group size differs by more than this factor
ASYM_UNSEEN = 0.10               # share of SCORE people unseen in FIT differs by more than this


# ---------------------------------------------------------------- keys and occupancy (pure functions)
def view_matrix(view, inc, occ):
    if view == 'income':
        a = inc
    elif view == 'occupation':
        a = occ
    elif view == 'pair':
        a = np.column_stack((np.asarray(inc, dtype=np.float64), np.asarray(occ, dtype=np.float64)))
    else:
        raise ValueError(f'unknown view {view!r}')
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def key_ids(view, inc, occ, decimals=None):
    """Integer id per person of the released vector(s) of a view: equal ids iff the float64 bytes are equal (the LOOKUP reader's
    key: scm_bench.readers.lookup_keys), or after rounding to `decimals` (+0.0 maps -0.0 to 0.0)."""
    a = view_matrix(view, inc, occ)
    if decimals is not None:
        a = np.ascontiguousarray(np.round(a, int(decimals)) + 0.0)
    v = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, inv = np.unique(v, return_inverse=True)
    return np.asarray(inv, dtype=np.int64).ravel()


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {f'q{int(round(q * 100))}': None for q in QUANTILES}
    return {f'q{int(round(q * 100))}': float(np.quantile(x, q)) for q in QUANTILES}


def _coverage(c):
    """FIT counts c of the vectors of a set of people (0 = the vector never occurs in FIT)."""
    c = np.asarray(c, dtype=np.float64)
    if c.size == 0:
        return {'people': 0}
    return {'people': int(c.size), 'share_unseen_in_fit': float(np.mean(c == 0)),
            f'share_fit_count_1_to_{SPARSE_MAX}': float(np.mean((c >= 1) & (c <= SPARSE_MAX))),
            f'share_fit_count_ge_{DENSE}': float(np.mean(c >= DENSE)), 'fit_count': _q(c),
            'lookup_data_weight_mean': float(np.mean(c / (c + LOOKUP_PSEUDO)))}


def occupancy(ids, fit, val, score):
    """Reader-FIT occupancy of one view and VALIDATION / SCORE coverage (label-free)."""
    ids = np.asarray(ids, dtype=np.int64)
    fit, val, score = (np.asarray(m, dtype=bool) for m in (fit, val, score))
    K = int(ids.max()) + 1 if ids.size else 0
    cnt = np.bincount(ids[fit], minlength=K)
    occ = cnt[cnt > 0]
    own = cnt[ids[fit]]
    sc_ids = np.unique(ids[score])
    rec = {'fit': {'people': int(fit.sum()), 'distinct_vectors': int(occ.size)},
           'validation': {**_coverage(cnt[ids[val]]), 'distinct_vectors': int(np.unique(ids[val]).size)},
           'score': {**_coverage(cnt[ids[score]]), 'distinct_vectors': int(sc_ids.size),
                     'share_distinct_score_vectors_unseen_in_fit': float(np.mean(cnt[sc_ids] == 0)) if sc_ids.size else None}}
    if occ.size:
        rec['fit']['people_per_vector'] = {'mean': float(occ.mean()), 'min': int(occ.min()), 'max': int(occ.max()), **_q(occ)}
        rec['fit']['person_weighted_group_size'] = {**_q(own), 'share_singleton': float(np.mean(own == 1)),
                                                    f'share_ge_{DENSE}': float(np.mean(own >= DENSE))}
    return rec


def all_people(inc, occ, decimals=ROUND_DECIMALS):
    """Distinct vectors over every MODEL_SELECT person, exact and rounded; joint structure of the pair view."""
    out = {}
    for view in VIEWS:
        out[view] = {'exact': int(key_ids(view, inc, occ).max()) + 1}
        for d in decimals:
            out[view][f'round{d}'] = int(key_ids(view, inc, occ, d).max()) + 1
    ni, no = out['income']['exact'], out['occupation']['exact']
    out['pair']['product_bound'] = int(ni * no)
    out['pair']['share_of_product_realized'] = float(out['pair']['exact'] / (ni * no))
    out['n_people'] = int(len(np.asarray(inc)))
    return out


def deterministic_budget(inc, occ, unit, reps=ALL_REPS, fractions=FRACTIONS, decimals=ROUND_DECIMALS):
    from . import devreaders as DR
    out = {'all_people': all_people(inc, occ, decimals), 'reps': {}}
    ids = {v: key_ids(v, inc, occ) for v in VIEWS}
    for rep in reps:
        r = DR.roles_of(unit, rep)
        out['reps'][rep] = {}
        for f in fractions:
            fit = DR.lc_fit(unit, r['FIT'], f)
            out['reps'][rep][str(float(f))] = {v: occupancy(ids[v], fit, r['VALIDATION'], r['SCORE']) for v in VIEWS}
    return out


def _headline(occ_rec):
    """The few numbers a comparison reads (one view, one repetition, one fraction)."""
    f, s = occ_rec['fit'], occ_rec['score']
    ppv, pw = f.get('people_per_vector') or {}, f.get('person_weighted_group_size') or {}
    return {'distinct_fit': f['distinct_vectors'], 'median_people_per_vector': ppv.get('q50'),
            'person_weighted_median_group': pw.get('q50'), 'share_fit_singleton': pw.get('share_singleton'),
            'score_share_unseen': s.get('share_unseen_in_fit'), f'score_share_ge_{DENSE}': s.get(f'share_fit_count_ge_{DENSE}'),
            'score_lookup_data_weight': s.get('lookup_data_weight_mean')}


def randomized_budget(inc, occ, unit, rule=None, reps=ALL_REPS, decimals=ROUND_DECIMALS, offsets=MC_OFFSETS):
    """(R, n, k) stacks (either task may be deterministic (n, k)): every draw on its own (full FIT only), the draws the grid reads
    per repetition under the member's rule and the plan's Monte Carlo offsets, and a cross-draw summary."""
    from . import devreaders as DR
    R = max(a.shape[0] for a in (inc, occ) if np.asarray(a).ndim == 3)
    pick = lambda a, d: a[d] if np.asarray(a).ndim == 3 else a
    draws = {}
    for d in range(R):
        draws[str(d)] = deterministic_budget(pick(inc, d), pick(occ, d), unit, reps, (1.0,), decimals)
    rule = dict(rule or {'draw': 'rep'})
    reads = {}
    for rep in reps:
        reads[rep] = {}
        for o in offsets:
            try:       # the grid's instance map (nrm.devreaders.REP_INSTANCE; offsets on R1/R2 only since 7222b0105)
                d = DR.draw_for({**rule, 'draw_offset': int(rule.get('draw_offset', 0)) + int(o)}, rep)
            except ValueError:
                d = None
            reads[rep][f'offset+{o}'] = d if d is not None and d < R else None
    summ = {}
    for view in VIEWS:
        rows = [_headline(draws[str(d)]['reps'][rep]['1.0'][view]) for d in range(R) for rep in reps]
        summ[view] = {k: {'min': min(x[k] for x in rows if x[k] is not None) if any(x[k] is not None for x in rows) else None,
                          'max': max(x[k] for x in rows if x[k] is not None) if any(x[k] is not None for x in rows) else None}
                      for k in rows[0]}
        summ[view]['all_people_exact'] = {'min': min(draws[str(d)]['all_people'][view]['exact'] for d in range(R)),
                                          'max': max(draws[str(d)]['all_people'][view]['exact'] for d in range(R))}
    return {'draws_stored': R, 'draw_rule': rule, 'grid_reads': reads, 'per_draw': draws, 'cross_draw_summary': summ}


def release_budget(rel, unit, reps=ALL_REPS, fractions=FRACTIONS, decimals=ROUND_DECIMALS, offsets=MC_OFFSETS):
    inc, occ = np.asarray(rel['income']), np.asarray(rel['occupation'])
    if inc.shape[-2] != len(unit) or occ.shape[-2] != len(unit):
        raise ValueError(f'release rows {inc.shape[-2]}/{occ.shape[-2]} differ from the {len(unit)} households-unit rows')
    if inc.ndim == 3 or occ.ndim == 3:
        return {'randomized': True, **randomized_budget(inc, occ, unit, rel.get('_draw_rule'), reps, decimals, offsets)}
    return {'randomized': False, **deterministic_budget(inc, occ, unit, reps, fractions, decimals)}


def contrast(a, b, view='pair', rep='R1', frac='1.0', draw=None):
    """Budget contrast of two release budgets (e.g. candidate vs its SEX-blind control) on one view / repetition / fraction:
    READER_BUDGET_ASYMMETRIC if the person-weighted median FIT group size differs by more than ASYM_RATIO or the share of SCORE
    people unseen in FIT differs by more than ASYM_UNSEEN, else READER_BUDGET_COMPARABLE. Diagnostic only (never a gate)."""
    def get(x):
        if x['randomized']:
            d = str(draw if draw is not None else 0)
            return _headline(x['per_draw'][d]['reps'][rep]['1.0'][view])
        return _headline(x['reps'][rep][frac][view])
    ha, hb = get(a), get(b)
    ma, mb = ha['person_weighted_median_group'] or 0.0, hb['person_weighted_median_group'] or 0.0
    ratio = (max(ma, mb) / min(ma, mb)) if min(ma, mb) > 0 else float('inf')
    dun = abs(ha['score_share_unseen'] - hb['score_share_unseen'])
    lab = 'READER_BUDGET_ASYMMETRIC' if ratio > ASYM_RATIO or dun > ASYM_UNSEEN else 'READER_BUDGET_COMPARABLE'
    return {'label': lab, 'view': view, 'rep': rep, 'fraction': frac, 'a': ha, 'b': hb, 'group_size_ratio': ratio,
            'score_unseen_difference': dun, 'thresholds': {'ratio': ASYM_RATIO, 'unseen': ASYM_UNSEEN}}


# ---------------------------------------------------------------- members (the grid's own loader; label-free)
def member_of(spec):
    """'F/C' | 'REF:<name>' | 'W' | 'BANK:<name>' | a member dict -> member dict (nrm.devreaders format)."""
    if isinstance(spec, dict):
        return spec
    if spec == 'W':
        return {'type': 'witness'}
    if spec.startswith('REF:'):
        return {'type': 'reference', 'name': spec[4:]}
    if spec.startswith('BANK:'):
        from .banks import REFERENCE_MEMBERS, DIAGNOSTIC_MEMBERS
        return {**REFERENCE_MEMBERS, **DIAGNOSTIC_MEMBERS}[spec[5:]]
    if '/' in spec:
        from .banks import family_member
        return family_member(spec)
    raise ValueError(f'unknown member spec {spec!r}')


def load(member, v, n):
    from . import devreaders as DR
    rel, prov = DR.load_member(member, v, n=n)
    return rel, prov


def run(run_id, members, versions=C.VERSIONS, offsets=MC_OFFSETS, out_dir=OUT):
    t0 = time.process_time()
    out_dir = Path(out_dir)
    jp, cp = out_dir / f'BUDGET_{run_id}.json', out_dir / f'BUDGET_{run_id}.csv'
    if jp.exists() or cp.exists():
        raise PermissionError(f'{jp.name} exists (write-once; use a new id)')
    feats = C.features('MODEL_SELECT')                      # label-free: household ids only
    unit = feats['unit']
    rec = {'schema': SCHEMA, 'id': run_id, 'utc': C.now(), 'role': 'MODEL_SELECT', 'n_people': int(len(unit)),
           'households': int(len(np.unique(unit))), 'splits_check': splits_check(unit), 'members': {},
           'definitions': {'key': 'exact float64 bytes of the view vector(s) (scm_bench.readers.lookup_keys); roundN = rounded to N decimals',
                           'dense': DENSE, 'sparse_max': SPARSE_MAX, 'lookup_pseudo': LOOKUP_PSEUDO,
                           'fractions': list(FRACTIONS), 'mc_offsets': list(offsets),
                           'asymmetry_flags': {'ratio': ASYM_RATIO, 'unseen': ASYM_UNSEEN}},
           'note': 'label-free (no SEX, no task label); reader budget of the development grid, not a recovery result'}
    rows = []
    for name, spec in members.items():
        m = member_of(spec)
        rec['members'][name] = {'member': m, 'versions': {}}
        for v in versions:
            rel, prov = load(m, v, len(unit))
            b = release_budget(rel, unit, offsets=offsets)
            b['wire_sha256'] = prov.get('wire_sha256')
            rec['members'][name]['versions'][str(v)] = b
            rows += csv_rows(name, v, b)
    rec['cpu_s'] = round(time.process_time() - t0, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_x(jp, rec)
    with open(cp, 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['member'])
        w.writeheader(); w.writerows(rows)
    return rec


def csv_rows(name, v, b):
    rows = []
    def emit(draw, reps_rec, allp):
        for rep, fr in reps_rec.items():
            for frac, views in fr.items():
                for view, o in views.items():
                    rows.append({'member': name, 'version': v, 'draw': draw, 'rep': rep, 'fit_fraction': frac, 'view': view,
                                 'distinct_all_exact': allp[view]['exact'], 'distinct_all_round3': allp[view]['round3'],
                                 **{k: (round(x, 5) if isinstance(x, float) else x) for k, x in _headline(o).items()}})
    if b['randomized']:
        for d, x in b['per_draw'].items():
            emit(d, x['reps'], x['all_people'])
    else:
        emit('', b['reps'], b['all_people'])
    return rows


# ---------------------------------------------------------------- frozen-split check (label-free)
def _hh_sha(unit, m):
    hh = np.unique(unit[m])
    return hashlib.sha256(','.join(map(str, hh)).encode()).hexdigest(), int(m.sum()), int(len(hh))


def splits_check(unit):
    """The grid masks rebuilt here equal the frozen READER_SPLITS.json (people, households, household-set SHA-256)."""
    from . import devreaders as DR
    man = json.loads(SPLITS_PATH.read_text())
    bad = []
    for rep in ALL_REPS:
        r = DR.roles_of(unit, rep)
        for role in ('FIT', 'VALIDATION', 'SCORE'):
            sha, people, hh = _hh_sha(unit, r[role])
            want = man['repetitions'][rep][role]
            if (sha, people, hh) != (want['household_set_sha256'], want['people'], want['households']):
                bad.append(f'{rep}/{role}')
    return {'ok': not bad, 'mismatches': bad, 'manifest_sha256': C.sha(SPLITS_PATH)}


# ---------------------------------------------------------------- invented-data selftest (no real data, no label)
def selftest():
    from scm_bench.readers import lookup_keys, LOOKUP_PSEUDO as LP
    from . import devreaders as DR
    chk = {'lookup_pseudo_matches_locked_reader': LP == LOOKUP_PSEUDO}
    rng = np.random.default_rng(20261020)
    n = 3000
    unit = np.repeat(np.arange(n // 2), 2)
    # finite release: 4 income cells x 8 occupation cells; a continuous release; a noisy copy of the finite release
    ic = rng.integers(0, 4, n); oc = rng.integers(0, 8, n)
    inc_lv = rng.dirichlet(np.ones(2), 4); occ_lv = rng.dirichlet(np.ones(5), 8)
    inc, occ = inc_lv[ic], occ_lv[oc]
    ids = {v: key_ids(v, inc, occ) for v in VIEWS}
    # ids partition equals the LOOKUP key partition
    def same_partition(a, keys):
        first = {}
        for i, k in zip(a, keys):
            if first.setdefault(i, k) != k:
                return False
        return len(set(a.tolist())) == len(set(keys))
    chk['ids_equal_lookup_partition'] = all(same_partition(ids[v], lookup_keys(*((inc,) if v == 'income' else (occ,) if v == 'occupation'
                                                                                  else (inc, occ)))) for v in VIEWS)
    chk['distinct_counts'] = [int(ids[v].max()) + 1 for v in VIEWS] == [4, 8, len(set(zip(ic.tolist(), oc.tolist())))]
    # occupancy against a brute-force count
    r = DR.roles_of(unit, 'R1')
    o = occupancy(ids['pair'], r['FIT'], r['VALIDATION'], r['SCORE'])
    fit_keys = [k for k, m in zip(ids['pair'], r['FIT']) if m]
    counts = {k: fit_keys.count(k) for k in set(fit_keys)}
    sc_counts = np.array([counts.get(k, 0) for k, m in zip(ids['pair'], r['SCORE']) if m])
    chk['fit_distinct_bruteforce'] = o['fit']['distinct_vectors'] == len(counts)
    chk['median_ppv_bruteforce'] = abs(o['fit']['people_per_vector']['q50'] - float(np.median(list(counts.values())))) < 1e-12
    chk['score_unseen_bruteforce'] = abs(o['score']['share_unseen_in_fit'] - float(np.mean(sc_counts == 0))) < 1e-12
    chk['lookup_weight_bruteforce'] = abs(o['score']['lookup_data_weight_mean'] - float(np.mean(sc_counts / (sc_counts + 2.0)))) < 1e-12
    pw = [counts[k] for k in fit_keys]
    chk['person_weighted_median_bruteforce'] = abs(o['fit']['person_weighted_group_size']['q50'] - float(np.median(pw))) < 1e-12
    # fractions shrink FIT; continuous release: every vector unique, everything unseen
    b = deterministic_budget(inc, occ, unit, ('R1',), FRACTIONS)
    fp = [b['reps']['R1'][str(f)]['pair']['fit']['people'] for f in FRACTIONS]
    chk['fractions_nested_sizes'] = fp[0] < fp[1] < fp[2] == int(r['FIT'].sum())
    cont_inc = rng.dirichlet(np.ones(2), n); cont_occ = rng.dirichlet(np.ones(5), n)
    bc = deterministic_budget(cont_inc, cont_occ, unit, ('R1',), (1.0,))
    sc = bc['reps']['R1']['1.0']['pair']['score']
    chk['continuous_all_unseen'] = sc['share_unseen_in_fit'] == 1.0 and sc['lookup_data_weight_mean'] == 0.0
    # (unit RR_BUDGET_SELFTEST_V1 tested the 7-coordinate pair view here, which stays unique at 2 decimals: a wrong expectation)
    chk['continuous_rounding_collapses'] = bc['all_people']['income']['round2'] < bc['all_people']['income']['exact'] == n
    # float noise below 1e-12 splits exact keys but not rounded keys
    noisy = inc + np.where(rng.random((n, 1)) < 0.5, 1e-13, 0.0) * np.array([[1.0, -1.0]])
    chk['float_noise_detected'] = (all_people(noisy, occ)['income']['exact'] > 4 and all_people(noisy, occ)['income']['round6'] == 4)
    chk['pair_product_bound'] = all_people(inc, occ)['pair']['product_bound'] == 32
    # randomized: per draw, grid reads (instance = repetition index + offset)
    st_inc = np.stack([inc] * 8); st_occ = np.stack([occ_lv[(oc + d) % 8] for d in range(8)])
    rb = release_budget({'income': st_inc, 'occupation': st_occ, '_draw_rule': {'draw': 'rep', 'draw_offset': 0}}, unit)
    chk['randomized_per_draw'] = rb['randomized'] and rb['draws_stored'] == 8 and len(rb['per_draw']) == 8
    # instance map since nrm.devreaders 7222b0105: R1 0, R2 1, R3 6, R4 7; Monte Carlo offsets on R1/R2 only
    chk['randomized_grid_reads'] = (rb['grid_reads']['R1'] == {'offset+0': 0, 'offset+2': 2, 'offset+4': 4}
                                    and rb['grid_reads']['R2'] == {'offset+0': 1, 'offset+2': 3, 'offset+4': 5}
                                    and rb['grid_reads']['R3'] == {'offset+0': 6, 'offset+2': None, 'offset+4': None}
                                    and rb['grid_reads']['R4'] == {'offset+0': 7, 'offset+2': None, 'offset+4': None})
    rb2 = release_budget({'income': st_inc, 'occupation': st_occ, '_draw_rule': {'draw': 'rep', 'draw_offset': 4}}, unit)
    chk['randomized_offset_member_reads'] = (rb2['grid_reads']['R1'] == {'offset+0': 4, 'offset+2': 6, 'offset+4': None}
                                             and rb2['grid_reads']['R4'] == {'offset+0': None, 'offset+2': None, 'offset+4': None})
    # contrast flag: a 4x8 release vs a 2x2 coarsening of it
    coarse = deterministic_budget(inc_lv[ic % 2], occ_lv[oc % 2], unit, ('R1',), (1.0,))
    fine = deterministic_budget(inc, occ, unit, ('R1',), (1.0,))
    chk['contrast_flags_asymmetry'] = contrast({'randomized': False, **fine}, {'randomized': False, **coarse})['label'] == 'READER_BUDGET_ASYMMETRIC'
    chk['contrast_self_comparable'] = contrast({'randomized': False, **fine}, {'randomized': False, **fine})['label'] == 'READER_BUDGET_COMPARABLE'
    rows = csv_rows('X', 0, {'randomized': False, **fine})
    chk['csv_rows'] = len(rows) == 3 and rows[2]['view'] == 'pair'
    chk['all_ok'] = all(v for v in chk.values() if isinstance(v, bool))
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('splits')
    r = sub.add_parser('run'); r.add_argument('--id', required=True); r.add_argument('--spec', default=None)
    r.add_argument('--member', action='append', default=[]); r.add_argument('--offsets', nargs='*', type=int, default=list(MC_OFFSETS))
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        chk = selftest(); print(json.dumps(chk, indent=1)); return 0 if chk['all_ok'] else 1
    if a.cmd == 'splits':
        res = splits_check(C.features('MODEL_SELECT')['unit']); print(json.dumps(res)); return 0 if res['ok'] else 1
    members = {}
    if a.spec:
        members.update(json.loads(Path(a.spec).read_text())['members'])
    for m in a.member:
        k, s = m.split('=', 1)
        members[k] = s
    if not members:
        raise SystemExit('no members')
    rec = run(a.id, members, offsets=tuple(a.offsets))
    print(json.dumps({'id': rec['id'], 'members': len(rec['members']), 'splits_ok': rec['splits_check']['ok'], 'cpu_s': rec['cpu_s']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
