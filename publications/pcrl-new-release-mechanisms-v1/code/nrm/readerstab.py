"""Reader-stability and robustness summaries over development-grid outputs of the PCRL new-release-mechanisms campaign (reader-
robustness investigator). Label-free: it reads only saved grid outputs (runs/GRID/<bank>/S<v>/GRID.json, BOOT.npz, BOOT_EXTRAS.npz),
the committed robustness plan and, optionally, a label-free reader-budget record (nrm.readerbudget). It applies the statistical
reviewer's frozen rules (nrm.rules, NRM1-rules-v1) unchanged. Every output is DESCRIPTIVE development evidence on adaptively used
MODEL_SELECT records: robustness cells never change a decision or a ranking (RULES.md sections 5.1 and 6), and lower recovery for the
declared readers is never protection against every reader.

What it does:
- merges several grid runs (the registered bank and the robustness banks; both versions) by cell; a cell present in two runs must
  agree bitwise (winner, score AUC, bootstrap draws, extras), otherwise CELL_CONFLICT is listed and the first (registered) run's
  value is kept;
- recomputes every paired cell from the saved winners' SCORE AUCs and bootstrap draws exactly as nrm.devreaders.paired does
  (pair: comparator - candidate; single views: candidate - comparator; SE = SD of the paired bootstrap differences, ddof 1), for
  ANY ordered pair of members, and cross-checks the GRID.json 'paired' block wherever it exists (PAIRED_MISMATCH otherwise);
- Monte Carlo: on cells where both sides' replicate members exist (registered cells of randomized members), nrm.rules.mc_cell over
  the instances, offset paired with offset (as nrm.claims.grid_entry); elsewhere the single instance (D = 1);
- per comparison: nrm.rules.comparison (registered), nrm.rules.robustness, benefit / recipient screens on the robustness cells, all
  cells and each robustness part alone (A: R3/R4 x k0/k1; B: R1/R2 x k2/k3), aggregates with sign reversals, n_eff, k_differs and
  the reader-seed qualifier, nrm.rules.reader_stability (registered and all cells; exact aliases counted once from wire hashes);
- the extras (nrm.readerextra): the same screens for the NOLOOKUP and ENHANCED (TABLE_CV) winners and the declared reader-targeting
  labels (ROBUST_PLAN.json): TABLE_READER_TARGETING (HOLDS_WITHOUT_TABLE_READERS / NEEDS_TABLE_READERS / NO_MAIN_BENEFIT) and
  TABLE_CV_SENSITIVITY (SURVIVES_TABLE_CV / REMOVED_BY_TABLE_CV / NO_MAIN_BENEFIT);
- grid validity: nrm.rules.grid_validity from the score fingerprints (k0 vs k1, every repetition present) with label-free distinct-
  input counts; k2 vs k3 equalities are counted separately (descriptive);
- winner composition per member (table-reader winners, seed-dependent winners, near ties) and the reader-budget contrast flags.

    python3 -m nrm.readerstab selftest                                     invented grid outputs only (no real data, no label)
    python3 -m nrm.readerstab run --plan results/<campaign>/readers/ROBUST_PLAN.json [--budget readers/BUDGET_<id>.json]
                                  --out results/<campaign>/readers/ROBUSTNESS.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import rules as R

SCHEMA = 'NRM1-readerstab-v1'
GRID_ROOT = C.RUNS / 'GRID'
KINDS = {'NOLOOKUP': 'nolookup', 'TABLE_CV': 'table_cv', 'ENHANCED': 'enhanced'}
VIEWS = R.VIEWS
TABLE_WORDS = ('LOOKUP', 'TABLE_CV')
PART_A = lambda c: (R.parse_cell(c) or (None, None, None))[1] in ('R3', 'R4') and (R.parse_cell(c) or (0, 0, -1))[2] in (0, 1)
PART_B = lambda c: (R.parse_cell(c) or (None, None, None))[1] in ('R1', 'R2') and (R.parse_cell(c) or (0, 0, -1))[2] in (2, 3)


# ---------------------------------------------------------------- loading and merging
def _npz(p):
    if not Path(p).is_file():
        return {}
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _same(a, b):
    """Bitwise agreement of two cell records, ignoring CPU-time fields (e.g. the extras' 'extras_cpu_s', which differ between runs
    even for identical cache re-reads)."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {k for k in a if not str(k).endswith('cpu_s')}
        kb = {k for k in b if not str(k).endswith('cpu_s')}
        return ka == kb and all(_same(a[k], b[k]) for k in ka)
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (math.isnan(a) and math.isnan(b))
    return a == b


def merge(banks, versions=C.VERSIONS, root=GRID_ROOT, dirs=None):
    """Merge grid runs. Returns {'cells', 'boot', 'bootx', 'paired', 'members', 'sources', 'conflicts', 'runs'}. `dirs` maps
    (bank, version) to an explicit run folder (a rerun written with --out, e.g. after a withdrawn unit left its default folder)."""
    M = {'cells': {}, 'boot': {}, 'bootx': {}, 'paired': {}, 'members': {v: {} for v in versions}, 'sources': {}, 'conflicts': [],
         'runs': []}
    for bank in banks:
        for v in versions:
            d = Path((dirs or {}).get((bank, v)) or (Path(root) / bank / f'S{v}'))
            g = json.loads((d / 'GRID.json').read_text())
            if int(g.get('seed')) != int(v):
                raise ValueError(f'{d}: GRID.json seed {g.get("seed")} differs from the folder')
            boot, bootx = _npz(d / 'BOOT.npz'), _npz(d / 'BOOT_EXTRAS.npz')
            M['runs'].append({'bank': bank, 'version': v, 'cells': len(g['cells']), 'unit_id': g.get('unit_id'), 'utc': g.get('utc'),
                              'grid_sha256': C.sha(d / 'GRID.json'), 'extras': bool(bootx),
                              'folder': str(d.relative_to(C.ROOT)) if d.is_relative_to(C.ROOT) else str(d)})
            for name, pv in (g.get('members') or {}).items():
                M['members'][v].setdefault(name, pv)
            for cell, cres in g['cells'].items():
                M['sources'].setdefault(cell, []).append(bank)
                for name, ent in cres.items():
                    for view, e in ent.items():
                        have = M['cells'].setdefault(cell, {}).setdefault(name, {}).get(view)
                        keys = [f'{cell}__{name}__{view}'] + [f'{cell}__{name}__{view}__{k}' for k in KINDS]
                        if have is None:
                            M['cells'][cell][name][view] = e
                            if keys[0] in boot:
                                M['boot'][keys[0]] = boot[keys[0]]
                            for k in keys[1:]:
                                if k in bootx:
                                    M['bootx'][k] = bootx[k]
                            continue
                        bad = []
                        for f in ('winner', 'score_auc', 'val_auc', 'near_tie'):
                            if not _same(have.get(f), e.get(f)):
                                bad.append(f)
                        if not _same(have.get('extras'), e.get('extras')):
                            bad.append('extras')
                        if keys[0] in boot and keys[0] in M['boot'] and not np.array_equal(boot[keys[0]], M['boot'][keys[0]]):
                            bad.append('boot')
                        for k in keys[1:]:
                            if k in bootx and k in M['bootx'] and not np.array_equal(bootx[k], M['bootx'][k]):
                                bad.append(k.rsplit('__', 1)[1])
                        if bad:
                            M['conflicts'].append({'label': 'CELL_CONFLICT', 'cell': cell, 'member': name, 'view': view, 'bank': bank,
                                                   'fields': bad})
            for cand, comps in (g.get('paired') or {}).items():
                for comp, views in comps.items():
                    for view, e in views.items():
                        d2 = M['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                        for cell, x in e['cells'].items():
                            if cell in d2['cells'] and (d2['cells'][cell] != x or d2['cell_se'][cell] != e['cell_se'][cell]):
                                M['conflicts'].append({'label': 'CELL_CONFLICT', 'cell': cell, 'member': f'{cand}|{comp}', 'view': view,
                                                       'bank': bank, 'fields': ['paired']})
                                continue
                            d2['cells'].setdefault(cell, x); d2['cell_se'].setdefault(cell, e['cell_se'][cell])
    return M


# ---------------------------------------------------------------- paired cells from saved winners (as nrm.devreaders.paired)
def _side(M, cell, name, view, kind):
    e = ((M['cells'].get(cell) or {}).get(name) or {}).get(view)
    if e is None:
        return None, None
    if kind is None:
        return e.get('score_auc'), M['boot'].get(f'{cell}__{name}__{view}')
    x = (e.get('extras') or {}).get(KINDS[kind]) or {}
    return x.get('score_auc'), M['bootx'].get(f'{cell}__{name}__{view}__{kind}')


def paired_cell(M, cell, cand, comp, view, kind=None):
    """(d, se) of one cell or None: pair d = comp - cand, single views d = cand - comp; se = SD of paired bootstrap differences."""
    a, ba = _side(M, cell, cand, view, kind)
    b, bb = _side(M, cell, comp, view, kind)
    if a is None or b is None or ba is None or bb is None:
        return None
    if view == 'pair':
        d, db = b - a, bb - ba
    else:
        d, db = a - b, ba - bb
    return float(d), float(np.std(db, ddof=1))


def entry(M, cand, comp, reps=None, kind=None, cell_filter=None):
    """{view: {'cells', 'cell_se'[, 'cell_se_mc'], 'mc_draws'}} for cand vs comp over every merged cell (learning-curve '@' cells
    excluded). Monte Carlo where every replicate pairing of the cell exists (offset paired with offset, nrm.claims.grid_entry)."""
    reps = reps or {}
    rc, rp = reps.get(cand, []), reps.get(comp, [])
    out = {}
    for view in VIEWS:
        cells, ses, mcs, draws = {}, {}, {}, {}
        for cell in sorted(M['cells']):
            if R.parse_cell(cell) is None or (cell_filter and not cell_filter(cell)):
                continue
            base = paired_cell(M, cell, cand, comp, view, kind)
            if base is None:
                continue
            vals, ss = [base[0]], [base[1]]
            if rc or rp:
                ok = True
                for i, a in enumerate(rc or [None] * len(rp)):
                    c2 = a or cand
                    k2 = rp[i] if i < len(rp) else comp
                    x = paired_cell(M, cell, c2, k2, view, kind)
                    if x is None:
                        ok = False
                        break
                    vals.append(x[0]); ss.append(x[1])
                if not ok:
                    vals, ss = [base[0]], [base[1]]
            if len(vals) > 1:
                m = R.mc_cell(vals, ss)
                cells[cell], ses[cell], mcs[cell] = m['value'], m['se'], m['se_mc']
            else:
                cells[cell], ses[cell] = vals[0], ss[0]
            draws[cell] = len(vals)
        out[view] = {'cells': cells, 'cell_se': ses, 'cell_se_mc': mcs, 'mc_draws': draws}
    return out


def crosscheck_paired(M):
    """Recomputed registered-winner cells against GRID.json 'paired' (exact equality expected)."""
    n, bad = 0, []
    for cand, comps in M['paired'].items():
        for comp, views in comps.items():
            for view, e in views.items():
                for cell, x in e['cells'].items():
                    y = paired_cell(M, cell, cand, comp, view)
                    n += 1
                    if y is None or y[0] != x or y[1] != e['cell_se'][cell]:
                        bad.append({'cand': cand, 'comp': comp, 'view': view, 'cell': cell, 'grid': [x, e['cell_se'][cell]],
                                    'recomputed': list(y) if y else None})
    return {'checked': n, 'mismatches': bad[:50], 'n_mismatches': len(bad), 'label': 'PAIRED_MATCH' if not bad else 'PAIRED_MISMATCH'}


# ---------------------------------------------------------------- wires, validity, winners
def _hashable(w):
    return tuple(w) if isinstance(w, list) else w


def wires_of(M, names):
    out = {}
    for name in names:
        per = {}
        for v, provs in M['members'].items():
            ws = (provs.get(name) or {}).get('wire_sha256')
            if ws:
                per[v] = {t: _hashable(ws[t]) for t in C.TASKS}
        if len(per) == len(M['members']):
            out[name] = per
    return out


def validity(M, minimum_set, manifest, distinct=None, members=None):
    """nrm.rules.grid_validity from score fingerprints: k0 vs k1 per recipe, view, member, version and repetition (every
    repetition present); k2 vs k3 equalities counted separately (descriptive). `members` restricts the checks to the robustness
    bank (the decision grid's own validity is the coordinator's nrm.claims record)."""
    seeds = {f'S{v}k{k}': manifest['reader_seeds'][f'V{v}k{k}'] for v in C.VERSIONS for k in range(4)}
    checks, k23 = [], {'pairs': 0, 'equal_seed_dependent': 0, 'equal_by_recipe': {}}
    for cell, cres in M['cells'].items():
        p = R.parse_cell(cell)
        if p is None or p[2] not in (0, 2):
            continue
        v, rep, k = p
        other = f'S{v}{rep}k{k + 1}'
        for name, ent in cres.items():
            if members is not None and name not in members:
                continue
            for view, e in ent.items():
                f0 = (e.get('extras') or {}).get('score_sha256') or {}
                f1 = ((((M['cells'].get(other) or {}).get(name) or {}).get(view) or {}).get('extras') or {}).get('score_sha256') or {}
                for recipe in R.SEED_DEPENDENT + R.SEED_INVARIANT:
                    if recipe not in f0 or recipe not in f1:
                        continue
                    eq = f0[recipe] == f1[recipe]
                    if k == 0:
                        checks.append({'recipe': recipe, 'view': view, 'wire': name, 'v': v, 'R': rep, 'equal_k0_k1': eq})
                    elif recipe in R.SEED_DEPENDENT:
                        k23['pairs'] += 1
                        if eq:
                            k23['equal_seed_dependent'] += 1
                            k23['equal_by_recipe'][recipe] = k23['equal_by_recipe'].get(recipe, 0) + 1
    gv = R.grid_validity(seeds, checks, minimum_set, distinct or {})
    nondet = sorted({(c['recipe'], c['view']) for c in checks if c['recipe'] in R.SEED_INVARIANT and c['equal_k0_k1'] is False})
    return {**gv, 'n_checks': len(checks), 'reps_checked': sorted({c['R'] for c in checks}),
            'seed_dependent_equalities': sum(1 for c in checks if c['recipe'] in R.SEED_DEPENDENT and c['equal_k0_k1']),
            'nondeterministic_seed_invariant': [list(x) for x in nondet], 'k2_vs_k3': k23}


def winners(M, names):
    """Per member and view: winner counts, table-reader winners, seed-dependent winners, near ties (registered winner and the
    extras winners), over every merged cell."""
    out = {}
    for name in names:
        rec = {}
        for view in VIEWS:
            ws, nt, nol, enh, tab = [], 0, [], [], []
            for cell in sorted(M['cells']):
                if R.parse_cell(cell) is None:
                    continue
                e = ((M['cells'][cell].get(name) or {}).get(view))
                if not e:
                    continue
                ws.append(e['winner']); nt += bool(e.get('near_tie'))
                x = e.get('extras') or {}
                if 'nolookup' in x:
                    nol.append(x['nolookup']['winner'])
                if 'enhanced' in x:
                    enh.append(x['enhanced']['winner'])
                if 'table_cv' in x:
                    tab.append({'a': x['table_cv'].get('a'), 'C': x['table_cv'].get('C')})
            cnt = lambda xs: {w: xs.count(w) for w in sorted(set(xs))}
            rec[view] = {'cells': len(ws), 'winners': cnt(ws), 'near_ties': nt,
                         'table_reader_winners': sum(any(t in w for t in TABLE_WORDS) for w in ws),
                         'seed_dependent_winners': sum(w.split('__')[-1] in R.SEED_DEPENDENT for w in ws),
                         'nolookup_winners': cnt(nol), 'enhanced_winners': cnt(enh),
                         'enhanced_table_cv_winners': sum('TABLE_CV' in w for w in enh),
                         'table_cv_smoothing': cnt([json.dumps(t, sort_keys=True) for t in tab])}
        out[name] = rec
    return out


# ---------------------------------------------------------------- one comparison
def _filter_entry(e, pred):
    return {view: {k: ({c: x for c, x in d.items() if pred(c)} if isinstance(d, dict) else d) for k, d in ent.items()}
            for view, ent in e.items()}


def _screens(e, cells):
    return {'pair': R.benefit_screen(e['pair'], cells=cells),
            'income': R.recipient_screen(e['income'], cells=cells),
            'occupation': R.recipient_screen(e['occupation'], cells=cells)}


def _target_label(main, other, yes, no):
    if main != 'AGG_POSITIVE':
        return 'NO_MAIN_BENEFIT'
    return yes if other == 'AGG_POSITIVE' else no


def _slim(screen):
    """A screen record without the per-cell value dump (kept: label, aggregate summary, flags, old rule)."""
    out = {k: v for k, v in screen.items() if k != 'aggregate'}
    a = screen.get('aggregate') or {}
    out['aggregate'] = {k: a.get(k) for k in ('complete', 'error', 'n_cells', 'mean', 'se_agg', 'se_household', 'se_reader_split',
                                               'se_mc', 'floor_binds', 'lower', 'upper', 'block_means', 'per_version',
                                               'n_eff_distinct_cells', 'k_differs', 'reader_seed_qualifier', 'sign_reversals',
                                               'alias') if k in a}
    return out


def compare(M, cand, comp, reps=None, wires=None, full=True):
    e = entry(M, cand, comp, reps)
    rec = {'cand': cand, 'comp': comp, 'cells': {v: e[v]['cells'] for v in VIEWS}, 'cell_se': {v: e[v]['cell_se'] for v in VIEWS},
           'cell_se_mc': {v: e[v]['cell_se_mc'] for v in VIEWS}, 'mc_draws': e['pair']['mc_draws']}
    reg = R.comparison(e, cells='registered')
    rec['registered'] = {'benefit': _slim(reg['benefit']), 'recipient': {v: _slim(reg['recipient'][v]) for v in R.SINGLE_VIEWS},
                         'benefit_positive': reg['benefit_positive'], 'recipient_gate_pass': reg['recipient_gate_pass'],
                         'flags': reg['flags']}
    rob = R.robustness(e)
    rec['robustness'] = {'label': rob['label']}
    for cs in ('robustness', 'all'):
        rec[f'screens_{cs}'] = {v: _slim(s) for v, s in _screens(e, cs).items()}
    # single-instance variant: every cell read from the base members' own instance (no Monte Carlo averaging), so the cells of a
    # block differ by the reader seed only; its k_differs / qualifier / sign reversals are the reader-seed reading (the 'all' set
    # above mixes Monte Carlo-averaged registered cells with single-instance k2/k3 cells inside the R1/R2 blocks)
    e1 = entry(M, cand, comp, None)
    rec['screens_all_single_instance'] = {v: _slim(s) for v, s in _screens(e1, 'all').items()}
    for part, pred in (('A_R3R4_k0k1', PART_A), ('B_R1R2_k2k3', PART_B)):
        fe = _filter_entry(e, pred)
        if fe['pair']['cells']:
            rec[f'screens_part_{part}'] = {v: _slim(s) for v, s in _screens(fe, 'robustness').items()}
    # extras: NOLOOKUP and ENHANCED (TABLE_CV) winners; TABLE_CV alone descriptively
    rec['extras'] = {}
    for kind in KINDS:
        ek = entry(M, cand, comp, reps, kind=kind)
        if not ek['pair']['cells']:
            continue
        rec['extras'][kind] = {cs: {v: _slim(s) for v, s in _screens(ek, cs).items()} for cs in ('registered', 'all')}
    for cs in ('registered', 'all'):
        main = (rec['registered'] if cs == 'registered' else rec['screens_all'])['benefit' if cs == 'registered' else 'pair']['label']
        nol = (((rec['extras'].get('NOLOOKUP') or {}).get(cs) or {}).get('pair') or {}).get('label')
        enh = (((rec['extras'].get('ENHANCED') or {}).get(cs) or {}).get('pair') or {}).get('label')
        rec[f'targeting_{cs}'] = {
            'TABLE_READER_TARGETING': _target_label(main, nol, 'HOLDS_WITHOUT_TABLE_READERS', 'NEEDS_TABLE_READERS'),
            'TABLE_CV_SENSITIVITY': _target_label(main, enh, 'SURVIVES_TABLE_CV', 'REMOVED_BY_TABLE_CV'),
            'main': main, 'nolookup': nol, 'enhanced': enh}
    if full:
        cells_reg = {c: x for c, x in M['cells'].items() if c in R.REGISTERED}
        cells_all = {c: x for c, x in M['cells'].items() if R.parse_cell(c) is not None}
        rec['stability'] = {'registered': R.reader_stability(cells_reg, cand, comp, wires=wires),
                            'all': R.reader_stability(cells_all, cand, comp, wires=wires)}
        for k in ('registered', 'all'):
            for view, s in rec['stability'][k].items():
                s.pop('cells', None)
                s.pop('winners', None)
    return rec


def headline(rec):
    """One row of the report table."""
    def lab(x, *ks):
        for k in ks:
            x = (x or {}).get(k)
        return x
    return {'cand': rec['cand'], 'comp': rec['comp'],
            'registered_benefit': lab(rec, 'registered', 'benefit', 'label'),
            'registered_mean': lab(rec, 'registered', 'benefit', 'aggregate', 'mean'),
            'robustness': rec['robustness']['label'],
            'robustness_cells_mean': lab(rec, 'screens_robustness', 'pair', 'aggregate', 'mean'),
            'robustness_cells_label': lab(rec, 'screens_robustness', 'pair', 'label'),
            'all_cells_mean': lab(rec, 'screens_all', 'pair', 'aggregate', 'mean'),
            'all_cells_label': lab(rec, 'screens_all', 'pair', 'label'),
            'part_A_mean': lab(rec, 'screens_part_A_R3R4_k0k1', 'pair', 'aggregate', 'mean'),
            'part_B_mean': lab(rec, 'screens_part_B_R1R2_k2k3', 'pair', 'aggregate', 'mean'),
            'sign_reversals_all': lab(rec, 'screens_all', 'pair', 'aggregate', 'sign_reversals'),
            'all_single_instance_mean': lab(rec, 'screens_all_single_instance', 'pair', 'aggregate', 'mean'),
            'all_single_instance_label': lab(rec, 'screens_all_single_instance', 'pair', 'label'),
            'sign_reversals_single_instance': lab(rec, 'screens_all_single_instance', 'pair', 'aggregate', 'sign_reversals'),
            'k_differs_single_instance': lab(rec, 'screens_all_single_instance', 'pair', 'aggregate', 'k_differs'),
            'qualifier_single_instance': lab(rec, 'screens_all_single_instance', 'pair', 'aggregate', 'reader_seed_qualifier'),
            'recipient_income_all': lab(rec, 'screens_all', 'income', 'label'),
            'recipient_occupation_all': lab(rec, 'screens_all', 'occupation', 'label'),
            'nolookup_all_mean': lab(rec, 'extras', 'NOLOOKUP', 'all', 'pair', 'aggregate', 'mean'),
            'enhanced_all_mean': lab(rec, 'extras', 'ENHANCED', 'all', 'pair', 'aggregate', 'mean'),
            'targeting_registered': rec['targeting_registered'], 'targeting_all': rec['targeting_all']}


# ---------------------------------------------------------------- run
def run(plan_path, out_path, budget_path=None, root=GRID_ROOT, dirs=None):
    t0 = time.process_time()
    plan_path, out_path = Path(plan_path), Path(out_path)
    if out_path.exists():
        raise PermissionError(f'{out_path.name} exists (write-once)')
    plan = json.loads(plan_path.read_text())
    banks = [plan['banks']['registered']] + list(plan['banks']['robustness'])
    M = merge(banks, root=root, dirs=dirs)
    names = list(plan['members'])
    reps = plan.get('replicates_registered') or {}
    wires = wires_of(M, names)
    manifest = json.loads((C.RESULTS / 'READER_SPLITS.json').read_text())
    distinct, budget = {}, None
    if budget_path:
        budget = json.loads(Path(budget_path).read_text())
        for name, m in budget['members'].items():
            for v, b in m['versions'].items():
                allp = b['per_draw']['0']['all_people'] if b['randomized'] else b['all_people']
                distinct[f'{name}|V{v}'] = {view: int(allp[view]['exact']) for view in VIEWS}
    minimum = list(plan.get('minimum_set') or [n for n in names if n.startswith('FB2_')])   # ROBUST_PLAN_V1: the five FB2 members
    rec = {'schema': SCHEMA, 'utc': C.now(), 'plan': {'path': str(plan_path.relative_to(C.ROOT)) if plan_path.is_relative_to(C.ROOT)
                                                      else str(plan_path), 'sha256': C.sha(plan_path), 'plan_id': plan.get('plan_id')},
           'banks': banks, 'runs': M['runs'], 'conflicts': M['conflicts'], 'paired_crosscheck': crosscheck_paired(M),
           'cells_merged': sorted(c for c in M['cells'] if R.parse_cell(c) is not None),
           'grid_validity': validity(M, minimum, manifest, distinct, members=names), 'minimum_set': minimum,
           'grid_validity_scope': 'robustness-bank members only (plan members); the decision grid validity is nrm.claims',
           'winners': winners(M, names), 'comparisons': {}, 'focus': [], 'scope': list(R.SCOPE),
           'note': plan.get('status')}
    focus = [(f['cand'], f['comp']) for f in plan['focus']]
    for cand in names:
        for comp in names:
            if cand == comp:
                continue
            full = (cand, comp) in focus
            try:
                c = compare(M, cand, comp, reps, wires, full=full)
            except (KeyError, ValueError) as ex:
                rec['comparisons'][f'{cand}|{comp}'] = {'error': repr(ex)}
                continue
            if not full:
                c = {k: c[k] for k in ('cand', 'comp', 'registered', 'robustness', 'screens_all', 'targeting_registered', 'targeting_all')}
                c['registered'] = {'benefit': {'label': c['registered']['benefit']['label'],
                                               'mean': c['registered']['benefit']['aggregate'].get('mean')},
                                   'recipient_gate_pass': c['registered']['recipient_gate_pass']}
                c['screens_all'] = {v: {'label': s['label'], 'mean': s['aggregate'].get('mean')} for v, s in c['screens_all'].items()}
            rec['comparisons'][f'{cand}|{comp}'] = c
    for cand, comp in focus:
        c = rec['comparisons'].get(f'{cand}|{comp}') or {}
        if 'error' not in c and 'screens_robustness' in c:
            rec['focus'].append({**headline(c), 'role': next(f['role'] for f in plan['focus'] if (f['cand'], f['comp']) == (cand, comp))})
    if budget:
        from . import readerbudget as RB
        bc = {}
        for cand, comp in focus:
            for v in C.VERSIONS:
                try:
                    a = budget['members'][cand]['versions'][str(v)]; b = budget['members'][comp]['versions'][str(v)]
                    bc[f'{cand}|{comp}|V{v}'] = RB.contrast(a, b, view='pair', rep='R1', frac='1.0', draw=0)
                except KeyError:
                    continue
        rec['budget_contrasts'] = bc
    rec['cpu_s'] = round(time.process_time() - t0, 2)
    C.write_json_x(out_path, rec)
    return rec


# ---------------------------------------------------------------- invented-data selftest (no real data, no label)
def _seeded(*parts):
    """Deterministic generator from text parts (Python's hash() is salted per process)."""
    import hashlib
    return np.random.default_rng(int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _fake_bank(root, bank, v, cells, members, values, extras=True, paired_pairs=(), seeds_fp=None):
    """Write GRID.json / BOOT.npz / BOOT_EXTRAS.npz in the nrm.devreaders format. values(cell, member, view, kind) -> AUC.
    Bootstrap noise is a deterministic function of (cell, view) plus a member part, so a cell re-written by another bank is bitwise
    identical (as a cache re-read is in the real grid)."""
    from . import devreaders as DR
    res = {}
    for cell in cells:
        res[cell] = {}
        common = _seeded('common', cell).normal(0, 0.006, 400)
        for name in members:
            ent = {}
            for view in ('income', 'occupation', 'pair'):
                auc0 = values(cell, name, view, None)
                own = 'TWIN' if name == 'ALIAS' else name                  # an exact alias shares the readers, hence the draws
                boot = auc0 + common + _seeded('own', cell, own, view).normal(0, 0.002, 400)
                e = {'winner': 'HGB15' if view != 'pair' else 'IGNORE_income__ET5', 'origin': 'new', 'val_auc': auc0,
                     'gap_to_runner_up': 0.02, 'near_tie': False, 'score_auc': auc0, 'boot': boot}
                if extras:
                    x, xb = {}, {}
                    for kind, key in KINDS.items():
                        a = values(cell, name, view, kind)
                        x[key] = {'winner': 'X', 'val_auc': a, 'near_tie': False, 'score_auc': a}
                        xb[kind] = a + (boot - auc0)                 # same bootstrap noise (paired as in a real cell)
                    x['score_sha256'] = seeds_fp(cell, name, view) if seeds_fp else {}
                    x['extras_cpu_s'] = float(np.random.default_rng().random())   # differs per write, as real CPU times do
                    e['extras'], e['extras_boot'] = x, xb
                ent[view] = e
            res[cell][name] = ent
    out = Path(root) / bank / f'S{v}'
    out.mkdir(parents=True)
    pairs = {}
    for cand, comp in paired_pairs:
        pairs.setdefault(cand, {})[comp] = DR.paired(res, cand, comp)
    np.savez(out / 'BOOT.npz', **{f'{c}__{n}__{w}': e['boot'] for c, cr in res.items() for n, en in cr.items() for w, e in en.items()})
    bx = {f'{c}__{n}__{w}__{k}': b for c, cr in res.items() for n, en in cr.items() for w, e in en.items()
          for k, b in e.get('extras_boot', {}).items()}
    if bx:
        np.savez(out / 'BOOT_EXTRAS.npz', **bx)
    wire = lambda n: {'income': 'inc_shared' if n in ('TWIN', 'ALIAS') else f'inc_{n}', 'occupation': f'occ_{n}'}
    g = {'schema': 'NRM1-devgrid-v1', 'bank_id': bank, 'seed': v, 'cells': DR.strip(res), 'paired': pairs,
         'members': {n: {'wire_sha256': wire(n)} for n in members}, 'unit_id': 'SELFTEST', 'utc': C.now()}
    (out / 'GRID.json').write_text(json.dumps(g, default=float))
    return res


def selftest():
    import tempfile
    root = Path(tempfile.mkdtemp())
    reg = [f'S{{v}}{r}k{k}' for r in ('R1', 'R2') for k in (0, 1)]
    rob_a = [f'S{{v}}{r}k{k}' for r in ('R3', 'R4') for k in (0, 1)]
    rob_b = [f'S{{v}}{r}k{k}' for r in ('R1', 'R2') for k in (2, 3)]
    base_auc = {'TWIN': 0.640, 'REF': 0.625, 'ALIAS': 0.640, 'CAND': 0.610, 'CAND__d2': 0.612, 'CAND__d4': 0.608}

    def values(cell, name, view, kind):
        if name == 'ALIAS':
            return values(cell, 'TWIN', view, kind)                       # an exact alias of TWIN
        z = float(_seeded('v', cell, name, view, kind).normal(0, 0.002))
        a = base_auc[name] + (0.0 if view == 'pair' else -0.05) + z
        if name.startswith('CAND') and view != 'pair':
            a = base_auc['TWIN'] - 0.05 + z                               # single views: no worsening
        if kind == 'ENHANCED' and name.startswith('CAND'):
            a = base_auc['TWIN'] + (0.0 if view == 'pair' else -0.05) + z  # TABLE_CV removes the candidate's benefit
        return float(a)

    def fp(cell, name, view):
        k = R.parse_cell(cell)[2]
        return {r: (f'{name}{view}{cell[:4]}k{k}' if r in R.SEED_DEPENDENT else f'{name}{view}{cell[:4]}') for r in
                R.SEED_DEPENDENT + R.SEED_INVARIANT}
    members_reg = ['CAND', 'CAND__d2', 'CAND__d4', 'TWIN', 'REF', 'ALIAS']
    members_rob = ['CAND', 'TWIN', 'REF', 'ALIAS']
    for v in (0, 1):
        _fake_bank(root, 'REG', v, [c.format(v=v) for c in reg], members_reg, values, seeds_fp=fp,
                   paired_pairs=[('CAND', 'TWIN'), ('CAND', 'REF')])
        _fake_bank(root, 'ROBA', v, [c.format(v=v) for c in reg + rob_a], members_rob, values, seeds_fp=fp,
                   paired_pairs=[('CAND', 'TWIN')])
        _fake_bank(root, 'ROBB', v, [c.format(v=v) for c in reg + rob_b], members_rob, values, seeds_fp=fp)
        bad = lambda cell, name, view, kind: values(cell, name, view, kind) + (0.01 if (cell, name) == ('S0R1k0', 'CAND') else 0.0)
        _fake_bank(root, 'BAD', v, [c.format(v=v) for c in reg], members_rob, bad, seeds_fp=fp)
    chk = {}
    M = merge(['REG', 'ROBA', 'ROBB'], root=root)
    chk['rewritten_cells_agree'] = M['conflicts'] == []          # cache re-reads (and differing CPU-time fields) are not conflicts
    import shutil                                                 # a rerun in an explicit folder (--out) reads identically
    shutil.copytree(root / 'ROBA' / 'S1', root / 'ROBA' / 'S1_V2')
    Mo = merge(['REG', 'ROBA', 'ROBB'], root=root, dirs={('ROBA', 1): root / 'ROBA' / 'S1_V2'})
    chk['run_dir_override'] = (Mo['cells'].keys() == M['cells'].keys() and Mo['conflicts'] == []
                               and [r['folder'].endswith('S1_V2') for r in Mo['runs'] if (r['bank'], r['version']) == ('ROBA', 1)] == [True])
    chk['merged_cells'] = len([c for c in M['cells'] if R.parse_cell(c)]) == 24
    Mb = merge(['REG', 'BAD'], root=root)
    chk['planted_conflict_detected'] = (len(Mb['conflicts']) > 0 and all(c['cell'] == 'S0R1k0' and c['member'] == 'CAND'
                                                                         for c in Mb['conflicts'])
                                        and Mb['cells']['S0R1k0']['CAND']['pair']['score_auc'] == M['cells']['S0R1k0']['CAND']['pair']['score_auc'])
    cc = crosscheck_paired(M)
    chk['paired_recomputed_equals_grid'] = cc['n_mismatches'] == 0 and cc['checked'] > 0
    reps = {'CAND': ['CAND__d2', 'CAND__d4']}
    e = entry(M, 'CAND', 'TWIN', reps)
    chk['mc_on_registered_cells_only'] = (all(e['pair']['mc_draws'][c] == 3 for c in e['pair']['mc_draws'] if c in R.REGISTERED)
                                          and all(e['pair']['mc_draws'][c] == 1 for c in e['pair']['mc_draws'] if c not in R.REGISTERED)
                                          and set(e['pair']['cell_se_mc']) == set(R.REGISTERED))
    m0 = R.mc_cell([paired_cell(M, 'S0R1k0', c2, 'TWIN', 'pair')[0] for c2 in ('CAND', 'CAND__d2', 'CAND__d4')],
                   [paired_cell(M, 'S0R1k0', c2, 'TWIN', 'pair')[1] for c2 in ('CAND', 'CAND__d2', 'CAND__d4')])
    chk['mc_cell_value'] = abs(e['pair']['cells']['S0R1k0'] - m0['value']) < 1e-15 and abs(e['pair']['cell_se_mc']['S0R1k0'] - m0['se_mc']) < 1e-15
    one = paired_cell(M, 'S0R1k0', 'CAND', 'TWIN', 'pair')
    d_direct = M['cells']['S0R1k0']['TWIN']['pair']['score_auc'] - M['cells']['S0R1k0']['CAND']['pair']['score_auc']
    chk['pair_sign_comp_minus_cand'] = abs(one[0] - d_direct) < 1e-15 and one[0] > 0
    inc1 = paired_cell(M, 'S0R1k0', 'CAND', 'TWIN', 'income')
    chk['single_view_sign_cand_minus_comp'] = abs(inc1[0] - (M['cells']['S0R1k0']['CAND']['income']['score_auc']
                                                             - M['cells']['S0R1k0']['TWIN']['income']['score_auc'])) < 1e-15
    wires = wires_of(M, members_rob)
    c = compare(M, 'CAND', 'TWIN', reps, wires)
    chk['registered_benefit_positive'] = c['registered']['benefit']['label'] == 'AGG_POSITIVE'
    chk['robustness_confirmed'] = c['robustness']['label'] == 'ROBUSTNESS_CONFIRMED'
    chk['parts_have_8_cells'] = (c['screens_part_A_R3R4_k0k1']['pair']['aggregate']['n_cells'] == 8
                                 and c['screens_part_B_R1R2_k2k3']['pair']['aggregate']['n_cells'] == 8)
    chk['all_cells_24'] = c['screens_all']['pair']['aggregate']['n_cells'] == 24
    chk['targeting_holds_without_tables'] = c['targeting_all']['TABLE_READER_TARGETING'] == 'HOLDS_WITHOUT_TABLE_READERS'
    chk['table_cv_removes_benefit'] = c['targeting_all']['TABLE_CV_SENSITIVITY'] == 'REMOVED_BY_TABLE_CV'
    chk['recipient_gate_pass'] = bool(c['registered']['recipient_gate_pass'])
    n = compare(M, 'TWIN', 'REF', None, wires)                     # benefit = 0.625 - 0.640 < 0
    chk['no_main_benefit_label'] = n['targeting_registered']['TABLE_READER_TARGETING'] == 'NO_MAIN_BENEFIT'
    z = compare(M, 'TWIN', 'ALIAS', None, wires)
    chk['exact_alias_counted_once'] = (z['stability']['registered']['income']['label'] == 'EXACT_ALIAS_COUNTED_ONCE'
                                       and z['registered']['benefit']['label'] == 'ALIAS_ZERO')
    chk['alias_robustness_not_confirmed'] = z['robustness']['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    man = json.loads((C.RESULTS / 'READER_SPLITS.json').read_text())
    gv = validity(M, ['CAND', 'TWIN'], man, {})
    chk['grid_validity_effective'] = gv['label'] == 'READER_SEEDS_EFFECTIVE' and gv['reps_checked'] == ['R1', 'R2', 'R3', 'R4']
    chk['k2_k3_pairs_counted'] = gv['k2_vs_k3']['pairs'] > 0 and gv['k2_vs_k3']['equal_seed_dependent'] == 0
    gvm = validity(M, ['CAND', 'TWIN'], man, {}, members=['TWIN'])
    chk['validity_member_scope'] = 0 < gvm['n_checks'] < gv['n_checks']
    M2 = json.loads(json.dumps(M['cells']))                        # plant a seed-dependent k0 == k1 equality on the minimum set
    for cell in M2:
        if cell.endswith('k1') and 'CAND' in M2[cell]:
            M2[cell]['CAND']['pair']['extras']['score_sha256']['ET5'] = M2[cell[:-1] + '0']['CAND']['pair']['extras']['score_sha256']['ET5']
    gv2 = validity({**M, 'cells': M2}, ['CAND', 'TWIN'], man, {})
    chk['planted_equality_invalidates'] = gv2['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    w = winners(M, ['CAND'])
    chk['winner_counts'] = (w['CAND']['pair']['cells'] == 24 and w['CAND']['pair']['table_reader_winners'] == 0
                            and w['CAND']['pair']['seed_dependent_winners'] == 24)
    h = headline(c)
    chk['headline_fields'] = h['robustness'] == 'ROBUSTNESS_CONFIRMED' and h['part_A_mean'] is not None
    e1 = entry(M, 'CAND', 'TWIN', None)
    chk['single_instance_variant'] = (c['screens_all_single_instance']['pair']['aggregate']['n_cells'] == 24
                                      and e1['pair']['cell_se_mc'] == {} and set(e1['pair']['mc_draws'].values()) == {1}
                                      and h['qualifier_single_instance'] == 'READER_SEEDS_EFFECTIVE_ALL_BLOCKS')
    chk['all_ok'] = all(v for v in chk.values() if isinstance(v, bool))
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--out', required=True)
    r.add_argument('--budget', default=None)
    r.add_argument('--run-dir', action='append', default=[], help='BANK:VERSION=FOLDER (an explicit run folder of a rerun)')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        chk = selftest(); print(json.dumps(chk, indent=1)); return 0 if chk['all_ok'] else 1
    dirs = {}
    for x in a.run_dir:
        key, folder = x.split('=', 1)
        bank, v = key.split(':')
        dirs[(bank, int(v))] = C.ROOT / folder if not Path(folder).is_absolute() else Path(folder)
    rec = run(a.plan, a.out, a.budget, dirs=dirs)
    print(json.dumps({'comparisons': len(rec['comparisons']), 'conflicts': len(rec['conflicts']),
                      'paired_crosscheck': rec['paired_crosscheck']['label'], 'grid_validity': rec['grid_validity']['label'],
                      'cpu_s': rec['cpu_s']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
