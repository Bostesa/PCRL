"""MODEL_SELECT utility of saved releases for the PCRL strong-anchor release campaign (operation 'benchmark', logged; MODEL_SELECT
was used adaptively by every earlier campaign: development evidence, never a fitting input). Derived from nrm.evaluate with this
campaign's paths and two additions: comparators 'NRM:F/C' (a saved release of the source campaign, read-only) and the `source`
command (a saved source release evaluated under this campaign's evaluator, written to this campaign's runs only).

    python3 -m sar.evaluate release --family F --config C [--compare F2/C2 | NRM:F2/C2 | REF:<name> | W ...] [--tag T]
    python3 -m sar.evaluate source --family F --config C [--compare ...] [--tag T]          (a saved nrm release)
    python3 -m sar.evaluate source_frc --family F --config C [--compare ...] [--tag T]      (a saved frc release, read-only)
    python3 -m sar.evaluate source_ddr --family F --config C [--compare ...] [--tag T]      (a saved ddr release, read-only)
    python3 -m sar.evaluate references --name W --name REF:<name> ... [--tag T]
    python3 -m sar.evaluate many --release F/C [--release F2/C2 ...] [--compare ...] [--tag T]   (one record per release)

writes runs/<F>/<C>/EVAL_MS[__<tag>].json (write-once) and appends one row per version to the private ledger runs/EVAL_LEDGER.jsonl
and to the public aggregate table results/<campaign>/dev/UTILITY_LEDGER.csv.

Unchanged definitions (ohn.evaluate.complete_eval -> ihs.evaluate.utility): per version v the 12 point clauses (income and occupation
x U0_S<v> and Ucal_S<v> x accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005; household-clustered paired
SEs) and the men's and women's occupation summed-Brier excess vs U0 (<= 0.005). Both tasks come from the release itself.
Development reserve (separate heuristic, never a cap): ohn.rules.reserve (whole release, 28 conditions over both versions, k = 2,
H_LJ SE floor) and tsz.rules.occupation_reserve (occupation side).
Paired per-person costs against each comparator (loss(this) - loss(comparator) on the same people; accuracy loss, NLL, summed Brier,
men's and women's summed Brier; income and occupation; household-clustered SEs).
Randomized releases (R draws): each metric is computed on each emitted draw (vector and its own argmax), then averaged over the R
draws. Clause values are draw averages; 'se' = mean of the per-draw household-clustered SEs; 'se_mc' = std over draws / sqrt(R),
reported separately (Monte Carlo uncertainty is not household sampling uncertainty).
Comparators: 'F2/C2' (a saved release of this campaign), 'NRM:F2/C2' (a saved release of the source campaign, nrm.release.load,
hash-checked), 'REF:<name>' (a frozen inherited release, ohn.devrun.reference_release),
'W' (the previous campaign's witness: A1_PO_16 S<v> occupation + H3 income S<v>).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from . import common as C
from . import release as RL

OUT_PUBLIC = C.RESULTS / 'dev'
LEDGER = C.RUNS / 'EVAL_LEDGER.jsonl'
METRICS = ('acc_loss', 'nll', 'brier', 'men_brier', 'women_brier')


def ms(purpose):
    return C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'benchmark', purpose)


def refs_of(v, S):
    from tsz.evaluate import refs_of as tsz_refs                     # frozen U0 / Ucal / T_TREE / TASK_REFIT / JOINT (no labels)
    return tsz_refs(v, S)


def comparator_release(spec, v):
    """{'income': (n,2), 'occupation': (n,5)} or randomized stacks, plus a provenance string."""
    if spec == 'W':
        from tsz.evaluate import source_a1, h3_income
        return {'income': h3_income(v), 'occupation': source_a1(v)}, 'W = A1_PO_16 S%d + H3 income S%d' % (v, v)
    if spec.startswith('REF:'):
        from ohn.devrun import reference_release
        return reference_release(spec[4:], v, 'MODEL_SELECT'), f'frozen reference {spec[4:]} S{v}'
    if spec.startswith('NRM:'):
        from nrm import release as NRL
        fam, cfg = spec[4:].split('/')
        arr, rec = NRL.load(fam, cfg, v, roles=('MODEL_SELECT',))
        return arr['MODEL_SELECT'], f'source nrm {fam}/{cfg}/V{v} ({rec["kind"]})'
    if spec.startswith('FRC:'):
        from frc import release as FRL
        fam, cfg = spec[4:].split('/')
        arr, rec = FRL.load(fam, cfg, v, roles=('MODEL_SELECT',))
        return arr['MODEL_SELECT'], f'source frc {fam}/{cfg}/V{v} ({rec["kind"]})'
    if spec.startswith('ROLE:'):
        from . import roles as RO
        real = RO.resolve(spec, v)
        if real is None:
            raise KeyError(f'{spec} has no version {v} configuration')
        if real.startswith('REF:'):
            from ohn.devrun import reference_release
            return reference_release(real[4:], v, 'MODEL_SELECT'), f'{spec} = frozen reference {real[4:]} S{v} (alias)'
        fam, cfg = real.split('/')
        arr, rec = RL.load(fam, cfg, v, roles=('MODEL_SELECT',))
        return arr['MODEL_SELECT'], f'{spec} = {fam}/{cfg}/V{v} ({rec["kind"]})'
    if spec.startswith('DDR:'):
        from ddr import release as DRL
        fam, cfg = spec[4:].split('/')
        arr, rec = DRL.load(fam, cfg, v, roles=('MODEL_SELECT',))
        return arr['MODEL_SELECT'], f'source ddr {fam}/{cfg}/V{v} ({rec["kind"]})'
    fam, cfg = spec.split('/')
    arr, rec = RL.load(fam, cfg, v, roles=('MODEL_SELECT',))
    return arr['MODEL_SELECT'], f'{fam}/{cfg}/V{v} ({rec["kind"]})'


def draws_of(rel):
    """List of per-draw {'income', 'occupation'} (a deterministic release is one draw)."""
    inc, occ = rel['income'], rel['occupation']
    if inc.ndim == 2 and occ.ndim == 2:
        return [{'income': inc, 'occupation': occ}]
    R = inc.shape[0] if inc.ndim == 3 else occ.shape[0]
    return [{'income': inc[r] if inc.ndim == 3 else inc, 'occupation': occ[r] if occ.ndim == 3 else occ} for r in range(R)]


def per_person(S, rel):
    """Per-person losses averaged over draws: {task: {'acc_loss', 'nll', 'brier'}} plus per-draw means for the MC SE."""
    out, per_draw = {}, []
    ds = draws_of(rel)
    for task in C.TASKS:
        acc = []; nll = []; br = []
        for d in ds:
            t = C.task_vectors(S[task], d[task], C.K[task])
            acc.append(1.0 - t['accuracy']); nll.append(t['nll']); br.append(t['brier'])
        out[task] = {'acc_loss': np.mean(acc, 0), 'nll': np.mean(nll, 0), 'brier': np.mean(br, 0),
                     'draws': {'acc_loss': np.asarray(acc), 'nll': np.asarray(nll), 'brier': np.asarray(br)}}
    return out


def _mc_se(draw_means):
    """Monte Carlo SE of one release's draw-averaged mean: SD over its per-draw means / sqrt(R); 0 for a deterministic release."""
    x = np.asarray(draw_means, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def paired_costs(a, b, S):
    """loss(a) - loss(b) per task and metric (a, b: per_person outputs), household-clustered SEs. Monte Carlo SE (ENG1-06): each
    release's own MC SE from its per-draw means, combined in quadrature (independent channels); 0 for deterministic releases."""
    from ihs.evaluate import cluster_se, subgroup_se
    unit, sex = S['unit'], S['sex']
    out = {}

    def mc(m, task=None, mask=None):
        parts = []
        for x in (a, b):
            if task is None:                              # total NLL: per-draw income mean + occupation mean
                ti = [d.mean() for d in x['income']['draws']['nll']]
                to = [d.mean() for d in x['occupation']['draws']['nll']]
                k = max(len(ti), len(to))
                ti, to = (ti * k if len(ti) == 1 else ti), (to * k if len(to) == 1 else to)
                parts.append(_mc_se(np.add(ti[:k], to[:k])))
            else:
                parts.append(_mc_se([d[mask].mean() if mask is not None else d.mean() for d in x[task]['draws'][m]]))
        return float(np.sqrt(sum(p * p for p in parts)))

    for task in C.TASKS:
        for m in ('acc_loss', 'nll', 'brier'):
            d = a[task][m] - b[task][m]
            out[f'{task}|{m}'] = {'value': float(d.mean()), 'se': cluster_se(d, unit), 'se_mc': mc(m, task)}
            if m == 'brier':
                for g, name in ((0, 'men_brier'), (1, 'women_brier')):
                    mask = sex == g
                    out[f'{task}|{name}'] = {'value': float(d[mask].mean()), 'se': subgroup_se(d, mask, unit),
                                             'se_mc': mc('brier', task, mask)}
    tot = (a['income']['nll'] + a['occupation']['nll']) - (b['income']['nll'] + b['occupation']['nll'])
    out['total|nll'] = {'value': float(tot.mean()), 'se': cluster_se(tot, unit), 'se_mc': mc('nll'),
                        'definition': 'per-person income + occupation NLL difference (the statistical reviewer\'s primary frontier cost)'}
    return out


def utility_of(v, rel, S, ctx):
    """complete_eval over draws (draw-averaged clause values; SE = mean per-draw SE; se_mc separately)."""
    from ohn import evaluate as OE
    refs, others, _ = ctx
    evs = [OE.complete_eval(v, d['income'], d['occupation'], S, refs, others) for d in draws_of(rel)]
    if len(evs) == 1:
        return evs[0], 1
    base = json.loads(json.dumps(evs[0]['utility'], default=float))
    for key in ('clauses', 'guards'):
        for i, c in enumerate(base[key]):
            vals = np.array([float(e['utility'][key][i]['value']) for e in evs])
            ses = np.array([float(e['utility'][key][i].get('se', np.nan)) for e in evs])
            c['value'] = float(vals.mean()); c['se'] = float(np.nanmean(ses))
            c['se_mc'] = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            c['draw_values'] = vals.tolist()
            # derived fields recomputed from the draw-averaged value (ENG1-06: never draw 0's)
            cap = float(c['cap'])
            c['margin'] = cap - c['value']
            if 'normalized' in c:                         # value / cap (verifier V2B: was left at draw 0's value)
                c['normalized'] = c['value'] / cap
            c['margin_se_units'] = c['margin'] / c['se'] if c['se'] > 0 else (np.inf if c['margin'] > 0 else -np.inf)
            c['point_pass'] = bool(c['value'] <= cap)
            c['draw_point_pass'] = [bool(x <= cap) for x in vals]
    base['point_pass_12'] = all(c['point_pass'] for c in base['clauses'])
    base['guard_pass'] = all(g['point_pass'] for g in base['guards'])
    base['by_sex'] = None
    base['task_summary'] = None
    base['draw_averaged_note'] = ('values, SEs and point_pass are draw averages over the emitted draws; by_sex and task_summary of '
                                  'single draws are not carried (see the per-draw records)')
    ev = {'utility': base, 'distinct_vectors': {t: [e['distinct_vectors'][t] for e in evs] for t in C.TASKS},
          'wire_sha256': [e['wire_sha256'] for e in evs], 'note': f'draw-averaged over {len(evs)} independent one-draw-per-person draws'}
    return ev, len(evs)


def wires_of(rel):
    """The selection wire hashes (ENG1-05): scm_bench wire_sha (probabilities + argmax) per task, as nrm.devreaders records them; a
    list over the emitted draws for a randomized release."""
    out = {}
    for t in C.TASKS:
        a = np.asarray(rel[t])
        out[t] = [C.wire_sha(a[r]) for r in range(a.shape[0])] if a.ndim == 3 else C.wire_sha(a)
    return out


def point_status(u):
    pts = [float(c['value']) <= float(c['cap']) for c in u['clauses']] + [float(g['value']) <= float(g['cap']) for g in u['guards']]
    worst = max([float(c['value']) / float(c['cap']) for c in u['clauses']] + [float(g['value']) / float(g['cap']) for g in u['guards']])
    return {'point_pass_12_plus_2': bool(all(pts)) and len(pts) == 14, 'max_value_over_cap': worst,
            'failed': [f"{c['task']}|{c['reference']}|{c['metric']}" for c in u['clauses'] if float(c['value']) > float(c['cap'])]
            + [f"guard|{g['group']}" for g in u['guards'] if float(g['value']) > float(g['cap'])]}


def reserve_of(u, v):
    from ohn import rules as OR
    from tsz import rules as TR
    whole = OR.reserve({v: u})
    occ = TR.occupation_reserve({v: u})['per_version'][v]
    return {'whole_release_met': whole['per_seed'][v]['status'] == OR.MET, 'min_whole_ratio': whole['min_ratio'],
            'whole_binding': whole['binding_condition'], 'occ_side_met': occ['met'], 'min_occ_ratio': occ['min_ratio'],
            'occ_binding': occ['binding'], 'income_reserve_met': occ['income']['income_reserve_met'],
            'min_income_ratio': occ['income']['min_income_ratio']}


def release_cmd(family, config, compare=(), tag=None, source=False):
    t0 = time.process_time()
    if source == 'role':                                   # a role member (sar.roles): per version the configuration it names
        from . import roles as RO
        d = C.RUNS / 'ROLES' / config

        def loader(f, c, v):
            real = RO.resolve('ROLE:' + c, v)
            if real.startswith('REF:'):                    # an exact alias of the frozen START reference in this version
                from ohn.devrun import reference_release
                arr = reference_release(real[4:], v, 'MODEL_SELECT')
                return ({'MODEL_SELECT': arr}, {'release_sha256': C.wire_sha(arr['income']) + C.wire_sha(arr['occupation']),
                                                'kind': f'alias of frozen reference {real[4:]}', 'card': None})
            fam_, cfg_ = real.split('/')
            return RL.load(fam_, cfg_, v, roles=('MODEL_SELECT',))
        exists = (lambda f, c, v: RO.resolve('ROLE:' + c, v) is not None)
    elif source == 'frc':
        from frc import release as FRL
        d = C.RUNS / 'SOURCE_FRC' / family / config
        loader, exists = (lambda f, c, v: FRL.load(f, c, v, roles=('MODEL_SELECT',))), FRL.exists
    elif source == 'ddr':
        from ddr import release as DRL
        d = C.RUNS / 'SOURCE_DDR' / family / config
        loader, exists = (lambda f, c, v: DRL.load(f, c, v, roles=('MODEL_SELECT',))), DRL.exists
    elif source:
        from nrm import release as NRL
        d = C.RUNS / 'SOURCE' / family / config
        loader, exists = (lambda f, c, v: NRL.load(f, c, v, roles=('MODEL_SELECT',))), NRL.exists
    else:
        d = RL.rdir(family, config, 0).parent
        loader, exists = (lambda f, c, v: RL.load(f, c, v, roles=('MODEL_SELECT',))), RL.exists
    out = d / (f'EVAL_MS__{tag}.json' if tag else 'EVAL_MS.json')
    if out.exists():
        raise PermissionError(f'{out.name} exists (write-once; use --tag for a new evaluation record)')
    S = ms(f'benchmark: MODEL_SELECT utility, reserve and paired costs of {"source nrm " if source else ""}{family}/{config} '
           f'(adaptive development evidence; comparators {list(compare)})')
    res = {'schema': 'SAR1-eval-ms-v1', 'family': family, 'config_id': config, 'source_release': bool(source), 'utc': C.now(),
           'versions': {}}
    rows = []
    for v in C.VERSIONS:
        if not exists(family, config, v):
            res['versions'][str(v)] = {'missing': True}
            continue
        arr, rec = loader(family, config, v)
        rel = arr['MODEL_SELECT']
        ctx = refs_of(v, S)
        ev, R = utility_of(v, rel, S, ctx)
        u = ev['utility']
        pp = per_person(S, rel)
        costs = {}
        for spec in compare:
            if spec.startswith('ROLE:'):
                from . import roles as RO
                if RO.resolve(spec, v) is None:
                    costs[spec] = {'missing': True, 'note': f'{spec} has no version {v} configuration'}
                    continue
            if '/' in spec and ':' not in spec and not RL.exists(*spec.split('/'), v):
                costs[spec] = {'missing': True, 'note': f'{spec} has no version {v} release (an alias or NOT_FEASIBLE there)'}
                continue
            other, prov = comparator_release(spec, v)
            costs[spec] = {'provenance': prov, 'costs': paired_costs(pp, per_person(S, other), S)}
        absolute = {t: {m: float(pp[t][m].mean()) for m in ('acc_loss', 'nll', 'brier')} for t in C.TASKS}
        absolute['occupation']['men_brier'] = float(pp['occupation']['brier'][S['sex'] == 0].mean())
        absolute['occupation']['women_brier'] = float(pp['occupation']['brier'][S['sex'] == 1].mean())
        ps, rs = point_status(u), reserve_of(u, v)
        res['versions'][str(v)] = {'release_sha256': rec['release_sha256'], 'kind': rec['kind'], 'card': rec.get('card'),
                                   'randomized': bool(rec.get('randomized')), 'draws': R, 'utility': u, 'point': ps,
                                   'reserve_without_mc': rs, 'absolute': absolute, 'paired_costs': costs,
                                   'distinct_vectors': ev.get('distinct_vectors'), 'wire_sha256_eval': wires_of(rel)}
        row = {'family': ('FRC:' if source == 'frc' else 'NRM:' if source else '') + family, 'config_id': config, 'version': v,
               'kind': rec['kind'], 'draws': R,
               'tag': tag or '',
               'point_pass_12_plus_2': ps['point_pass_12_plus_2'], 'max_value_over_cap': round(ps['max_value_over_cap'], 4),
               'failed': ';'.join(ps['failed']), 'whole_reserve_met': rs['whole_release_met'],
               'min_whole_ratio': round(rs['min_whole_ratio'], 3), 'occ_reserve_met': rs['occ_side_met'],
               'min_occ_ratio': round(rs['min_occ_ratio'], 3), 'min_income_ratio': round(rs['min_income_ratio'], 3)}
        for c in u['clauses']:
            row[f"{c['task']}|{c['reference']}|{c['metric']}"] = round(float(c['value']), 6)
        for g in u['guards']:
            row[f"guard|{g['group']}"] = round(float(g['value']), 6)
        rows.append(row)
    vs = {v: res['versions'][str(v)] for v in C.VERSIONS if not res['versions'][str(v)].get('missing')}
    if len(vs) == len(C.VERSIONS):                       # the governing reserve rule (MC SE in quadrature; ENG1-06)
        from nrm import rules as RU                      # the NRM1 reserve rule, unchanged
        try:
            res['reserve_rules'] = RU.reserve_status({v: {'utility': e['utility']} for v, e in vs.items()})
        except Exception as ex:                          # recorded, never hidden
            res['reserve_rules'] = {'status': 'ERROR', 'error': f'{type(ex).__name__}: {ex}'}
    res['cpu_s'] = time.process_time() - t0
    res['unit_id'] = os.environ.get(C.UNIT_ENV)
    res['note'] = ('development evidence on adaptively used MODEL_SELECT; the two-SE reserve is a separate heuristic, not a cap; '
                   'never a fitting input; reserve_without_mc per version is descriptive, reserve_rules governs')
    C.write_json_x(out, res)
    LEDGER.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(LEDGER, 'a') as f:
        for r in rows:
            f.write(json.dumps({**r, 'utc': res['utc'], 'eval_file': str(out.relative_to(C.ROOT))}, sort_keys=True) + '\n')
    _public_table()
    print(json.dumps([{k: r[k] for k in ('family', 'config_id', 'version', 'point_pass_12_plus_2', 'max_value_over_cap', 'failed',
                                         'min_whole_ratio', 'min_occ_ratio')} for r in rows]))
    return res


def reference_cmd(names, tag=None):
    """The same utility, reserve and absolute metrics for frozen comparators ('W', 'REF:<name>'), written once to
    runs/REFERENCES/EVAL_MS[__<tag>].json and the ledgers (family 'REF')."""
    out = C.RUNS / 'REFERENCES' / (f'EVAL_MS__{tag}.json' if tag else 'EVAL_MS.json')
    if out.exists():
        raise PermissionError(f'runs/REFERENCES/{out.name} exists (write-once)')
    S = ms(f'benchmark: MODEL_SELECT utility and reserve of the frozen comparators {list(names)} under this campaign\'s evaluator '
           '(adaptive development evidence)')
    res, rows = {'schema': 'SAR1-eval-ms-refs-v1', 'utc': C.now(), 'references': {}}, []
    for name in names:
        for v in C.VERSIONS:
            rel, prov = comparator_release(name, v)
            ev, _ = utility_of(v, rel, S, refs_of(v, S))
            u = ev['utility']
            pp = per_person(S, rel)
            absolute = {t: {m: float(pp[t][m].mean()) for m in ('acc_loss', 'nll', 'brier')} for t in C.TASKS}
            absolute['occupation']['men_brier'] = float(pp['occupation']['brier'][S['sex'] == 0].mean())
            absolute['occupation']['women_brier'] = float(pp['occupation']['brier'][S['sex'] == 1].mean())
            ps, rs = point_status(u), reserve_of(u, v)
            res['references'].setdefault(name, {})[str(v)] = {'provenance': prov, 'utility': u, 'point': ps, 'reserve': rs,
                                                              'absolute': absolute, 'distinct_vectors': ev.get('distinct_vectors'),
                                                              'wire_sha256': wires_of(rel)}
            row = {'family': 'REF', 'config_id': name, 'version': v, 'kind': 'reference', 'draws': 1, 'tag': '',
                   'point_pass_12_plus_2': ps['point_pass_12_plus_2'], 'max_value_over_cap': round(ps['max_value_over_cap'], 4),
                   'failed': ';'.join(ps['failed']), 'whole_reserve_met': rs['whole_release_met'],
                   'min_whole_ratio': round(rs['min_whole_ratio'], 3), 'occ_reserve_met': rs['occ_side_met'],
                   'min_occ_ratio': round(rs['min_occ_ratio'], 3), 'min_income_ratio': round(rs['min_income_ratio'], 3)}
            for c in u['clauses']:
                row[f"{c['task']}|{c['reference']}|{c['metric']}"] = round(float(c['value']), 6)
            for g in u['guards']:
                row[f"guard|{g['group']}"] = round(float(g['value']), 6)
            rows.append(row)
    res['unit_id'] = os.environ.get(C.UNIT_ENV)
    C.write_json_x(out, res)
    LEDGER.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(LEDGER, 'a') as f:
        for r in rows:
            f.write(json.dumps({**r, 'utc': res['utc'], 'eval_file': str(out.relative_to(C.ROOT))}, sort_keys=True) + '\n')
    _public_table()
    print(json.dumps([{k: r[k] for k in ('config_id', 'version', 'point_pass_12_plus_2', 'max_value_over_cap', 'min_whole_ratio')}
                      for r in rows]))
    return res


def _public_table():
    if not LEDGER.is_file():
        return
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    keys = sorted({k for r in rows for k in r if k != 'eval_file'})
    lead = ['utc', 'family', 'config_id', 'version', 'kind', 'tag', 'draws', 'point_pass_12_plus_2', 'max_value_over_cap', 'failed',
            'whole_reserve_met', 'min_whole_ratio', 'occ_reserve_met', 'min_occ_ratio', 'min_income_ratio']
    cols = [k for k in lead if k in keys] + [k for k in keys if k not in lead]
    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(OUT_PUBLIC / 'UTILITY_LEDGER.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def selftest():
    rng = np.random.default_rng(1)
    n = 400
    S = {'unit': np.arange(n) // 2, 'sex': rng.integers(0, 2, n), 'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n)}
    rel = {'income': rng.dirichlet(np.ones(2), n), 'occupation': rng.dirichlet(np.ones(5), n)}
    a = per_person(S, rel)
    z = paired_costs(a, a, S)
    ok = {'identical_zero_cost': all(abs(r['value']) == 0.0 for r in z.values())}
    stack = {'income': np.stack([rel['income']] * 3), 'occupation': np.stack([rel['occupation']] * 3)}
    b = per_person(S, stack)
    zb = paired_costs(b, a, S)
    ok['constant_draws_equal_deterministic'] = all(abs(r['value']) < 1e-12 for r in zb.values() if isinstance(r, dict))
    ok['three_draws'] = len(draws_of(stack)) == 3
    ok['constant_draws_zero_mc_se'] = all(abs(r.get('se_mc', 0.0)) < 1e-12 for r in zb.values() if isinstance(r, dict))
    # wire hashes in the grid's format (ENG1-05): deterministic -> one hash per task; randomized -> one per draw
    from scm_bench.common import wire_sha
    w1, w3 = wires_of(rel), wires_of(stack)
    ok['wire_format_deterministic'] = w1['income'] == wire_sha(rel['income']) and w1['occupation'] == wire_sha(rel['occupation'])
    ok['wire_format_randomized'] = w3['occupation'] == [wire_sha(stack['occupation'][r]) for r in range(3)]
    # randomized MC SE: draws that differ give a positive MC SE
    noisy = {'income': stack['income'], 'occupation': np.stack([rel['occupation'], rng.dirichlet(np.ones(5), n), rel['occupation']])}
    zn = paired_costs(per_person(S, noisy), a, S)
    ok['differing_draws_positive_mc_se'] = zn['occupation|nll']['se_mc'] > 0 and zn['total|nll']['se_mc'] > 0
    ok['all_ok'] = all(ok.values())
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('release'); r.add_argument('--family', required=True); r.add_argument('--config', required=True)
    r.add_argument('--compare', action='append', default=[]); r.add_argument('--tag', default=None)
    s = sub.add_parser('source'); s.add_argument('--family', required=True); s.add_argument('--config', required=True)
    s.add_argument('--compare', action='append', default=[]); s.add_argument('--tag', default=None)
    s2 = sub.add_parser('source_frc'); s2.add_argument('--family', required=True); s2.add_argument('--config', required=True)
    s2.add_argument('--compare', action='append', default=[]); s2.add_argument('--tag', default=None)
    s3 = sub.add_parser('source_ddr'); s3.add_argument('--family', required=True); s3.add_argument('--config', required=True)
    s3.add_argument('--compare', action='append', default=[]); s3.add_argument('--tag', default=None)
    s4 = sub.add_parser('roles'); s4.add_argument('--role', action='append', required=True)
    s4.add_argument('--compare', action='append', default=[]); s4.add_argument('--tag', default=None)
    f = sub.add_parser('references'); f.add_argument('--name', action='append', required=True); f.add_argument('--tag', default=None)
    mny = sub.add_parser('many'); mny.add_argument('--release', action='append', required=True, help='FAMILY/CONFIG (this campaign)')
    mny.add_argument('--compare', action='append', default=[]); mny.add_argument('--tag', default=None)
    sub.add_parser('selftest')
    a = ap.parse_args(argv)
    if a.cmd == 'release':
        release_cmd(a.family, a.config, a.compare, a.tag); return 0
    if a.cmd == 'source':
        release_cmd(a.family, a.config, a.compare, a.tag, source=True); return 0
    if a.cmd == 'source_frc':
        release_cmd(a.family, a.config, a.compare, a.tag, source='frc'); return 0
    if a.cmd == 'source_ddr':
        release_cmd(a.family, a.config, a.compare, a.tag, source='ddr'); return 0
    if a.cmd == 'roles':                                   # role members (sar.roles), one record each under runs/ROLES/<name>
        for name in a.role:
            release_cmd('ROLE', name, [c for c in a.compare if c != f'ROLE:{name}'], a.tag, source='role')
        return 0
    if a.cmd == 'references':
        reference_cmd(a.name, a.tag); return 0
    if a.cmd == 'many':                                   # several releases of this campaign in one unit (one record each)
        for spec in a.release:
            fam, cfg = spec.split('/')
            release_cmd(fam, cfg, [c for c in a.compare if c != spec], a.tag)
        return 0
    res = selftest(); print(json.dumps(res)); return 0 if res['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
