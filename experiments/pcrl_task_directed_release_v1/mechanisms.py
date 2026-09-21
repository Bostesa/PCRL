"""Empirical channel tables and deployment releases, with no data loading/fitting.

Privacy laws use their own protected-label masks. Utility costs contain state
mass already. A deployment channel emits a token; its entire probability row
is used only for exact expectation, never appended to the released features.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from scipy.special import logit

from . import finite
from .config import OUT
from .encoding import CLIP

FAMILIES = ('T0', 'Ttask', 'Trisk')
AGGREGATION_TOLERANCE = 1e-12
PROBABILITY_TOLERANCE = 1e-8
HISTORICAL = ('leace_A0', 'splince_A0', 'optnet16_L1', 'optnet16_L2', 'optnet16_C1')


def _codes(values, n_states, n_rows):
    a = np.asarray(values)
    if (a.shape != (n_rows,) or not np.isfinite(a).all()
            or np.any(a != np.floor(a)) or np.any((a < 0) | (a >= n_states))):
        raise ValueError('Invalid code indices')
    return a.astype(np.int64)


def _probabilities(values, rows=None):
    a = np.asarray(values, float)
    if (a.ndim != 2 or not a.shape[1] or (rows is not None and len(a) != rows)
            or not np.isfinite(a).all() or np.any((a < 0) | (a > 1))
            or np.max(np.abs(a.sum(1)-1), initial=0) > PROBABILITY_TOLERANCE):
        raise ValueError('Invalid row-stochastic token probabilities')
    return a


def _predictions(values, rows):
    a = np.asarray(values, float)
    if (a.ndim != 2 or len(a) != rows or not a.shape[1]
            or not np.isfinite(a).all() or np.any((a <= 0) | (a >= 1))):
        raise ValueError('Invalid fixed action probabilities')
    return a


def _raw_support(s, c, t, shape, weights):
    fields = {}
    for name, w in (('count', np.ones(len(t), dtype=np.int64)),
                    ('sumw', weights), ('sumw2', weights**2)):
        a = np.zeros(shape, dtype=w.dtype)
        np.add.at(a, (s, c, t), w)
        fields[name] = a
    fields['ess'] = np.divide(fields['sumw']**2, fields['sumw2'],
                             out=np.zeros(shape), where=fields['sumw2'] > 0)
    return fields


def _aggregate_last(a, n_parent):
    a = np.asarray(a)
    return a.reshape(*a.shape[:-1], n_parent, 2).sum(-1)


def _maximum_error(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b)), initial=0))


def estimate_tables(ctx, encoder, encoded, roles, *, actions=17):
    """Estimate U/PWGTP laws on mechanism RF rows, retaining empty cells.

    ``state_mass`` is the count of *all* mechanism people, including missing
    task/protected labels, for the registered unsupported-state convention.
    Counts and weighted ESS are descriptive support, without pseudocounts.
    """
    pool = ctx['pools']['representation_fit']
    enc = encoded['representation_fit']
    n = len(pool['ha'])
    idx = _codes(roles['mechanism'], n, len(roles['mechanism']))
    if not len(idx) or len(np.unique(idx)) != len(idx):
        raise ValueError('Mechanism rows must be nonempty and unique')
    w = np.asarray(pool['weights'], float)[idx]
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('PWGTP must be finite and positive')
    y = np.asarray(pool['labels']['same_residence'])[idx]
    valid_y = np.isin(y, [0, 1])
    if not np.any(valid_y):
        raise ValueError('No valid utility labels in mechanism rows')
    predictions = _predictions(enc['actions'][actions], n)[idx]
    ca, cab = encoder.partitions.assign(pool['ha'], pool['hb'])
    n_ca = len(encoder.partitions.centers)
    conditioning = {'A': (_codes(ca, n_ca, n)[idx], n_ca),
                    'AB': (_codes(cab, 2*n_ca, n)[idx], 2*n_ca)}
    tables = {}
    for family in FAMILIES:
        nt = encoder.code.n_states(family)
        all_t = _codes(enc['codes'][family], nt, n)
        if family != 'T0' and not np.array_equal(all_t//2, enc['codes']['T0']):
            raise ValueError('Refined input does not preserve its T0 parent')
        t = all_t[idx]
        d_u = finite.cost_table(t[valid_y], y[valid_y], predictions[valid_y], nt)
        d_w = finite.cost_table(t[valid_y], y[valid_y], predictions[valid_y], nt, w[valid_y])
        laws, support, valid_rows = {}, {}, {}
        for protected, ns in (('SEX', 2), ('RAC1P', 9)):
            s = np.asarray(pool['labels'][protected])[idx]
            valid = np.isfinite(s) & (s == np.floor(s)) & (s >= 0) & (s < ns)
            if not np.any(valid):
                raise ValueError(f'No valid {protected} labels in mechanism rows')
            ss = s[valid].astype(np.int64)
            valid_rows[protected] = int(valid.sum())
            for view, (c, nc) in conditioning.items():
                key = f'{view}/{protected}'
                laws[key+'/U'] = finite.joint_table(ss, c[valid], t[valid], ns, nc, nt)
                laws[key+'/W'] = finite.joint_table(ss, c[valid], t[valid], ns, nc, nt, w[valid])
                support[key] = _raw_support(ss, c[valid], t[valid], (ns, nc, nt), w[valid])
        tables[family] = {'cost_U': d_u, 'cost_W': d_w, 'cost': .5*(d_u+d_w),
                          'state_mass': np.bincount(t, minlength=nt), 'roles': laws,
                          'support': support, 'actions': actions,
                          'population': {'mechanism_rows': len(idx),
                                         'utility_valid_rows': int(valid_y.sum()),
                                         'protected_valid_rows': valid_rows}}
    coarse = tables['T0']
    nc = encoder.code.n_states('T0')
    for family in FAMILIES[1:]:
        fine = tables[family]
        errors = {key: _maximum_error(fine[key].reshape(nc, 2, -1).sum(1), coarse[key])
                  for key in ('cost_U', 'cost_W', 'cost')}
        errors['state_mass'] = _maximum_error(fine['state_mass'].reshape(nc, 2).sum(1), coarse['state_mass'])
        for key, p in fine['roles'].items():
            errors['joint/'+key] = _maximum_error(_aggregate_last(p, nc), coarse['roles'][key])
        for key, fields in fine['support'].items():
            for field in ('count', 'sumw', 'sumw2'):
                # Raw survey-weight sums can be large: normalized support laws
                # already receive the absolute 1e-12 check above.
                a = _aggregate_last(fields[field], nc)
                b = coarse['support'][key][field]
                if not np.allclose(a, b, rtol=1e-12, atol=1e-12):
                    raise AssertionError('Refined support does not aggregate')
        maximum = max(errors.values())
        if maximum > AGGREGATION_TOLERANCE:
            raise AssertionError(f'Refinement aggregation failed: {errors}')
        fine['aggregation'] = {'maximum_error': maximum, 'errors': errors}
    coarse['aggregation'] = {'maximum_error': 0., 'errors': {}}
    return tables


def _embedding(ctx, encoder, encoded, tables, family, coarse_q, actions):
    parents = np.asarray(encoder.code.parents(family), dtype=int)
    q = coarse_q[parents]
    errors = {'objective': abs(float(np.sum(tables[family]['cost']*q))
                               -float(np.sum(tables['T0']['cost']*coarse_q)))}
    for key, p in tables[family]['roles'].items():
        fine_law = np.einsum('sct,tz->scz', p, q)
        coarse_law = np.einsum('sct,tz->scz', tables['T0']['roles'][key], coarse_q)
        errors['joint/'+key] = _maximum_error(fine_law, coarse_law)
    for pool, data in ctx['pools'].items():
        enc = encoded[pool]
        coarse_t = _codes(enc['codes']['T0'], len(coarse_q), len(data['ha']))
        fine_t = _codes(enc['codes'][family], len(q), len(data['ha']))
        if not np.array_equal(parents[fine_t], coarse_t):
            raise ValueError('Deployment refinement does not preserve parent')
        errors['deployment/'+pool] = _maximum_error(q[fine_t], coarse_q[coarse_t])
        g = _predictions(enc['actions'][actions], len(coarse_t))
        errors['prediction/'+pool] = _maximum_error(np.sum(q[fine_t]*g, 1), np.sum(coarse_q[coarse_t]*g, 1))
    maximum = max(errors.values())
    if maximum > AGGREGATION_TOLERANCE:
        raise AssertionError(f'Coarse deployment embedding failed: {errors}')
    return q, {'maximum_error': maximum, 'errors': errors}


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fit_map(ctx, encoder, encoded, tables, spec, out_dir, *, coarse_solution=None):
    """Fit a registered finite map and retain independent empirical checks."""
    family, policy, budget = spec['input'], spec['policy'], spec['budget']
    if family not in FAMILIES or policy not in ('L', 'C', 'U'):
        raise ValueError('Unknown finite map family/policy')
    if (policy == 'U') != (budget is None):
        raise ValueError('Only U policy has an unconstrained budget')
    actions = spec.get('max_actions', 17)
    table = tables[family]
    if table['actions'] != actions:
        raise ValueError('Tables and map must use the same action dictionary')
    selected = {k: p for k, p in table['roles'].items()
                if policy == 'C' or (policy == 'L' and k.startswith('A/'))}
    witness, embedding = None, None
    if coarse_solution is not None:
        if (family == 'T0' or coarse_solution['input'] != 'T0'
                or coarse_solution['policy'] != policy or coarse_solution['budget'] != budget
                or coarse_solution['actions'] != actions or not coarse_solution['feasible']):
            raise ValueError('Coarse witness must match policy, budget and action dictionary')
        coarse_q = _probabilities(coarse_solution['Q'], encoder.code.n_states('T0'))
        witness, embedding = _embedding(ctx, encoder, encoded, tables, family, coarse_q, actions)
    result = finite.solve(table['cost'], selected, budget,
                          parent=encoder.code.parents(family), state_mass=table['state_mass'],
                          zero_action=encoder.dictionaries[actions]['zero_action'], embedded_q=witness)
    result.update({'input': family, 'policy': policy, 'budget': budget,
                   'configuration': spec['configuration'], 'actions': actions,
                   'constrained_roles': sorted(selected), 'embedding': embedding,
                   'aggregation': table['aggregation'], 'population': table['population'],
                   'constant_action': int(np.argmin(tables['T0']['cost'].sum(0))),
                   'cost': table['cost']})
    if result['Q'] is not None:
        q = _probabilities(result['Q'], encoder.code.n_states(family))
        information = {k: finite.cmi(p, q) for k, p in table['roles'].items()}
        objective = float(np.sum(table['cost']*q))
        if abs(objective-result['objective']) > 1e-12:
            raise AssertionError('Returned objective differs from empirical cost')
        if result['feasible'] and budget is not None:
            if any(information[k] > budget+finite.ACCEPTANCE_TOLERANCES['cmi'] for k in selected):
                raise AssertionError('Independent constrained-role check failed')
        result['independent_cmi'] = information
        result['independent_objective'] = objective
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {key: table[key] for key in ('cost_U', 'cost_W', 'cost', 'state_mass')}
    arrays.update({'joint/'+k: p for k, p in table['roles'].items()})
    arrays.update({'support/'+k+'/'+f: a for k, fields in table['support'].items() for f, a in fields.items()})
    np.savez_compressed(path/'tables.npz', **arrays)
    if result['Q'] is not None:
        np.savez_compressed(path/'Q.npz', Q=result['Q'])
    metadata = {k: v for k, v in result.items() if k not in ('Q', 'cost')}
    (path/'metadata.json').write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True, allow_nan=False)+'\n')
    return result


def _channel(mechanism, encoder, family, actions):
    if mechanism is None or not mechanism.get('feasible', False):
        raise ValueError('A verified feasible mechanism is required')
    if mechanism['input'] != family or mechanism['actions'] != actions:
        raise ValueError('Mechanism input family/action dictionary mismatch')
    q = _probabilities(mechanism['Q'], encoder.code.n_states(family))
    if q.shape[1] != len(encoder.dictionaries[actions]['offsets']):
        raise ValueError('Channel width differs from action dictionary')
    return q


def build_release(ctx, encoder, encoded, name, *, mechanism=None, inputs_root=None,
                  erasers=None, actions=17):
    """Build exact token laws and optional deterministic auxiliary features.

    H is supplied unchanged by the audit caller. ``global_offsets`` contains
    the H-dependent predictions for the frozen action library, for the H-only
    utility comparator. Neither that library nor token probabilities are aux.
    """
    mixture = re.fullmatch(r'(T0|Ttask|Trisk)_(withhold|rr)_([0-9.]+)', name)
    code_match = re.fullmatch(r'(T0|Ttask|Trisk)_code', name)
    history = None
    q = None
    if mixture:
        family, kind, rho_text = mixture.groups()
        rho = float(rho_text)
        if not 0 <= rho <= 1:
            raise ValueError('Mixture probability must lie in [0,1]')
        if mechanism is None or mechanism['policy'] != 'U' or mechanism['budget'] is not None:
            raise ValueError('Mixture controls require the unconstrained family map')
        q = _channel(mechanism, encoder, family, actions)
    elif name in HISTORICAL:
        root = Path(inputs_root) if inputs_root is not None else OUT/'private'/'inputs'
        file = root/'results'/'pcrl_invariant_baselines_v1'/f"seed_{ctx['anchor']}"/'releases'/name/'releases.npz'
        with np.load(file, allow_pickle=False) as archive:
            history = {pool: archive[f'wire/A/{pool}'].copy() for pool in ctx['pools']}
    elif mechanism is not None and name == mechanism.get('configuration'):
        family = mechanism['input']
        q = _channel(mechanism, encoder, family, actions)
    elif name not in ('H', 'J', 'continuous_task', 'constant_best', 'independent_token') and not code_match:
        if erasers is None or name not in erasers or not hasattr(erasers[name], 'transform'):
            raise ValueError(f'Unknown or unavailable release: {name}')
    result = {}
    for pool, data in ctx['pools'].items():
        enc = encoded[pool]
        h = np.asarray(data['ha'])
        n = len(h)
        g = _predictions(enc['actions'][actions], n)
        b = _predictions(np.asarray(enc['b'])[:, None], n)
        arm = {'aux': None, 'token_probs': np.ones((n, 1)),
               'global_offsets': _predictions(enc.get('global_offsets', g), n)}
        if q is not None:
            t = _codes(enc['codes'][family], len(q), n)
            token = q[t]
            if mixture and kind == 'withhold':
                arm['token_probs'] = np.column_stack((rho*token, np.full(n, 1-rho)))
                arm['fixed_probabilities'] = np.column_stack((g, b))
            else:
                arm['token_probs'] = rho*token+(1-rho)/q.shape[1] if mixture else token
                arm['fixed_probabilities'] = g
        elif name == 'H':
            arm['fixed_probabilities'] = b
        elif name == 'J':
            arm['aux'] = np.asarray(data['J'])
        elif name == 'continuous_task':
            arm['aux'] = _predictions(np.asarray(enc['p'])[:, None], n)
            arm['fixed_probabilities'] = arm['aux']
        elif name == 'independent_token':
            arm['token_probs'] = np.full_like(g, 1/g.shape[1])
            arm['fixed_probabilities'] = g
        elif name == 'constant_best':
            if mechanism is None or 'constant_action' not in mechanism:
                raise ValueError('Constant action must be chosen on mechanism-fit costs')
            action = mechanism['constant_action']
            if not isinstance(action, (int, np.integer)) or not 0 <= action < g.shape[1]:
                raise ValueError('Invalid constant action index')
            arm['fixed_probabilities'] = g[:, [action]]
        elif code_match:
            family = code_match.group(1)
            nt = encoder.code.n_states(family)
            t = _codes(enc['codes'][family], nt, n)
            arm['token_probs'] = np.eye(nt)[t]
        elif history is not None:
            wire = np.asarray(history[pool])
            if wire.ndim != 2 or wire.shape[0] != n or wire.shape[1] < h.shape[1]:
                raise ValueError('Historical wire has invalid H/aux dimensions')
            old_h = wire[:, :h.shape[1]]
            if old_h.dtype != h.dtype or old_h.tobytes() != h.tobytes():
                raise ValueError('Historical H columns are not byte-identical')
            arm['aux'] = wire[:, h.shape[1]:]
        else:
            p = np.asarray(enc['p'], float)
            if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('Invalid teacher probabilities for supervised eraser')
            features = np.column_stack((data['x'], logit(np.clip(p, CLIP, 1-CLIP))))
            if features.shape != (n, 33):
                raise ValueError('Supervised eraser requires PCA32 and teacher logit')
            arm['aux'] = erasers[name].transform(features)
        arm['token_probs'] = _probabilities(arm['token_probs'], n)
        if arm['aux'] is not None:
            aux = np.asarray(arm['aux'])
            if aux.ndim != 2 or len(aux) != n or not np.isfinite(aux).all():
                raise ValueError('Invalid auxiliary feature array')
            arm['aux'] = aux
        if 'fixed_probabilities' in arm:
            fp = _predictions(arm['fixed_probabilities'], n)
            if fp.shape != arm['token_probs'].shape:
                raise ValueError('Fixed prediction and token shapes differ')
        result[pool] = arm
    return result
