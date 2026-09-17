"""Locked 2017 temporal transport of the frozen 14-interface ACS slate.

Every 2018 object is applied unchanged. Mode B fits only utility probes,
reference probes, priors and attackers on the 2017 fitting partitions and
selects on the 2017 validation partitions. The final partition is readable only
through ``load_final`` after ``TRANSPORT_LOCK.json`` verifies every hashed input.
"""
from __future__ import annotations
import copy, hashlib, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from experiments import acs_fixed_predictions_audits as old
from experiments import acs_fixed_predictions_training as train
from experiments import acs_spectral_audits as spec
from experiments.acs_preservation_audits import fit_extended_auditors, save_extended_audits
from experiments.acs_selective_teachers import load_teachers
from experiments.acs_transfer_data import FEATURES, CovariatePreprocessor, array_hash, sha_file, write_json
from experiments.acs_transfer_heads import fit_candidates, fit_prior, load_candidate, metrics, _hash_array
from experiments.acs_transfer_models import SourceEncoder
from experiments.run_acs_transfer import subset_indices

ROOT = Path(__file__).resolve().parents[1]
HIST_ROOT = Path('/Users/nathansamson/PCRL')
DEV = ROOT/'results/redesign_20260910_acs_residual_spectral_v1'
OUT = ROOT/'results/redesign_20260917_acs_spectral_transport_v1'
LOCAL = ROOT/'data/acs_spectral_transport'
FIXED = HIST_ROOT/'results/redesign_20260909_acs_fixed_predictions_v1'
TRANSFER = HIST_ROOT/'results/redesign_20260907_acs_transfer_v1'
PROTECTION = HIST_ROOT/'results/redesign_20260908_acs_protection_v1'
COALITION = HIST_ROOT/'results/redesign_20260908_acs_coalition_v1'
SELECTIVE = HIST_ROOT/'results/redesign_20260908_acs_selective_preservation_v1'
HIST = ('H', 'E', 'A0', 'L025', 'L20', 'J')
LEARNED = ('A0', 'L025', 'L20', 'J')
SPECTRAL = tuple('spectral_'+a for a in ('S0', 'M025', 'M1', 'L025', 'L1', 'C025', 'C1', 'L2'))
INTERFACES = HIST+SPECTRAL
TASKS = ('same_residence', 'commute_over20', 'income_binary', 'civilian_at_work', 'public_coverage')
TARGETS = old.TARGETS
AUTHORIZED = {'A': ('income_binary', 'civilian_at_work', 'same_residence'), 'B': ('public_coverage', 'commute_over20')}
SOURCE_TASKS = ('income_binary', 'civilian_at_work', 'public_coverage')
FIT_PARTITIONS = ('attacker_fit', 'attacker_validation', 'task_fit', 'task_validation')
FINAL = 'final_evaluation'
YEAR = 2017
BUDGETS = (120, 360)
MODE_B_SCOPES = {'common_fresh': ({'fresh'}, {'wire'}), 'fresh_expanded': ({'fresh'}, {'wire', 'derived'}),
                 'fresh_catchup': ({'fresh', 'catchup'}, {'wire', 'derived'}),
                 'transport_all': ({'fresh', 'catchup', 'frozen2018'}, {'wire', 'derived'})}


def read(path): return json.loads(Path(path).read_text())


def now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------- data access
def _frame(z):
    return pd.DataFrame(z['raw_features'], columns=list(FEATURES))


def _partition(name):
    with np.load(LOCAL/f'{YEAR}_{name}.npz', allow_pickle=False) as z:
        return {'frame': _frame(z), 'labels': {t: z[t].astype(np.int64) for t in TARGETS},
                'weights': z['PWGTP'].astype(float), 'serialno': z['SERIALNO'].astype(str),
                'sporder': z['SPORDER'].astype(str), 'raw_rows': z['raw_rows']}


def load_fit_partition(name):
    if name not in FIT_PARTITIONS: raise PermissionError('Only fitting/validation partitions are open before the lock')
    return _partition(name)


def lock_path(out=OUT): return Path(out)/'TRANSPORT_LOCK.json'


def verify_lock(out=OUT):
    """Abort unless every hashed pre-final input is unchanged."""
    path = lock_path(out)
    if not path.exists(): raise PermissionError('Final partition sealed: TRANSPORT_LOCK.json missing')
    lock = read(path)
    for name, digest in lock['files'].items():
        p = Path(name) if name.startswith('/') else ROOT/name
        if sha_file(p) != digest: raise PermissionError('Locked input changed: '+name)
    access = Path(out)/'FINAL_ACCESS_LOG.json'
    log = read(access) if access.exists() else []
    log.append({'utc': now(), 'lock_sha256': sha_file(path)})
    write_json(access, log)
    return lock


import os as _os
DRY_RUN_PARTITION = _os.environ.get('PCRL_TRANSPORT_DRY_RUN')  # code exercise only: a validation partition stands in


def load_final(out=OUT):
    if DRY_RUN_PARTITION is not None:
        if Path(out).resolve() == OUT.resolve() or DRY_RUN_PARTITION not in FIT_PARTITIONS:
            raise PermissionError('Dry runs must use a fitting/validation partition and a separate output directory')
        return _partition(DRY_RUN_PARTITION)
    import os
    if lock_path(out).exists() and os.environ.get('PCRL_TRANSPORT_LOCK_VERIFIED') == sha_file(lock_path(out)):
        lock = read(lock_path(out))  # the parent process verified every hashed input
    else:
        lock = verify_lock(out)
    if sha_file(LOCAL/f'{YEAR}_{FINAL}.npz') != lock['final_partition_sha256']: raise PermissionError('Final partition changed')
    return _partition(FINAL)


# ------------------------------------------------------------ frozen objects
class FrozenSeed:
    """All 2018-fitted objects for one seed; nothing here can be refitted."""
    def __init__(self, seed):
        self.seed = seed; d = TRANSFER/f'seed_{seed}'
        prep = read(d/'preprocessing.json'); self.pre = CovariatePreprocessor()
        self.pre.numeric, self.pre.categories, self.pre.feature_names = (prep[k] for k in ('numeric', 'categories', 'feature_names'))
        sel = read(d/'source_selection.json'); self.bank_keys = sel['bank_keys']
        self.encoder = SourceEncoder.load(d/'source'/f"encoder_{sel['selected_lr']}"/'selected.pt')
        self.tree = joblib.load(d/'source'/f"tree_{sel['selected_tree_leaves']}"/'source_tree_bank.joblib')
        self.pca = joblib.load(d/'release_maps.joblib')['pca']
        rec = {r['task']: r for r in read(FIXED/'ANCHOR_PARITY.json') if r['seed'] == seed}
        self.anchor_paths = {t: HIST_ROOT/rec[t]['path'] for t in SOURCE_TASKS}
        self.anchor_models = {t: load_candidate(p) for t, p in self.anchor_paths.items()}
        self.teacher = load_teachers(SELECTIVE/'static'/f'seed_{seed}'/'teachers')['maps']['E']
        ident = read(FIXED/'PREFIT_IDENTITY.json')[str(seed)]
        init = torch.load(HIST_ROOT/ident['initialization_path'], weights_only=True)['model_state']
        self.aux, self.observers = {}, {}
        for c in HIST:
            cp = torch.load(FIXED/f'seed_{seed}'/'training'/c/'final.pt', weights_only=True)
            if c in LEARNED:
                m = train.Auxiliary(ident['preprocessing'], init); m.load_state_dict(cp['model_state']); self.aux[c] = m.freeze()
            o, _ = train.make_observers(seed, c != 'H'); o.load_state_dict(cp['adversary_state'])
            self.observers[c] = o.eval().requires_grad_(False)
        self.spectral = joblib.load(DEV/f'seed_{seed}'/'maps.joblib')

    def object_files(self):
        d = TRANSFER/f'seed_{self.seed}'; sel = read(d/'source_selection.json'); fixed = FIXED/f'seed_{self.seed}'
        ident = read(FIXED/'PREFIT_IDENTITY.json')[str(self.seed)]
        files = [d/'preprocessing.json', d/'source_selection.json', d/'release_maps.joblib',
                 d/'source'/f"encoder_{sel['selected_lr']}"/'selected.pt',
                 d/'source'/f"tree_{sel['selected_tree_leaves']}"/'source_tree_bank.joblib',
                 FIXED/'ANCHOR_PARITY.json', FIXED/'PREFIT_IDENTITY.json', HIST_ROOT/ident['initialization_path'],
                 DEV/f'seed_{self.seed}'/'maps.joblib', *[fixed/'training'/c/'final.pt' for c in HIST]]
        for p in self.anchor_paths.values(): files += [q for q in sorted(p.iterdir()) if q.is_file()]
        files += sorted(q for q in (SELECTIVE/'static'/f'seed_{self.seed}'/'teachers').iterdir() if q.is_file())
        return files

    def base(self, frame):
        x = self.pre.transform(frame)
        pca = np.asarray(self.pca.transform(x), dtype=np.float32)
        anchors = {'A': np.column_stack([self.anchor_models[t].predict_proba(pca) for t in ('income_binary', 'civilian_at_work')]),
                   'B': self.anchor_models['public_coverage'].predict_proba(pca).copy()}
        p = self.encoder.probabilities(x); tree = self.tree.probabilities(x)
        banks = {'B_rich_bank': np.asarray(np.concatenate([p[k] for k in self.bank_keys], axis=1), dtype=np.float32),
                 'C_tree_bank': np.asarray(np.concatenate([tree[k] for k in self.bank_keys], axis=1), dtype=np.float32)}
        return pca, anchors, banks

    def interface(self, condition, pca, anchors):
        if condition in SPECTRAL:
            z = self.spectral.transform(pca, anchors['A'], condition)
            a = np.column_stack((anchors['A'], z)); b = anchors['B'].copy()
            return {'A': a, 'B': b, 'AB': np.column_stack((a, b))}, None
        model = self.aux.get(condition)
        if condition == 'H': aux = None
        elif condition == 'E': aux = self.teacher.apply(pca[:, :16])
        else:
            with torch.no_grad(): aux = model(model.standardize(pca))[0].numpy()
        return train.array_wires(aux, anchors, model)


def unseen_category_counts(pre, frame):
    """Feature-support audit through the frozen preprocessor's existing columns."""
    x = pre.transform(frame); names = pre.feature_names; out = {}
    for j, name in enumerate(names):
        if name.endswith('=unseen') or name.endswith('=missing') or name.endswith('_missing'):
            out[name] = int(x[:, j].sum())
    return out


# ------------------------------------------------------------------ releases
def build_releases(frozen, frame):
    pca, anchors, banks = frozen.base(frame)
    wires, derived = {}, {}
    for c in INTERFACES:
        w, d = frozen.interface(c, pca, anchors)
        assert np.array_equal(w['A'][:, :4], anchors['A']) and np.array_equal(w['B'], anchors['B'])
        assert np.array_equal(w['AB'][:, -2:], anchors['B']) and all(v.dtype == np.float64 for v in w.values())
        wires[c], derived[c] = w, d
    return {'pca': pca, 'anchors': anchors, 'banks': banks, 'wire': wires, 'derived': derived}


def save_releases(path, rel):
    arrays = {'pca': rel['pca'], 'anchor/A': rel['anchors']['A'], 'anchor/B': rel['anchors']['B'],
              **{'bank/'+k: v for k, v in rel['banks'].items()}}
    for c in INTERFACES:
        for v, x in rel['wire'][c].items(): arrays[f'wire/{c}/{v}'] = x
        if rel['derived'][c] is not None:
            for v, x in rel['derived'][c].items(): arrays[f'derived/{c}/{v}'] = x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_releases(path):
    with np.load(path) as z:
        rel = {'pca': z['pca'], 'anchors': {'A': z['anchor/A'], 'B': z['anchor/B']},
               'banks': {k: z['bank/'+k] for k in ('B_rich_bank', 'C_tree_bank')}, 'wire': {}, 'derived': {}}
        for c in INTERFACES:
            rel['wire'][c] = {v: z[f'wire/{c}/{v}'] for v in old.VIEWS}
            rel['derived'][c] = {v: z[f'derived/{c}/{v}'] for v in old.VIEWS} if f'derived/{c}/A' in z.files else None
    return rel


def replay_2018(frozen, seed):
    """Bitwise reproduction of every saved 2018 array by this transport code."""
    from experiments.acs_transfer_data import load_cohort
    cfg = read(TRANSFER/'config.json')
    frame, _ = load_cohort(HIST_ROOT/cfg['raw_path'], cfg['sample_cap'], cfg['sample_seed'])
    fixed = FIXED/f'seed_{seed}'; report = {'seed': seed, 'pools': {}}
    rows = dict(np.load(fixed/'split_rows.npz'))
    lookup = pd.Series(np.arange(len(frame)), index=frame._raw_row.to_numpy())
    pca_saved, anchors_saved = dict(np.load(fixed/'pca.npz')), dict(np.load(fixed/'anchors.npz'))
    banks_saved = {k: dict(np.load(TRANSFER/f'seed_{seed}'/f'release_{k}.npz')) for k in ('B_rich_bank', 'C_tree_bank')}
    hist_saved = {c: dict(np.load(fixed/'training'/c/'releases.npz')) for c in HIST}
    spec_saved = {c: dict(np.load(DEV/f'seed_{seed}'/'releases'/c/'releases.npz')) for c in SPECTRAL}
    for pool, raw in rows.items():
        rel = build_releases(frozen, frame.iloc[lookup.loc[raw].to_numpy()])
        checks = {'pca': np.array_equal(rel['pca'], pca_saved[pool]),
                  'anchor_A': np.array_equal(rel['anchors']['A'], anchors_saved[pool+'/A']),
                  'anchor_B': np.array_equal(rel['anchors']['B'], anchors_saved[pool+'/B'])}
        for k, saved in banks_saved.items():
            if pool in saved: checks[k] = np.array_equal(rel['banks'][k], saved[pool])
        for c in HIST:
            for v in old.VIEWS:
                checks[f'{c}/wire/{v}'] = np.array_equal(rel['wire'][c][v], hist_saved[c][f'wire/{v}/{pool}'])
                if c in LEARNED: checks[f'{c}/derived/{v}'] = np.array_equal(rel['derived'][c][v], hist_saved[c][f'derived/{v}/{pool}'])
        for c in SPECTRAL:
            for v in old.VIEWS: checks[f'{c}/wire/{v}'] = np.array_equal(rel['wire'][c][v], spec_saved[c][f'wire/{v}/{pool}'])
        report['pools'][pool] = {'rows': int(len(raw)), 'checks': len(checks), 'all_bitwise_equal': all(checks.values()),
                                 'failures': [k for k, ok in checks.items() if not ok]}
    report['all_bitwise_equal'] = all(p['all_bitwise_equal'] for p in report['pools'].values())
    return report


# --------------------------------------------------------- candidate records
def _candidate_record(base_dir, space, columns, origin, cid, view, target, budget, *, source_view=None,
                      source_cid=None, inherited=False, anchor=False, diagnostic=False, family=None):
    return {'candidate_id': cid, 'base_candidate_directory': str(Path(base_dir).resolve()),
            'base_metadata_sha256': sha_file(Path(base_dir)/'metadata.json'), 'space': space,
            'projection_columns': list(columns) if columns is not None else None, 'transport_origin': origin,
            'view': view, 'target': target, 'audit_budget': budget, 'source_view': source_view or view,
            'source_candidate_id': source_cid or cid, 'inherited_singleton': inherited, 'anchor_ancestor': anchor,
            'diagnostic_only': diagnostic, 'family': family}


class Loaded:
    """Deduplicated base-model loader and prediction cache."""
    def __init__(self): self.models, self.preds = {}, {}

    def base(self, path):
        if path not in self.models: self.models[path] = spec.load_base(path)
        return self.models[path]

    def predict(self, rec, bundle, tag):
        x = bundle[rec['space']]
        cols = rec['projection_columns']
        if cols is not None: x = x[:, cols]
        x = np.ascontiguousarray(x)
        key = (rec['base_candidate_directory'], tag, array_hash(x))
        if key not in self.preds: self.preds[key] = self.base(rec['base_candidate_directory']).predict_proba(x)
        return self.preds[key]


def scope_members(records, scope):
    origins, spaces = MODE_B_SCOPES[scope]
    return sorted(cid for cid, r in records.items() if not r['diagnostic_only']
                  and r['transport_origin'] in origins and r['space'] in spaces)


def select(records, scores):
    out = {}
    for scope in MODE_B_SCOPES:
        ids = scope_members(records, scope)
        if not ids: raise ValueError('Empty Mode B scope '+scope)
        out[scope] = min(ids, key=lambda c: (scores[c]['log_loss'], c))
    return out


def inherit(roles, widths):
    """AB inherits every A/B sensitive candidate on its coordinate block."""
    for target in ('SEX', 'RAC1P'):
        coalition = roles['AB/'+target]
        for view in ('A', 'B'):
            for cid, r in list(roles[view+'/'+target].items()):
                wa, wb = widths[r['space']]
                block = tuple(range(wa)) if view == 'A' else tuple(range(wa, wa+wb))
                cols = tuple(block[j] for j in r['projection_columns']) if r['projection_columns'] is not None else block
                new = f'inherited_{view}__{cid}'
                coalition[new] = {**copy.deepcopy(r), 'candidate_id': new, 'view': 'AB', 'projection_columns': list(cols),
                                  'source_view': view, 'source_candidate_id': cid, 'inherited_singleton': True}


def frozen_2018_records(seed, condition, budget):
    """Every non-diagnostic, non-routed 2018 candidate of this interface."""
    record = read(DEV/f'seed_{seed}'/condition/'audits'/'audit_selection.json')['candidates'][str(budget)]
    out = {}
    for role, cs in record.items():
        view, target = role.split('/'); out[role] = {}
        if view == 'B' and condition != 'H': continue
        for cid, m in cs.items():
            if m.get('diagnostic_only') or cid.startswith('anchor__') or m.get('inherited_singleton'): continue
            new = 'frozen2018__'+cid
            out[role][new] = _candidate_record(m['base_candidate_directory'], m['space'], m['projection_columns'], 'frozen2018',
                                               new, view, target, budget, family=m.get('family'))
    return out


# ------------------------------------------------------------ Mode B fitting
def _xy(rel, condition, space, view, part, labels, target, ix=None):
    source = rel[part]['wire' if space == 'wire' else 'derived'][condition][view]
    y = labels[part][target]
    if ix is None:
        valid = y >= 0; return source[valid], y[valid]
    return source[ix], y[ix]


def fit_unit(out, seed, condition, rel, labels, frozen):
    """One interface's Mode B utility probes, attackers and selections."""
    dest = Path(out)/f'seed_{seed}'/condition
    if (dest/'fit_complete.json').exists(): return read(dest/'fit_complete.json')
    tick = time.perf_counter(); dest.mkdir(parents=True, exist_ok=True)
    ti = {t: subset_indices(labels['task_fit'][t], 2048, 1230000+100*seed+j) for j, t in enumerate(TASKS)}
    ai = {t: subset_indices(labels['attacker_fit'][t], 4096, 1240000+100*seed+j) for j, t in enumerate(TARGETS)}
    h = Path(out)/f'seed_{seed}'/'H'
    counts = {'utility_fits': 0, 'five_candidate_roles': 0, 'kernel_feature_fits': 0, 'catchup_trajectories': 0}
    # utility
    utility = {'selection': {}, 'candidates': {}}
    for view, tasks in AUTHORIZED.items():
        for t in tasks:
            role = view+'/'+t
            if view == 'B' and condition != 'H':
                hu = read(h/'utility_selection.json'); utility['selection'][role] = hu['selection'][role]
                utility['candidates'][role] = hu['candidates'][role]; continue
            xf, yf = _xy(rel, condition, 'wire', view, 'task_fit', labels, t, ti[t])
            xv, yv = _xy(rel, condition, 'wire', view, 'task_validation', labels, t)
            path = dest/'fitted/utility'/view/t
            if not all((path/c/'metadata.json').exists() for c in ('logistic', 'mlp')):
                r = fit_candidates(xf, yf, xv, yv, 2, 1250000+100*seed+TASKS.index(t), budget={'mlp': {'epochs': 40}})
                for cid, cand in r['candidates'].items(): cand.save(path/cid)
                counts['utility_fits'] += 2
            cs = {}
            for cid in ('logistic', 'mlp'):
                cand = load_candidate(path/cid)
                assert cand.metadata['fit_hashes'] == {'x': _hash_array(xf), 'y': _hash_array(yf)}
                assert cand.metadata['validation_hashes'] == {'x': _hash_array(xv), 'y': _hash_array(yv)}
                cs[cid] = {'dir': str((path/cid).resolve()), 'validation_log_loss': cand.metadata['validation_scores']['log_loss']}
            utility['candidates'][role] = cs
            utility['selection'][role] = min(cs, key=lambda k: (cs[k]['validation_log_loss'], k))
    write_json(dest/'utility_selection.json', utility)
    # attackers
    spaces = ('wire', 'derived') if condition in LEARNED else ('wire',)
    records = {b: {v+'/'+t: {} for v in old.VIEWS for t in old.AUDIT_ROLES[v]} for b in BUDGETS}
    for view in old.VIEWS:
        for target in old.AUDIT_ROLES[view]:
            if view == 'B' and condition != 'H': continue
            role = view+'/'+target; k = old.CLASSES[target]
            for space in spaces:
                if view == 'B' and space == 'derived': continue
                xf, yf = _xy(rel, condition, space, view, 'attacker_fit', labels, target, ai[target])
                xv, yv = _xy(rel, condition, space, view, 'attacker_validation', labels, target)
                path = dest/'fitted'/space/view/target/'fresh'
                if not (path/'unit_complete.json').exists():
                    if path.exists(): path.rename(path.with_name('incomplete_'+str(time.time_ns())))
                    static = old.fit_static_auditors(xf, yf, xv, yv, k, old.role_seed(seed, view, target))
                    res = fit_extended_auditors(xf, yf, xv, yv, k, old.role_seed(seed, view, target), static_candidates=static, epochs=360, nested_epochs=120)
                    save_extended_audits(res, path)
                    write_json(path/'unit_complete.json', {'files': {str(p.relative_to(path)): sha_file(p) for p in path.rglob('*') if p.is_file()}})
                    counts['five_candidate_roles'] += 1
                for tag, b in (('nested120', 120), ('nested360', 360)):
                    for cid in old.FRESH:
                        base = load_candidate(path/tag/cid)
                        assert base.metadata['fit_hashes'] == {'x': _hash_array(xf), 'y': _hash_array(yf)}
                        assert base.metadata['validation_hashes'] == {'x': _hash_array(xv), 'y': _hash_array(yv)}
                        ident = space+'__'+cid
                        records[b][role][ident] = _candidate_record(path/tag/cid, space, None, 'fresh', ident, view, target, b, family=base.metadata['family'])
                kpath = dest/'fitted'/space/view/target/'kernel'
                done = (kpath/'complete.json').exists()
                kernels = spec.fit_kernel(xf, yf, xv, yv, k, 20263910+100*seed+10*old.VIEWS.index(view)+old.TARGETS.index(target), kpath)
                counts['kernel_feature_fits'] += 0 if done else 1
                for a in kernels:
                    ident = space+'__kernel__'+a
                    for b in BUDGETS: records[b][role][ident] = _candidate_record(kpath/a, space, None, 'fresh', ident, view, target, b, family='kernel_ridge')
            okey = view+'__'+target
            if condition in frozen.observers and okey in frozen.observers[condition]:
                xf, yf = _xy(rel, condition, 'wire', view, 'attacker_fit', labels, target, ai[target])
                xv, yv = _xy(rel, condition, 'wire', view, 'attacker_validation', labels, target)
                path = dest/'fitted/wire'/view/target/'saved_start'
                if not (path/'unit_complete.json').exists():
                    if path.exists(): path.rename(path.with_name('incomplete_'+str(time.time_ns())))
                    res = old.fit_role_catchup(frozen.observers[condition][okey], xf, yf, xv, yv, k,
                        old.role_seed(seed, view, target, catchup=True), epochs=360, nested_epochs=120,
                        inherited_exposure={'origin': '2018 saved observer of this interface', 'study': 'redesign_20260909_acs_fixed_predictions_v1'})
                    save_extended_audits(res, path)
                    write_json(path/'unit_complete.json', {'files': {str(p.relative_to(path)): sha_file(p) for p in path.rglob('*') if p.is_file()}})
                    counts['catchup_trajectories'] += 1
                for tag, b in (('nested120', 120), ('nested360', 360)):
                    records[b][role]['catchup'] = _candidate_record(path/tag/'catchup', 'wire', None, 'catchup', 'catchup', view, target, b, family='mlp')
                    records[b][role]['saved_adversary'] = _candidate_record(path/'saved', 'wire', None, 'catchup', 'saved_adversary', view, target, b, diagnostic=True, family='mlp')
    # frozen 2018 candidates, H ancestors and B aliases
    width_a = rel['attacker_fit']['wire'][condition]['A'].shape[1]
    hsel = read(h/'audit_selection.json') if condition != 'H' else None
    for b in BUDGETS:
        for role, cs in frozen_2018_records(seed, condition, b).items(): records[b][role].update(cs)
        if hsel is None: continue
        for role in records[b]:
            view, target = role.split('/')
            if view == 'B':
                records[b][role] = copy.deepcopy(hsel['candidates'][str(b)][role]); continue
            for cid, r in hsel['candidates'][str(b)][role].items():
                if r['inherited_singleton']: continue
                cols = spec.route_ancestor(view, width_a, tuple(r['projection_columns']) if r['projection_columns'] is not None else None)
                new = 'anchor__'+cid
                records[b][role][new] = {**copy.deepcopy(r), 'candidate_id': new, 'view': view, 'projection_columns': list(cols), 'anchor_ancestor': True}
    widths = {'wire': (width_a, 2)}
    if condition in LEARNED: widths['derived'] = (rel['attacker_fit']['derived'][condition]['A'].shape[1], 2)
    for b in BUDGETS:
        # B-view candidates of other interfaces are H's; drop B-view inheritance duplicates only by exact ID.
        roles = records[b]
        for role in ('AB/SEX', 'AB/RAC1P'):
            for cid in [c for c in roles[role] if c.startswith('inherited_')]: del roles[role][cid]
        inherit(roles, widths)
    # validation scoring and selection
    cache = Loaded(); scores = {b: {} for b in BUDGETS}; selection = {b: {} for b in BUDGETS}
    for b in BUDGETS:
        for role, cs in records[b].items():
            view, target = role.split('/'); y = labels['attacker_validation'][target]; valid = y >= 0
            bundle = {'wire': rel['attacker_validation']['wire'][condition][view][valid]}
            if rel['attacker_validation']['derived'][condition] is not None:
                bundle['derived'] = rel['attacker_validation']['derived'][condition][view][valid]
            scores[b][role] = {}
            for cid, r in cs.items():
                p = cache.predict(r, bundle, 'attacker_validation')
                scores[b][role][cid] = metrics(y[valid], p, old.CLASSES[target])
                if r['transport_origin'] == 'fresh' and not r['inherited_singleton'] and not r['anchor_ancestor'] and r['family'] != 'kernel_ridge':
                    saved = cache.base(r['base_candidate_directory']).metadata['validation_scores']
                    assert saved['log_loss'] == scores[b][role][cid]['log_loss'], (role, cid)
            selection[b][role] = select(cs, {c: s for c, s in scores[b][role].items()})
    write_json(dest/'audit_selection.json', {'seed': seed, 'condition': condition, 'candidates': {str(b): v for b, v in records.items()},
        'validation_scores': {str(b): v for b, v in scores.items()}, 'selections': {str(b): v for b, v in selection.items()},
        'final_partition_received': False, 'selection_rule': 'minimum unweighted 2017 attacker-validation log loss, then candidate ID'})
    result = {'utc': now(), 'seed': seed, 'condition': condition, 'runtime_seconds': time.perf_counter()-tick, 'counts': counts,
              'candidate_counts': {str(b): {role: len(cs) for role, cs in v.items()} for b, v in records.items()},
              'selection_sha256': sha_file(dest/'audit_selection.json'), 'utility_sha256': sha_file(dest/'utility_selection.json')}
    write_json(dest/'fit_complete.json', result)
    return result


def fit_reference(out, seed, rel, labels):
    """PCA32/bank reference probes and fitting priors (Mode B parents)."""
    dest = Path(out)/f'seed_{seed}'/'reference'
    if (dest/'fit_complete.json').exists(): return read(dest/'fit_complete.json')
    tick = time.perf_counter(); dest.mkdir(parents=True, exist_ok=True)
    ti = {t: subset_indices(labels['task_fit'][t], 2048, 1230000+100*seed+j) for j, t in enumerate(TASKS)}
    ai = {t: subset_indices(labels['attacker_fit'][t], 4096, 1240000+100*seed+j) for j, t in enumerate(TARGETS)}
    sources = {'E_pca': ('pca', TASKS), 'B_rich_bank': ('bank', ('same_residence',)), 'C_tree_bank': ('bank', ('same_residence',))}
    selection = {}
    for name, (kind, tasks) in sources.items():
        get = (lambda part: rel[part]['pca']) if kind == 'pca' else (lambda part, n=name: rel[part]['banks'][n])
        for t in tasks:
            yfull = labels['task_fit'][t]; yval = labels['task_validation'][t]; valid = yval >= 0
            xf, yf = get('task_fit')[ti[t]], yfull[ti[t]]; xv, yv = get('task_validation')[valid], yval[valid]
            path = dest/'fitted'/name/t
            if not all((path/c/'metadata.json').exists() for c in ('logistic', 'mlp')):
                r = fit_candidates(xf, yf, xv, yv, 2, 1250000+100*seed+TASKS.index(t), budget={'mlp': {'epochs': 40}})
                for cid, cand in r['candidates'].items(): cand.save(path/cid)
            cs = {cid: load_candidate(path/cid) for cid in ('logistic', 'mlp')}
            for c in cs.values(): assert c.metadata['fit_hashes'] == {'x': _hash_array(np.asarray(xf, np.float64)), 'y': _hash_array(yf)}
            selection[name+'/'+t] = {'selected': min(cs, key=lambda k: (cs[k].metadata['validation_scores']['log_loss'], k)),
                                     'dirs': {k: str((path/k).resolve()) for k in cs}}
    priors = {}
    for t in TARGETS:
        p = fit_prior(labels['attacker_fit'][t][ai[t]], old.CLASSES[t]); p.save(dest/'prior'/t)
        priors[t] = str((dest/'prior'/t).resolve())
    write_json(dest/'selection.json', {'reference_probes': selection, 'priors': priors})
    result = {'utc': now(), 'runtime_seconds': time.perf_counter()-tick, 'selection_sha256': sha_file(dest/'selection.json')}
    write_json(dest/'fit_complete.json', result)
    return result
