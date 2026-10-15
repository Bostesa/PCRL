"""Confirmation runner for the FROZEN-RELEASE estimand (L_TREE packages and frozen comparators).

Modes (``--mode``):
  dry-run               validate the protocol and every binding hash; print what is frozen vs refitted; load no data.
  synthetic-rehearsal   run the full pipeline on synthetic fixtures (synthetic features/labels/SEX/references) with
                        the REAL frozen packages; exercises locks, role separation, reader freeze, inference, report.
  development-rehearsal run on ALREADY-USED development roles (attacker = ATT, assessment = INNER) through the shared
                        queue; must reproduce the frozen original clause statistics. Not evidence.
  execute-fresh         HARD-DISABLED in this build. A fresh population requires a separately authorized,
                        reviewed protocol (status APPROVED_FOR_FRESH_EXECUTION), an authorization file whose sha256
                        is listed in that protocol, verified unexposed role manifests and runnable U0/Ucal reference
                        packages. This command always refuses here.

Pipeline (development/synthetic): (1) protocol + package/comparator hash locks; (2) schema lock (83 ordered
names) and role separation (attacker vs assessment rows and duplicate groups disjoint); (3) releases from the
frozen services only (probabilities + decision); (4) fresh readers with the frozen procedure on attacker groups,
winners chosen on attacker validation, READER_FREEZE written BEFORE any assessment label is read; (5) assessment:
utility clause statistics vs U0/Ucal, coalition/recipient AUCs of ATT-validation winners; (6) inference: group-level
influence-function SEs; IUT one-sided alpha = 0.025 per clause for the ONE fixed global claim (primary) and two-sided
Bonferroni simultaneous bounds (sensitivity); (7) write-once report directory with a SHA-256 manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

FRESH_DISABLED_MESSAGE = ('REFUSED: execute-fresh is hard-disabled in this build. Required before any fresh run: (a) a reviewed '
                          'protocol with status APPROVED_FOR_FRESH_EXECUTION, (b) an owner authorization file whose sha256 '
                          'is listed in it, (c) verified unexposed attacker and assessment role manifests (no overlap with any '
                          'used role), (d) runnable, hash-bound U0/Ucal reference packages, (e) a new pushed lock receipt.')
UTILITY_CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
TASKS = ('income', 'occupation')
VIEWS = ('income', 'occupation', 'pair')


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Immutable:
    """Write-once report directory; every file is hashed into MANIFEST.json at close."""

    def __init__(self, path):
        self.root = Path(path)
        if self.root.exists():
            raise PermissionError('report directory exists (immutable reporting)')
        self.root.mkdir(parents=True, mode=0o700); self.files = {}

    def write_json(self, name, obj):
        p = self.root / name
        with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=float)
        self.files[name] = _sha(p); return p

    def close(self):
        p = self.root / 'MANIFEST.json'
        with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400), 'w') as f:
            json.dump(self.files, f, indent=2, sort_keys=True)
        for name in self.files:
            os.chmod(self.root / name, 0o400)
        return _sha(p)


# ---------------------------------------------------------------- frozen services (releases only)
def tree_package_release(binding, X, names):
    from task_headroom import deploy
    m, teachers, theta = deploy.load_package(binding['package_dir'], binding['manifest_sha256'])
    o = deploy.predict(m, teachers, theta, X, names)
    return [{'probability': o['p_income'], 'decision': o['decision_income']}, {'probability': o['p_occupation'], 'decision': o['decision_occupation']}]


def raw_proj_release(binding, X, names, seed):
    from task_conditioned_representation.models import load_checkpoint
    from .rawproj.wires import prepare_geometry
    from .rawproj.services import service_forward
    g = binding['geometry']
    if _sha(g['path']) != g['sha256'] or _sha(binding['checkpoint']['path']) != binding['checkpoint']['sha256'] or _sha(binding['adapter_binding']['path']) != binding['adapter_binding']['sha256']:
        raise PermissionError('RAW_PROJ binding hash differs')
    geometry = prepare_geometry(json.loads(Path(g['path']).read_text()), expected_sha256=g['binding_sha256'], source_seed=seed)
    model, _ = load_checkpoint(binding['checkpoint']['path'], binding['checkpoint']['sha256'])
    ab = json.loads(Path(binding['adapter_binding']['path']).read_text())
    w = service_forward({'kind': 'RAW_PROJ'}, X, names, model=model, geometry=geometry, verify=lambda p, s: Path(p), seed=seed,
                        source_checkpoint_sha256=binding['checkpoint']['sha256'], adapter_binding_sha256=binding['adapter_binding']['sha256'], binding_scales=ab['scales'])
    return [{'probability': np.asarray(w[i]['probability'], dtype=np.float64), 'decision': np.asarray(w[i]['decision'], dtype=np.int64)} for i in range(2)]


def qpc_policy_release(binding, X, names, workdir, seed):
    """Historical code comparator run through the frozen hcal.deploy CLI (U teacher + registered policy, mean decoder)."""
    import subprocess
    for k in ('policy', 'schema', 'teacher_model'):
        if _sha(binding[k]['path']) != binding[k]['sha256']:
            raise PermissionError(f'code comparator binding hash differs: {k}')
    inp = Path(workdir) / f'qpc_input_{seed}_{os.getpid()}.npz'; out = Path(workdir) / f'qpc_out_{seed}_{os.getpid()}.npz'
    np.savez(inp, X=np.asarray(X, dtype=np.float32), feature_names=np.asarray(names))
    env = dict(os.environ, OMP_NUM_THREADS='1')
    r = subprocess.run([sys.executable, '-m', 'hcal.deploy', '--unit', binding['teacher_unit'], '--policy', binding['policy']['path'], '--X', str(inp),
                        '--schema', binding['schema']['path'], '--out', str(out), '--seed', str(binding['teacher_seed'])], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError('hcal.deploy refused or failed: ' + r.stderr[-500:])
    with np.load(out) as z:
        res = [{'probability': z['probs_1'].astype(np.float64), 'decision': z['decision_1'].astype(np.int64)},
               {'probability': z['probs_2'].astype(np.float64), 'decision': z['decision_2'].astype(np.int64)}]
    inp.unlink(); out.unlink()
    return res


def release(service, X, names, seed, workdir):
    kind = service['kind']
    if kind == 'tree_package':
        return tree_package_release(service['per_seed'][str(seed)], X, names)
    if kind == 'raw_proj':
        return raw_proj_release(service['per_seed'][str(seed)], X, names, seed)
    if kind == 'qpc_policy':
        return qpc_policy_release(service['per_seed'][str(seed)], X, names, workdir, seed)
    raise ValueError('unregistered comparator kind')


# ---------------------------------------------------------------- readers (frozen procedure)
def fit_readers(att_X_by_service, att_sex, att_unit, adult_split, seed):
    import warnings
    from task_headroom.reader_worker import SLATE, tables
    from task_conditioned_representation.attacks import MODEL_SEED_BASE, attack_split, fit_reader, predict_reader, orientation, orient
    from task_conditioned_representation.inference import auc
    train, val = attack_split(att_unit, require_adult=adult_split)
    frozen = {}
    for label, w in att_X_by_service.items():
        t = tables(w); single = {}; rec = {'validation': {}, 'winner': {}, 'models': {}}
        for view in VIEWS:
            cands = {}
            for name, kind in SLATE:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = fit_reader(t[view][kind][train], att_sex[train], name, MODEL_SEED_BASE + seed)
                raw = predict_reader(m, t[view][kind][val]); sign = orientation(att_sex[val], raw)
                a = auc(att_sex[val], orient(raw, sign)); cands[name] = a
                rec['models'][(view, name)] = (m, sign, kind)
                if view != 'pair':
                    single[f'IGNORE_{"occupation" if view == "income" else "income"}__{name}'] = (a, view, name)
            if view == 'pair':
                for n, (a, v, nm) in single.items():
                    cands[n] = a
            order = list(cands); win = max(order, key=lambda n: (cands[n], -order.index(n)))
            rec['winner'][view] = win; rec['validation'][view] = cands
        frozen[label] = rec
    return frozen


def score_readers(frozen, X_by_service, seed):
    from task_headroom.reader_worker import tables
    from task_conditioned_representation.attacks import predict_reader, orient
    out = {}
    for label, rec in frozen.items():
        t = tables(X_by_service[label]); out[label] = {}
        for view in VIEWS:
            win = rec['winner'][view]
            if win.startswith('IGNORE_'):
                v, name = ('income' if win.split('__')[0] == 'IGNORE_occupation' else 'occupation'), win.split('__')[1]
            else:
                v, name = view, win
            m, sign, kind = rec['models'][(v, name)]
            out[label][view] = orient(predict_reader(m, t[v][kind]), sign)
    return out


# ---------------------------------------------------------------- inference
def auc_if(y, s):
    from tree_completion.feasibility_worker import auc_if as f
    return f(y, s)


def group_phi(rows_if, unit):
    groups, inv = np.unique(unit, return_inverse=True); G = len(groups)
    psi = np.zeros((rows_if.shape[0], G)); np.add.at(psi.T, inv, rows_if.T)
    return psi * (G / rows_if.shape[1]), G


def clauses_for(claim, comparators):
    req = comparators['required'] if claim == 'A_original' else ['RAW_PROJ']
    sup = [] if claim == 'A_original' else [comparators['strongest_code']]
    return req, sup


def infer(stats_points, phi, specs, G, alpha=0.025, family_K=None):
    from scipy import stats
    cov = np.cov(phi, ddof=1)
    rows = []
    q_iut = stats.t.ppf(1 - alpha, G - 1); K = family_K or len(specs); q_sim = stats.t.ppf(1 - 0.05 / (2 * K), G - 1)
    for name, a, direction, bound in specs:
        pt = float(a @ stats_points); se = float(np.sqrt(a @ cov @ a / G))
        iut = (pt + q_iut * se <= bound) if direction == +1 else (pt - q_iut * se > bound)
        sim = (pt + q_sim * se <= bound) if direction == +1 else (pt - q_sim * se > bound)
        rows.append({'clause': name, 'point': pt, 'se': se, 'bound': bound, 'direction': 'upper<=bound' if direction == 1 else 'lower>bound',
                     'iut_one_sided_bound': pt + q_iut * se if direction == 1 else pt - q_iut * se, 'iut_pass': bool(iut),
                     'sim_bound': pt + q_sim * se if direction == 1 else pt - q_sim * se, 'sim_pass': bool(sim)})
    return {'rows': rows, 'iut_global_claim_confirmed': all(r['iut_pass'] for r in rows), 'sim_all_pass': all(r['sim_pass'] for r in rows),
            'q_iut': float(q_iut), 'q_sim': float(q_sim), 'K_sim': K, 'G': G}


def assess(protocol, rel_assess, scores, y, unit, refs):
    """Clause statistics with per-group influence functions (both seeds share people)."""
    from task_headroom.data import task_vectors, K
    names, pts, rif = [], [], []
    for s in protocol['seeds']:
        P = rel_assess['CANDIDATE'][s]
        for i, t in enumerate(TASKS):
            a = task_vectors(y[t], P[i]['probability'], K[t])
            for ref in ('U0', 'Ucal'):
                r = task_vectors(y[t], refs[s][ref][i], K[t])
                for metric, d in (('accuracy_loss', r['accuracy'] - a['accuracy']), ('nll_excess', a['nll'] - r['nll']), ('brier_excess', a['brier'] - r['brier'])):
                    names.append(f'U|{s}|{t}|{ref}|{metric}'); pts.append(float(d.mean())); rif.append(d - d.mean())
    for label in scores:
        for s in protocol['seeds']:
            for v in VIEWS:
                A, IF = auc_if(y['sex'], scores[label][s][v]); names.append(f'AUC|{label}|{s}|{v}'); pts.append(A); rif.append(IF)
    phi, G = group_phi(np.vstack(rif), unit)
    idx = {n: i for i, n in enumerate(names)}; M = len(names); specs = []
    for n in names:
        if n.startswith('U|'):
            _, s, t, ref, metric = n.split('|'); a = np.zeros(M); a[idx[n]] = 1
            specs.append((f'utility|s{s}|{t}|{ref}|{metric}', a, +1, UTILITY_CAPS[metric]))
    nseeds = len(protocol['seeds'])
    def lin(c, cand, view, sign):
        a = np.zeros(M)
        for s in protocol['seeds']:
            a[idx[f'AUC|{c}|{s}|{view}']] += sign / nseeds; a[idx[f'AUC|{cand}|{s}|{view}']] -= sign / nseeds
        return a
    req, sup = clauses_for(protocol['claim'], protocol['comparators_roles'])
    for c in req:
        specs.append((f'benefit_vs_{c}', lin(c, 'CANDIDATE', 'pair', 1), -1, 0.020))
        for v in ('income', 'occupation'):
            specs.append((f'excess_{v}_vs_{c}', lin(c, 'CANDIDATE', v, -1), +1, 0.010))
    for c in sup:
        specs.append((f'superiority_vs_{c}', lin(c, 'CANDIDATE', 'pair', 1), -1, 0.0))
        for v in ('income', 'occupation'):
            specs.append((f'excess_{v}_vs_{c}', lin(c, 'CANDIDATE', v, -1), +1, 0.010))
    return names, np.array(pts), phi, G, specs


# ---------------------------------------------------------------- orchestration
def check_roles(att, ass):
    if tuple(map(str, att['feature_names'])) != tuple(map(str, ass['feature_names'])):
        raise PermissionError('schema differs between roles')
    if np.intersect1d(att['row_id'], ass['row_id']).size or np.intersect1d(att['unit'], ass['unit']).size:
        raise PermissionError('attacker and assessment roles overlap (rows or duplicate groups)')
    return {'attacker_rows': int(len(att['row_id'])), 'assessment_rows': int(len(ass['row_id'])), 'assessment_groups': int(len(np.unique(ass['unit']))), 'overlap': 0}


def run_pipeline(protocol, att, ass_features, load_assessment_labels, refs_loader, report_dir, workdir, adult_split):
    rep = Immutable(report_dir)
    names = att['feature_names']
    if hashlib.sha256('\n'.join(map(str, names)).encode()).hexdigest() != protocol['feature_names_sha256']:
        raise PermissionError('feature schema lock differs')
    roles = check_roles(att, ass_features)
    services = {'CANDIDATE': protocol['candidate'], **protocol['comparators']}
    rel_att, rel_ass = {}, {}
    for label, svc in services.items():
        rel_att[label] = {s: release(svc, att['X'], names, int(s), workdir) for s in protocol['seeds']}
        rel_ass[label] = {s: release(svc, ass_features['X'], names, int(s), workdir) for s in protocol['seeds']}
    frozen, scores = {}, {lab: {} for lab in services}
    for s in protocol['seeds']:
        fr = fit_readers({lab: rel_att[lab][s] for lab in services}, att['sex'], att['unit'], adult_split, int(s))
        frozen[s] = fr
        sc = score_readers(fr, {lab: rel_ass[lab][s] for lab in services}, int(s))
        for lab in services:
            scores[lab][s] = sc[lab]
    rep.write_json('READER_FREEZE.json', {s: {lab: {'winner': r['winner'], 'validation_auc': r['validation']} for lab, r in fr.items()} for s, fr in frozen.items()})
    # Only now: assessment labels.
    y, unit = load_assessment_labels()
    refs = refs_loader()
    names_s, pts, phi, G, specs = assess(protocol, rel_ass, scores, y, unit, refs)
    res = infer(pts, phi, specs, G, alpha=protocol['decision']['alpha_one_sided'], family_K=protocol['decision'].get('sim_family_K'))
    res.update({'roles': roles, 'claim': protocol['claim'], 'estimand': protocol['estimand'], 'mode_note': protocol.get('mode_note')})
    rep.write_json('CLAUSE_REPORT.json', res)
    manifest_sha = rep.close()
    return res, manifest_sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', required=True, choices=('dry-run', 'synthetic-rehearsal', 'development-rehearsal', 'execute-fresh'))
    ap.add_argument('--protocol', required=True); ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--spec', help='queue config for development-rehearsal')
    a = ap.parse_args(argv)
    if a.mode == 'execute-fresh':
        print(FRESH_DISABLED_MESSAGE); return 2
    if _sha(a.protocol) != a.protocol_sha256:
        print('REFUSED: protocol sha256 differs'); return 2
    protocol = json.loads(Path(a.protocol).read_text())
    if a.mode == 'dry-run':
        from .runner_rehearsal import validate_bindings
        validate_bindings(protocol)
        print(json.dumps({'status': 'DRY_RUN_OK', 'bindings_verified': True, 'estimand': protocol['estimand'], 'frozen': protocol['binding_summary']['frozen'],
                          'refitted': protocol['binding_summary']['refitted'], 'claim': protocol['claim'], 'decision': protocol['decision']}, indent=2))
        return 0
    from .runner_rehearsal import rehearse
    return rehearse(a.mode, protocol, a.spec)


if __name__ == '__main__':
    sys.exit(main())
