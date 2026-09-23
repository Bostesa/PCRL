"""Independent exact-token scoring and legal recipient views for 2018 development.

All arrays remain private. A prediction tensor has one row per original person,
not one independent observation per token. This module never fits or selects on
an assessment pool and cannot access the hidden code or channel row as features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FLOOR = 1e-9
ROLES = (
    'utility:A/same_residence',
    'attack:A/SEX', 'attack:A/RAC1P',
    'attack:AB/SEX', 'attack:AB/RAC1P',
)


def parse_role(role: str) -> tuple[str, str, str]:
    try:
        kind, rest = role.split(':')
        view, target = rest.split('/')
    except ValueError as exc:
        raise ValueError(f'Invalid role: {role}') from exc
    if role not in ROLES and role not in ('attack:B/SEX', 'attack:B/RAC1P'):
        raise ValueError(f'Undeclared role: {role}')
    return kind, view, target


def view_features(ha: np.ndarray, hb: np.ndarray | None, view: str) -> np.ndarray:
    """Return only H actually available to recipient A, B, or coalition AB."""
    a = np.asarray(ha, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
        raise ValueError('H_A must have four finite columns')
    if view == 'A':
        return a
    b = np.asarray(hb, dtype=np.float64)
    if b.shape != (len(a), 2) or not np.isfinite(b).all():
        raise ValueError('H_B must have two finite aligned columns')
    if view == 'B':
        return b
    if view == 'AB':
        return np.column_stack((a, b))
    raise ValueError('Unknown recipient view')


def routed_features(rows: dict, view: str, wire: str) -> np.ndarray:
    """Assemble the actual H and optional continuous auxiliary service wire."""
    if wire not in ('H', 'release') or (view == 'B' and wire != 'H'):
        raise ValueError('Undeclared recipient wire')
    base = view_features(rows['ha'], rows.get('hb'), view)
    if wire == 'H' or rows.get('aux') is None:
        return base
    aux = np.asarray(rows['aux'], dtype=np.float64)
    if aux.ndim != 2 or aux.shape[0] != len(base) or aux.shape[1] < 1 or not np.isfinite(aux).all():
        raise ValueError('Continuous released auxiliary coordinates misalign original people')
    return np.column_stack((base, aux))


def model_directory_hash(directory: str | Path) -> str:
    """Hash the saved candidate files before deserializing a locked predictor."""
    root = Path(directory)
    files = sorted(p for p in root.iterdir() if p.is_file())
    if not files:
        raise ValueError('Empty fitted model directory')
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode('utf-8'))
        digest.update(bytes.fromhex(_file_hash(path)))
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _role_input_hash(arrays: dict) -> str:
    """Pin exact filtered fitting/selection people without writing their rows."""
    digest = hashlib.sha256()
    digest.update(arrays['role'].encode())
    for name in ('h', 'token_probs', 'labels', 'weights'):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    for name in ('ids', 'households'):
        digest.update(name.encode())
        for value in arrays[name]:
            encoded = json.dumps([type(value).__name__, str(value)], separators=(',', ':')).encode()
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
    return digest.hexdigest()


def _artifact_inventory(root: Path) -> dict[str, str]:
    artifacts = {}
    for path in sorted(root.rglob('*')):
        if path.name == 'own_registry.json' and path.parent == root:
            continue
        if path.is_symlink():
            raise ValueError('Fitted slate contains a symlink')
        if path.is_file():
            artifacts[str(path.relative_to(root))] = _file_hash(path)
    return artifacts


def _reuse_role_slate(root: Path, expected: dict) -> dict:
    registry_path = root / 'own_registry.json'
    if not registry_path.is_file():
        raise FileExistsError('Partial audit slate retained; quarantine before retry')
    registry = json.loads(registry_path.read_text())
    for key, value in expected.items():
        if registry.get(key) != value:
            raise ValueError(f'Existing audit slate {key} differs; preserve original')
    if not registry.get('models') or registry.get('own_selection') not in registry['models']:
        raise ValueError('Existing audit slate has no selected fitted model')
    for model in registry['models'].values():
        relative = model.get('model_relative_directory')
        if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Existing audit model lacks safe restore-relative path')
        path = (root / relative).resolve()
        if not path.is_relative_to((root / 'models').resolve()):
            raise ValueError('Existing audit model lies outside owned output')
        if model_directory_hash(path) != model.get('model_sha256'):
            raise ValueError('Existing audit model hash differs; preserve original')
        model['model_directory'] = str(path)
    if _artifact_inventory(root) != registry.get('artifacts_sha256'):
        raise ValueError('Existing audit slate artifact inventory differs; preserve original')
    return registry


def validate_token_probs(token_probs: np.ndarray, *, n: int | None = None) -> np.ndarray:
    p = np.asarray(token_probs, dtype=np.float64)
    if (p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0
            or (n is not None and len(p) != n) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-8)):
        raise ValueError('Every person needs a finite normalized token law')
    return p


def expected_token_loss(predictions: np.ndarray, token_probs: np.ndarray,
                        labels: np.ndarray, *, floor: float = FLOOR) -> np.ndarray:
    """Return E_Z[-log predictor(label | H,Z)] for each original person."""
    p = validate_token_probs(token_probs)
    q = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(labels)
    if (q.ndim != 3 or q.shape[:2] != p.shape or q.shape[2] < 2
            or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()
            or not np.allclose(q.sum(axis=2), 1., atol=1e-8, rtol=0)):
        raise ValueError('Predictor must return normalized full-class token probabilities')
    if (y.shape != (len(p),) or y.dtype.kind not in 'biuf' or not np.isfinite(y).all()
            or not np.equal(y, np.floor(y)).all() or (y < 0).any() or (y >= q.shape[2]).any()):
        raise ValueError('Labels must index the full declared class schema')
    if not np.isfinite(floor) or not 0 < floor < 1:
        raise ValueError('Invalid probability floor')
    observed = q[np.arange(len(y))[:, None], np.arange(p.shape[1])[None, :], y.astype(int)[:, None]]
    return np.einsum('nz,nz->n', p, -np.log(np.maximum(observed, floor)), optimize=True)


def score_weightings(loss: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    l, w = np.asarray(loss, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if (l.ndim != 1 or w.shape != l.shape or len(l) == 0
            or not np.isfinite(l).all() or (l < 0).any()
            or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0):
        raise ValueError('Finite per-person losses and positive-total weights required')
    u = float(l.mean())
    pwgtp = float(np.dot(l, w / w.sum()))
    return {'U': u, 'PWGTP': pwgtp, 'balanced': .5 * (u + pwgtp)}


def select_validation(losses: dict[str, np.ndarray], weights: np.ndarray) -> dict:
    """Select only from a supplied inner validation loss bank, lexical tie break."""
    if not losses or any(not isinstance(cid, str) or not cid for cid in losses):
        raise ValueError('Named nonempty validation bank required')
    scores = {cid: score_weightings(loss, weights) for cid, loss in losses.items()}
    selected = min(scores, key=lambda cid: (scores[cid]['balanced'], cid))
    return {'selected': selected, 'scores': scores,
            'rule': 'min 0.5 U + 0.5 PWGTP exact expected token CE; lexical tie'}


def assert_household_disjoint(*pools: np.ndarray) -> None:
    seen: set[str] = set()
    for pool in pools:
        h = np.asarray(pool)
        if h.ndim != 1 or not len(h) or any(not str(x).strip() for x in h):
            raise ValueError('Nonempty household vectors required')
        here = set(h.astype(str))
        if here & seen:
            raise ValueError('Household split overlap')
        seen.update(here)


def score_prediction_bank(predictions: dict[str, np.ndarray], token_probs: np.ndarray,
                          labels: np.ndarray, weights: np.ndarray) -> tuple[dict, dict]:
    """Score a frozen bank without fitting, selecting, or changing its candidate set."""
    losses = {cid: expected_token_loss(q, token_probs, labels) for cid, q in predictions.items()}
    scores = {cid: score_weightings(loss, weights) for cid, loss in losses.items()}
    return losses, scores


def _smoke() -> dict:
    p = np.asarray([[.5, .5]])
    q = np.asarray([[[.9, .1], [.1, .9]]])
    actual = expected_token_loss(q, p, np.asarray([1]))[0]
    expected = -.5 * np.log(.1) - .5 * np.log(.9)
    if not np.isclose(actual, expected, atol=1e-12):
        raise AssertionError('Exact-token replay failed')
    if np.isclose(actual, -np.log(.5)):
        raise AssertionError('Averaged predictions replaced expected loss')
    return {'status': 'pass', 'n_original_people': 1, 'tokens': 2,
            'expected_token_loss': float(actual), 'loss_of_average_probability': float(-np.log(.5))}


def _synthetic_fit_release_audit() -> dict:
    """Public toy LP fit through the actual one-token API and exact scorer."""
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
    from .release import ChannelArtifact, OneReleaseSession
    from .solver import solve_p1
    cost = np.ones((32, 17))
    cost[:, 0] = 0.
    cost[0, 1] = 1.
    cost[0, 2:] = 2.
    cut_coefficient = np.zeros_like(cost)
    cut_coefficient[0, 1] = 1.
    fitted = solve_p1(cost, [{'id': 'toy-sensitive-loss-floor', 'coeff': cut_coefficient,
                              'rho': .5, 'delta': 0., 'floor': .5}])
    if not fitted['feasible']:
        raise AssertionError('Toy finite-channel fit failed')
    artifact = ChannelArtifact(fitted['Q'], '0'*64, 'public-synthetic-fit')
    class Code:
        def n_states(self, name):
            if name != 'T0':
                raise ValueError('Toy encoder has only T0')
            return 32
    class Encoder:
        code = Code()
        def encode(self, inputs):
            return {'codes': {'T0': np.zeros(len(inputs.h_a), dtype=int)}}
    ha = np.arange(16, dtype=np.float64).reshape(4, 4)
    inputs = RuntimeInputs(np.zeros((4, 32)), ha)
    session = OneReleaseSession(artifact, Encoder(), rng=np.random.default_rng(7))
    wire = session.emit(inputs, ['toy-0', 'toy-1', 'toy-2', 'toy-3'])
    if wire['h_a'].tobytes() != ha.tobytes():
        raise AssertionError('Toy H_A service parity failed')
    token_law = artifact.expected_token_law(Encoder(), inputs)
    predictions = np.full((4, 17, 2), .5)
    predictions[:, 0] = (.7, .3)
    predictions[:, 1] = (.3, .7)
    y = np.array([0, 1, 0, 1])
    score = score_weightings(expected_token_loss(predictions, token_law, y), np.ones(4))
    return {'status': 'pass', 'scope': 'public synthetic data only',
            'fitted_channel_shape': list(artifact.Q.shape),
            'fixed_bank_objective': fitted['objective'],
            'fixed_bank_dual_lower_bound': fitted['dual_lower_bound'],
            'sampled_tokens': wire['token'].tolist(), 'service_byte_parity': True,
            'n_original_people': 4, 'independent_exact_task_loss_U': score['U']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'synthetic'))
    args = parser.parse_args()
    if args.command == 'smoke':
        print(json.dumps(_smoke(), sort_keys=True))
    elif args.command == 'synthetic':
        print(json.dumps(_synthetic_fit_release_audit(), sort_keys=True))



def role_arrays(rows: dict, token_probs: np.ndarray, role: str, *, require_full_fit_support: bool = False):
    """Validated complete-case arrays for one declared role and original people."""
    kind, view, target = parse_role(role)
    labels = np.asarray(rows['labels'][target])
    n_classes = 9 if target == 'RAC1P' else 2
    if (labels.ndim != 1 or labels.dtype.kind not in 'biuf'
            or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()
            or (labels < -1).any() or (labels >= n_classes).any()):
        raise ValueError('Original fixed class schema or missing-label code changed')
    n = len(labels)
    p = validate_token_probs(token_probs, n=n)
    wire = 'H' if view == 'B' or (p.shape[1] == 1 and rows.get('aux') is None) else 'release'
    h = routed_features(rows, view, wire)
    if view == 'B':
        if p.shape[1] != 1:
            raise ValueError('B-only service cannot receive the new A token')
    weights = np.asarray(rows['weights'], dtype=np.float64)
    ids = np.asarray(rows['ids'])
    households = np.asarray(rows['households']).astype(str)
    if (h.shape[0] != n or weights.shape != (n,) or ids.shape != (n,)
            or households.shape != (n,) or len(np.unique(ids)) != n
            or not np.isfinite(weights).all() or (weights < 0).any()
            or any(not value.strip() for value in households)):
        raise ValueError('Private role arrays misalign original people')
    mask = labels >= 0
    if not mask.any() or weights[mask].sum() <= 0:
        raise ValueError('No positive-weight observed labels for role')
    counts = np.bincount(labels[mask].astype(int), minlength=n_classes)
    missing = np.flatnonzero(counts == 0).tolist()
    if require_full_fit_support and missing:
        raise ValueError(f'Missing fitting support for declared classes: {missing}')
    return {'h': h[mask], 'token_probs': p[mask], 'labels': labels[mask].astype(int),
            'weights': weights[mask], 'ids': ids[mask], 'households': households[mask],
            'class_counts': counts, 'missing_classes': missing, 'n_classes': n_classes,
            'role': role, 'kind': kind, 'view': view, 'target': target}


def fit_role_slate(fit_rows: dict, fit_token_probs: np.ndarray,
                   validation_rows: dict, validation_token_probs: np.ndarray,
                   role: str, output_dir: str | Path, seed: int, *, release_id: str,
                   slate: str = 'catchup') -> dict:
    """Fit inherited equal-budget audit recipes on allowed inner pools only.

    Returns a private registry; the caller must merge legal ancestors and lock a
    final selected route before any outer assessment. No assessment path enters
    this function. The declared class schema remains full even if a historical
    fitting pool has zero examples of a class; this gap is recorded explicitly.
    """
    from experiments.pcrl_task_directed_release_v1.audits import fit_slate
    if not isinstance(release_id, str) or not release_id:
        raise ValueError('Named release identity is required')
    if slate not in ('standard', 'catchup'):
        raise ValueError('Bank/audit slate must be standard or catchup')
    fit = role_arrays(fit_rows, fit_token_probs, role)
    validation = role_arrays(validation_rows, validation_token_probs, role)
    assert_household_disjoint(fit['households'], validation['households'])
    if fit['n_classes'] != validation['n_classes'] or fit['h'].shape[1] != validation['h'].shape[1]:
        raise ValueError('Fitting/validation schema changed')
    root = Path(output_dir)
    expected = {'schema': 1, 'role': role, 'release_id': release_id, 'seed': int(seed),
                'slate': slate, 'fit_input_sha256': _role_input_hash(fit),
                'validation_input_sha256': _role_input_hash(validation)}
    if root.exists() and any(root.iterdir()):
        return _reuse_role_slate(root, expected)
    root.mkdir(parents=True, exist_ok=True)
    result = fit_slate(fit['h'], fit['token_probs'], fit['labels'], fit['weights'],
                       validation['h'], validation['token_probs'], validation['labels'], validation['weights'],
                       fit['n_classes'], seed, root / 'models', slate=slate)
    registry = {
        **expected, 'fit_rows': len(fit['labels']),
        'validation_rows': len(validation['labels']),
        'fit_class_counts': fit['class_counts'].tolist(),
        'fit_missing_classes': fit['missing_classes'],
        'validation_class_counts': validation['class_counts'].tolist(),
        'validation_missing_classes': validation['missing_classes'],
        'class_support_policy': 'full declared schema retained; unsupported fitting classes receive inherited predictor floor, never relabeled or supplied from assessment',
        'own_selection': result['selection'],
        'own_scores': result['metadata'].get('validation_scores', {}),
        'models': {cid: {'kind': 'model', 'model_directory': str(Path(candidate.metadata['directory']).resolve()),
                         'model_relative_directory': str(Path(candidate.metadata['directory']).resolve().relative_to(root.resolve())),
                         'model_sha256': model_directory_hash(candidate.metadata['directory']),
                         'source_view': fit['view'], 'wire': 'H' if fit['view'] == 'B' or (fit['token_probs'].shape[1] == 1 and fit_rows.get('aux') is None) else 'release',
                         'target': fit['target'], 'n_classes': fit['n_classes'],
                         'source_release_id': release_id}
                   for cid, candidate in result['candidates'].items()},
        'selection_note': 'own-slate provisional; add H-only and legal A/B ancestors before assessment lock',
    }
    registry['artifacts_sha256'] = _artifact_inventory(root)
    temporary = root / f'own_registry.json.tmp.{os.getpid()}'
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, root / 'own_registry.json')
    return registry


def predict_model_route(model_directory: str | Path, rows: dict, token_probs: np.ndarray,
                        *, source_view: str, wire: str,
                        expected_model_sha256: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Replay a frozen predictor on only the H/token wire it was trained to read."""
    from experiments.pcrl_task_directed_release_v1.audits import load_candidate
    if source_view not in ('A', 'B', 'AB') or wire not in ('H', 'release'):
        raise ValueError('Undeclared predictor route')
    if source_view == 'B' and wire != 'H':
        raise ValueError('B has no candidate token')
    h = routed_features(rows, source_view, wire)
    p = np.ones((len(h), 1), dtype=np.float64) if wire == 'H' else validate_token_probs(token_probs, n=len(h))
    if expected_model_sha256 is not None and model_directory_hash(model_directory) != expected_model_sha256:
        raise ValueError('Frozen predictor directory hash mismatch')
    candidate = load_candidate(model_directory)
    if candidate.n_tokens != p.shape[1]:
        raise ValueError('Token alphabet differs from fitted predictor; adapter/refit required')
    if len(candidate.mean) != h.shape[1]:
        raise ValueError('Service view width differs from fitted predictor')
    q = candidate.predict_token_proba(h, p.shape[1])
    # Check normalized probabilities before returning to any scoring caller.
    if (q.shape != (len(h), p.shape[1], candidate.n_classes)
            or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()
            or not np.allclose(q.sum(2), 1., atol=1e-8, rtol=0)):
        raise ValueError('Corrupt frozen predictor output')
    return q, p


def _legal_route(target_role: str, route: dict, *, release_id: str) -> None:
    kind, target_view, target = parse_role(target_role)
    if route.get('target') != target:
        raise ValueError('Predictor target/class schema differs from scored role')
    if route.get('wire') == 'release' and route.get('source_release_id') != release_id:
        raise ValueError('Token semantics differ across releases; refit or explicit adapter required')
    source_view = route['source_view']
    wire = route['wire']
    if target_view == 'A' and source_view != 'A':
        raise ValueError('A predictor cannot receive B/coalition inputs')
    if target_view == 'B' and (source_view != 'B' or wire != 'H'):
        raise ValueError('B-only route cannot receive A token or A H')
    if target_view == 'AB' and source_view not in ('A', 'B', 'AB'):
        raise ValueError('Illegal coalition ancestor')
    if source_view == 'B' and wire != 'H':
        raise ValueError('B has no new token')
    route_kind = route.get('kind', 'model')
    if route_kind == 'fixed_decoder':
        if (kind != 'utility' or target_view != 'A' or target != 'same_residence'
                or source_view != 'A' or wire != 'release'):
            raise ValueError('Fixed residence decoder is only an A release route')
    elif route_kind == 'model':
        pinned = route.get('model_sha256')
        if not isinstance(pinned, str) or len(pinned) != 64 or any(c not in '0123456789abcdef' for c in pinned):
            raise ValueError('Frozen model route requires a SHA-256 pin')
    else:
        raise ValueError('Undeclared frozen predictor route kind')


def predict_frozen_route(route: dict, rows: dict,
                         token_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replay a model or released fixed A task decoder on its exact wire."""
    if route.get('kind') == 'fixed_decoder':
        p = validate_token_probs(token_probs, n=len(rows['ha']))
        positive = np.asarray(rows['fixed_probabilities'], dtype=np.float64)
        if (positive.shape != p.shape or not np.isfinite(positive).all()
                or (positive < 0).any() or (positive > 1).any()):
            raise ValueError('Fixed decoder probabilities misalign released token law')
        return np.stack((1.-positive, positive), axis=2), p
    if route.get('kind') != 'model':
        raise ValueError('Undeclared frozen predictor route kind')
    return predict_model_route(route['model_directory'], rows, token_probs,
                               source_view=route['source_view'], wire=route['wire'],
                               expected_model_sha256=route['model_sha256'])


def build_role_route_bank(role: str, release_id: str, own: dict, h_only: dict,
                          *, a_same: dict | None = None, b_h_only: dict | None = None,
                          include_fixed_decoder: bool = False) -> dict[str, dict]:
    """Assemble equal-slate own and legal ancestors before validation selection.

    The caller supplies immutable fitted registries. AB requires both an A
    same-release token slate and a B H-only slate. Every non-H release requires
    the same-role H-only slate. J is a comparator, never an appended ancestor.
    """
    _, view, target = parse_role(role)
    if not isinstance(release_id, str) or not release_id:
        raise ValueError('Named release identity required')
    if own.get('role') != role or own.get('release_id') != release_id:
        raise ValueError('Own registry role or release differs')
    if h_only.get('role') != role or h_only.get('release_id') != 'H':
        raise ValueError('H-only ancestor role differs')
    if view == 'AB' and (a_same is None or b_h_only is None):
        raise ValueError('Complete coalition ancestors require A same-release and B H-only registries')
    grouped = [('own', own), ('H', h_only)]
    if view == 'AB':
        if a_same.get('role') != f'attack:A/{target}' or a_same.get('release_id') != release_id:
            raise ValueError('A coalition ancestor role or release differs')
        if b_h_only.get('role') != f'attack:B/{target}' or b_h_only.get('release_id') != 'H':
            raise ValueError('B coalition ancestor role differs')
        grouped += [('A', a_same), ('B', b_h_only)]
    slates = {registry.get('slate') for _, registry in grouped}
    if len(slates) != 1 or None in slates:
        raise ValueError('Own and ancestor predictors require the same declared slate')
    routes = {}
    for prefix, registry in grouped:
        models = registry.get('models')
        if not isinstance(models, dict) or not models:
            raise ValueError('Empty fitted route slate')
        for cid, route in models.items():
            if prefix in ('H', 'B') and route.get('wire') != 'H':
                raise ValueError('H-only ancestor attempted to read release')
            _legal_route(role, route, release_id=release_id)
            routes[f'{prefix}/{cid}'] = dict(route)
    if include_fixed_decoder:
        route = {'kind': 'fixed_decoder', 'target': 'same_residence',
                 'source_view': 'A', 'wire': 'release', 'source_release_id': release_id}
        _legal_route(role, route, release_id=release_id)
        routes['fixed_decoder'] = route
    return routes


def select_frozen_routes(validation_rows: dict, token_probs: np.ndarray, role: str,
                         routes: dict[str, dict], *, release_id: str) -> dict:
    """Validation-only selection over complete compatible own/ancestor routes."""
    if not routes:
        raise ValueError('A complete frozen audit route bank is required')
    _, _, target = parse_role(role)
    labels = np.asarray(validation_rows['labels'][target])
    mask = labels >= 0
    if not mask.any():
        raise ValueError('No validation labels for role')
    subset = {k: (v if k == 'labels' else np.asarray(v)[mask]) for k, v in validation_rows.items()}
    subset['labels'] = {k: np.asarray(v)[mask] for k, v in validation_rows['labels'].items()}
    p = validate_token_probs(token_probs, n=len(labels))[mask]
    losses = {}
    for cid, route in routes.items():
        _legal_route(role, route, release_id=release_id)
        q, rp = predict_frozen_route(route, subset, p)
        losses[cid] = expected_token_loss(q, rp, subset['labels'][target])
    selected = select_validation(losses, subset['weights'])
    return {**selected, 'role': role, 'release_id': release_id, 'route': routes[selected['selected']],
            'candidate_count': len(routes), 'source': 'inner validation only'}


def score_frozen_route(rows: dict, token_probs: np.ndarray, lock: dict) -> dict:
    """Assessment scoring of an already selected route; no model fit or selection."""
    role = lock['role']
    route = lock['route']
    _legal_route(role, route, release_id=lock['release_id'])
    _, _, target = parse_role(role)
    labels = np.asarray(rows['labels'][target])
    mask = labels >= 0
    if not mask.any():
        raise ValueError('No scored labels for role')
    subset = {k: (v if k == 'labels' else np.asarray(v)[mask]) for k, v in rows.items()}
    subset['labels'] = {k: np.asarray(v)[mask] for k, v in rows['labels'].items()}
    p = validate_token_probs(token_probs, n=len(labels))[mask]
    q, rp = predict_frozen_route(route, subset, p)
    loss = expected_token_loss(q, rp, subset['labels'][target])
    return {'ids': np.asarray(subset['ids']), 'households': np.asarray(subset['households']).astype(str),
            'weights': np.asarray(subset['weights'], dtype=np.float64), 'loss': loss,
            'scores': score_weightings(loss, subset['weights']),
            'selected_candidate': lock['selected'], 'role': role}


def score_fixed_binary_decoder(positive: np.ndarray, token_probs: np.ndarray,
                               labels: np.ndarray, weights: np.ndarray) -> dict:
    """Exact frozen/common binary decoder score, separate from fresh-probe selection."""
    p = validate_token_probs(token_probs)
    positive = np.asarray(positive, dtype=np.float64)
    if (positive.shape != p.shape or not np.isfinite(positive).all()
            or (positive < 0).any() or (positive > 1).any()):
        raise ValueError('Binary decoder must give one valid prediction per person/token')
    q = np.stack((1. - positive, positive), axis=-1)
    loss = expected_token_loss(q, p, labels)
    return {'loss': loss, 'scores': score_weightings(loss, weights)}

if __name__ == '__main__':
    main()
