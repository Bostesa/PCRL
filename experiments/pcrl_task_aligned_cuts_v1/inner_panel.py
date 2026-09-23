"""Independent, private inner-development audit of a frozen 2018 release.

The audit fits on downstream_fit, selects predictors on the registered
inner_selection households, and scores only inner_pilot households. It does
not index outer outcomes; the archived monolithic joblib seal has the
transient-deserialization limitation recorded in VALIDATION.md.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np

from . import audit, data


def _slug(role: str) -> str:
    return role.replace(':', '_').replace('/', '_')


def _json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    os.replace(temporary, path)


def _registry_routes(role: str, release_id: str, own: dict, h_only: dict,
                     own_by_role: dict[str, dict], h_by_role: dict[str, dict]) -> dict:
    _, view, target = audit.parse_role(role)
    kwargs = {}
    if view == 'AB':
        kwargs = {'a_same': own_by_role[f'attack:A/{target}'],
                  'b_h_only': h_by_role[f'attack:B/{target}']}
    return audit.build_role_route_bank(role, release_id, own, h_only, **kwargs)


def _verified_shared_h_receipt(h_root: Path) -> dict:
    """Verify every reused H model byte against its completed source unit."""
    if h_root.name != 'H':
        raise ValueError('Shared H root must be the H directory of a completed audit unit')
    source_unit = h_root.parent
    receipt_path = source_unit / 'COMPLETE.json'
    if not receipt_path.is_file():
        raise ValueError('Shared H slate lacks an immutable completed source receipt')
    receipt = json.loads(receipt_path.read_text())
    pinned = {name: digest for name, digest in receipt.get('artifacts', {}).items()
              if name.startswith('H/')}
    actual = {str(path.relative_to(source_unit)): data.sha256_file(path)
              for path in sorted(h_root.rglob('*')) if path.is_file()}
    if not pinned or pinned != actual:
        raise ValueError('Shared H slate differs from completed source artifact inventory')
    return {'source_unit_id': receipt.get('unit_id'),
            'source_complete_sha256': data.sha256_file(receipt_path),
            'artifact_count': len(pinned)}


def run_inner_panel(prepared: dict, candidate_q: np.ndarray, split: dict,
                    release_id: str, output_dir: str | Path, *,
                    slate: str = 'standard',
                    shared_h_root: str | Path | None = None) -> dict:
    """Fit and replay a complete five-role independent inner pilot.

    ``split`` is the globally assigned household split from
    :func:`data.global_validation_split`; the candidate is frozen before this
    call.  The provisional pilot uses the registered inherited standard slate
    uniformly; final outer assessment uses the separately registered catchup
    slate.  The result contains aggregate scores; per-person contributions and
    fitted models are retained only in ``output_dir`` under private storage.
    """
    if slate not in ('standard', 'catchup'):
        raise ValueError('Undeclared audit slate')
    if not isinstance(release_id, str) or not release_id or release_id == 'H':
        raise ValueError('Named non-H release required')
    root = Path(output_dir)
    if 'private' not in root.parts:
        raise ValueError('Inner pilot fitted objects and person contributions must remain private')
    if (root / 'INNER_PANEL.json').exists():
        raise FileExistsError('Completed inner panel is immutable; use its recorded output')
    anchor = prepared['ctx']['anchor']
    if anchor not in (0, 1, 2) or anchor not in split.get('anchors', {}):
        raise ValueError('Registered global household split lacks this anchor')
    if split.get('assignment_sha256') != '9624c02a5dfc1797c602c0ab49b55f0853d2ca12a2a631ebbf2bf307253c27bf':
        raise ValueError('Global household assignment differs from committed protocol')
    q = np.asarray(candidate_q, dtype=np.float64)
    if (q.ndim != 2 or q.shape[0] != 32 or q.shape[1] < 2
            or not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(axis=1), 1., atol=1e-8, rtol=0)):
        raise ValueError('Release must be a valid frozen 32-state channel')
    q_hash = hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()
    one = np.ones((32, 1), dtype=np.float64)
    fit_m = data.token_pool(prepared, 'downstream_fit', q)
    fit_h = data.token_pool(prepared, 'downstream_fit', one)
    all_m = data.token_pool(prepared, 'downstream_validation', q)
    all_h = data.token_pool(prepared, 'downstream_validation', one)
    assignment = split['anchors'][anchor]['rows']
    select_m = data.subset_token_pool(all_m, assignment['inner_selection'])
    pilot_m = data.subset_token_pool(all_m, assignment['inner_pilot'])
    select_h = data.subset_token_pool(all_h, assignment['inner_selection'])
    pilot_h = data.subset_token_pool(all_h, assignment['inner_pilot'])
    audit.assert_household_disjoint(fit_m['households'], select_m['households'],
                                    pilot_m['households'])
    if (not np.array_equal(fit_m['ids'], fit_h['ids'])
            or not np.array_equal(select_m['ids'], select_h['ids'])
            or not np.array_equal(pilot_m['ids'], pilot_h['ids'])):
        raise ValueError('H and release audit rows differ')
    root.mkdir(parents=True, exist_ok=True)
    h_root = Path(shared_h_root) if shared_h_root is not None else root / 'H'
    if 'private' not in h_root.parts:
        raise ValueError('Shared H-only fitted objects must remain private')
    shared_h_receipt = (_verified_shared_h_receipt(h_root)
                        if shared_h_root is not None else None)

    # Fit the B-only ancestor solely on H_B; A and AB slates use exactly their
    # declared service columns and token.  The same seed is used for each role
    # across releases, and the same slate is used for own and ancestor routes.
    h_roles = (*audit.ROLES, 'attack:B/SEX', 'attack:B/RAC1P')
    h_by_role: dict[str, dict] = {}
    own_by_role: dict[str, dict] = {}
    for role_index, role in enumerate(h_roles):
        seed = 16000 + 1000 * anchor + role_index
        h_by_role[role] = audit.fit_role_slate(
            fit_h, fit_h['token_probs'], select_h, select_h['token_probs'],
            role, h_root / _slug(role), seed, release_id='H', slate=slate)
    h_registry_hashes = {
        role: data.sha256_file(h_root / _slug(role) / 'own_registry.json')
        for role in h_roles}
    for role_index, role in enumerate(audit.ROLES):
        seed = 16000 + 1000 * anchor + role_index
        own_by_role[role] = audit.fit_role_slate(
            fit_m, fit_m['token_probs'], select_m, select_m['token_probs'],
            role, root / 'candidate' / _slug(role), seed,
            release_id=release_id, slate=slate)

    roles = {}
    contributions = {}
    for role in audit.ROLES:
        candidate_routes = _registry_routes(role, release_id, own_by_role[role],
                                            h_by_role[role], own_by_role, h_by_role)
        h_routes = _registry_routes(role, 'H', h_by_role[role], h_by_role[role],
                                    h_by_role, h_by_role)
        candidate_lock = audit.select_frozen_routes(
            select_m, select_m['token_probs'], role, candidate_routes,
            release_id=release_id)
        h_lock = audit.select_frozen_routes(
            select_h, select_h['token_probs'], role, h_routes, release_id='H')
        candidate_score = audit.score_frozen_route(
            pilot_m, pilot_m['token_probs'], candidate_lock)
        h_score = audit.score_frozen_route(
            pilot_h, pilot_h['token_probs'], h_lock)
        if (not np.array_equal(candidate_score['ids'], h_score['ids'])
                or not np.array_equal(candidate_score['households'], h_score['households'])
                or not np.array_equal(candidate_score['weights'], h_score['weights'])):
            raise ValueError(f'H and candidate pilot pairs misalign for {role}')
        slug = _slug(role)
        contributions[f'{slug}_ids'] = candidate_score['ids'].astype(str)
        contributions[f'{slug}_households'] = candidate_score['households']
        contributions[f'{slug}_weights'] = candidate_score['weights']
        contributions[f'{slug}_candidate_loss'] = candidate_score['loss']
        contributions[f'{slug}_H_loss'] = h_score['loss']
        roles[role] = {
            'n_scored_people': len(candidate_score['loss']),
            'n_scored_households': len(set(candidate_score['households'])),
            'candidate': candidate_score['scores'], 'H': h_score['scores'],
            'H_minus_candidate': {weighting: h_score['scores'][weighting]
                                  - candidate_score['scores'][weighting]
                                  for weighting in ('U', 'PWGTP')},
            'candidate_selection': {key: candidate_lock[key]
                                    for key in ('selected', 'scores', 'candidate_count', 'rule')},
            'H_selection': {key: h_lock[key]
                            for key in ('selected', 'scores', 'candidate_count', 'rule')},
            'candidate_selected_model_sha256': candidate_lock['route']['model_sha256'],
            'H_selected_model_sha256': h_lock['route']['model_sha256'],
            'fit_missing_classes': own_by_role[role]['fit_missing_classes'],
            'selection_missing_classes': own_by_role[role]['validation_missing_classes'],
        }
    contribution_path = root / 'INNER_PILOT_CONTRIBUTIONS.npz'
    if contribution_path.exists():
        raise FileExistsError('Existing private pilot contributions retained')
    np.savez_compressed(contribution_path, **contributions)
    report = {
        'schema': 1, 'kind': 'independent_inner_development_pilot',
        'not_confirmation': True, 'anchor': anchor, 'release_id': release_id,
        'channel_sha256': q_hash, 'slate': slate,
        'H_slate_root': str(h_root.resolve()),
        'H_registry_sha256': h_registry_hashes,
        'shared_H_source_receipt': shared_h_receipt,
        'split_assignment_sha256': split['assignment_sha256'],
        'fit_pool': 'downstream_fit', 'selection_pool': 'downstream_validation:inner_selection',
        'score_pool': 'downstream_validation:inner_pilot',
        'outer_pool_opened': False, 'probability_floor': audit.FLOOR,
        'roles': roles,
        'contributions_relative_path': contribution_path.name,
        'contributions_sha256': data.sha256_file(contribution_path),
    }
    _json_atomic(root / 'INNER_PANEL.json', report)
    return report


def _under_root(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Sidecar path must be root-relative')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Sidecar path escapes study worktree')
    return resolved


def _sidecar_inputs(root: Path, registration: Path, expected_sha: str,
                    name: str) -> tuple[dict, dict, Path, Path]:
    from . import controls
    if data.sha256_file(registration) != expected_sha:
        raise ValueError('Control-audit sidecar SHA differs from locked argv')
    sidecar = json.loads(registration.read_text())
    if (sidecar.get('schema') != 'pcrl-control-audit-sidecar-v1'
            or sidecar.get('anchor') != 0 or sidecar.get('slate') != 'standard'
            or sidecar.get('outer_pool_opened') is not False):
        raise ValueError('Unexpected registered control-audit scope')
    for path_key, hash_key in (
            ('input_index_relative_path', 'input_index_sha256'),
            ('simple_map_manifest_relative_path', 'simple_map_manifest_sha256'),
            ('fitted_control_source_relative_path', 'fitted_control_source_sha256')):
        source = _under_root(root, sidecar[path_key])
        if data.sha256_file(source) != sidecar[hash_key]:
            raise ValueError(f'Sidecar dependency changed: {path_key}')
    if data.sha256_file(root / 'results/pcrl_task_aligned_cuts_v1/PROTOCOL_LOCK.json') != sidecar['protocol_lock_sha256']:
        raise ValueError('Original protocol lock changed')
    h_root = _under_root(root, sidecar['shared_H_root_relative_path'])
    h_receipt = _verified_shared_h_receipt(h_root)
    if h_receipt['source_complete_sha256'] != sidecar['shared_H_source_complete_sha256']:
        raise ValueError('Shared H source receipt changed')
    matches = [(family, record) for family in ('simple_map_audit_units', 'fitted_control_audits')
               for label, record in sidecar[family].items() if label == name]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous registered control label')
    _, record = matches[0]
    if record['status'] == 'aliased_to_registered_unit':
        raise ValueError(f'Exact Q alias; audit canonical {record["canonical_release_id"]}')
    if record['status'] != 'registered_unrun' or record['canonical_release_id'] != record['release_id']:
        raise ValueError('Control label is not a distinct registered release')
    source = _under_root(root, record['source_relative_path'])
    if data.sha256_file(source) != record['source_file_sha256']:
        raise ValueError('Pinned control release file changed')
    with np.load(source, allow_pickle=False) as saved:
        if list(saved) != ['Q']:
            raise ValueError('Control release archive must contain only Q')
        q = np.ascontiguousarray(np.asarray(saved['Q'], dtype=np.float64))
    if (list(q.shape) != record['shape'] or q.shape[0] != 32 or q.shape[1] < 2
            or not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(axis=1), 1., atol=1e-8, rtol=0)
            or controls._array_sha(q) != record['source_array_sha256']):
        raise ValueError('Pinned control release array changed or is malformed')
    output = _under_root(root, record['output_relative_dir'])
    if 'private' not in output.parts:
        raise ValueError('Control audit output must be private')
    return sidecar, record, q, output


def _artifact_hashes(output: Path) -> dict[str, str]:
    result = {}
    for path in sorted(output.rglob('*')):
        if path.is_symlink():
            raise ValueError('Scientific receipt cannot contain symlinks')
        if path.is_file() and path != output / 'SIDECAR_COMPLETE.json':
            result[str(path.relative_to(output))] = data.sha256_file(path)
    return result


def audit_registered_control(registration_path: str | Path, expected_sidecar_sha256: str,
                             name: str, *, verify_only: bool = False) -> dict:
    """Run or verify one hash-pinned distinct control on the common inner slate."""
    root = Path.cwd().resolve()
    registration = Path(registration_path).resolve()
    if not registration.is_relative_to(root):
        raise ValueError('Control registration must belong to this worktree')
    sidecar, record, q, output = _sidecar_inputs(
        root, registration, expected_sidecar_sha256, name)
    receipt_path = output / 'SIDECAR_COMPLETE.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        report_path = output / 'INNER_PANEL.json'
        if not report_path.is_file():
            raise ValueError('Completed control audit lacks its inner-panel report')
        report = json.loads(report_path.read_text())
        raw_q_sha = hashlib.sha256(q.tobytes()).hexdigest()
        if (receipt.get('schema') != 'pcrl-control-audit-complete-v1'
                or receipt.get('unit_id') != record['release_id']
                or receipt.get('release_id') != record['release_id']
                or receipt.get('sidecar_sha256') != expected_sidecar_sha256
                or receipt.get('source_file_sha256') != record['source_file_sha256']
                or receipt.get('source_array_sha256') != record['source_array_sha256']
                or receipt.get('inner_panel_channel_sha256') != raw_q_sha
                or receipt.get('protocol_lock_sha256') != sidecar['protocol_lock_sha256']
                or receipt.get('input_index_sha256') != sidecar['input_index_sha256']
                or receipt.get('H_source_complete_sha256') != sidecar['shared_H_source_complete_sha256']
                or report.get('release_id') != record['release_id']
                or report.get('channel_sha256') != raw_q_sha
                or report.get('split_assignment_sha256') != sidecar['inner_split_sha256']
                or report.get('outer_pool_opened') is not False
                or receipt.get('artifacts') != _artifact_hashes(output)):
            raise ValueError('Completed control audit receipt or artifacts changed')
        return {'status': 'verified_complete', 'unit_id': record['release_id'],
                'receipt_sha256': data.sha256_file(receipt_path)}
    if verify_only:
        raise FileNotFoundError('Control audit has no completed receipt')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Partial control audit preserved; quarantine before retry')
    output.mkdir(parents=True, exist_ok=True)
    from .pipeline import source_tree
    source_pin = source_tree()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    value = data.index(_under_root(root, sidecar['input_index_relative_path']))
    all_prepared = {anchor: data.load_prepared(value, anchor) for anchor in (0, 1, 2)}
    split = data.global_validation_split(all_prepared)
    if split['assignment_sha256'] != sidecar['inner_split_sha256']:
        raise ValueError('Registered inner household split changed')
    report = run_inner_panel(
        all_prepared[sidecar['anchor']], q, split, record['release_id'], output,
        slate=sidecar['slate'],
        shared_h_root=_under_root(root, sidecar['shared_H_root_relative_path']))
    raw_q_sha = hashlib.sha256(q.tobytes()).hexdigest()
    if report['channel_sha256'] != raw_q_sha:
        raise ValueError('Inner-pilot raw Q digest differs from validated control array')
    provenance = {
        'schema': 1, 'unit_id': record['release_id'],
        'git_head_at_audit': head, **source_pin,
        'sidecar_sha256': expected_sidecar_sha256,
        'source_file_sha256': record['source_file_sha256'],
        'source_array_sha256': record['source_array_sha256'],
        'inner_panel_channel_sha256': raw_q_sha,
        'protocol_lock_sha256': sidecar['protocol_lock_sha256'],
        'input_index_sha256': sidecar['input_index_sha256'],
        'split_assignment_sha256': split['assignment_sha256'],
        'shared_H_source_complete_sha256': sidecar['shared_H_source_complete_sha256'],
        'inner_panel_sha256': data.sha256_file(output / 'INNER_PANEL.json'),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json_atomic(output / 'SOURCE_PROVENANCE.json', provenance)
    receipt = {
        'schema': 'pcrl-control-audit-complete-v1',
        'unit_id': record['release_id'], 'release_id': record['release_id'],
        'sidecar_sha256': expected_sidecar_sha256,
        'source_file_sha256': record['source_file_sha256'],
        'source_array_sha256': record['source_array_sha256'],
        'inner_panel_channel_sha256': raw_q_sha,
        'protocol_lock_sha256': sidecar['protocol_lock_sha256'],
        'input_index_sha256': sidecar['input_index_sha256'],
        'H_source_complete_sha256': sidecar['shared_H_source_complete_sha256'],
        'source_tree_sha256': source_pin['source_tree_sha256'],
        'artifacts': _artifact_hashes(output),
        'completed_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json_atomic(receipt_path, receipt)
    return {'status': 'complete', 'unit_id': record['release_id'],
            'receipt_sha256': data.sha256_file(receipt_path)}


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit-control',))
    parser.add_argument('--sidecar', required=True)
    parser.add_argument('--sidecar-sha256', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    result = audit_registered_control(args.sidecar, args.sidecar_sha256,
                                      args.name, verify_only=args.verify_only)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
