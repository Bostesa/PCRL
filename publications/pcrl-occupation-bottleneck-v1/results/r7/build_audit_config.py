"""Prepare independent replay inputs using JSON receipts only; never run science.

No checkpoint, array, reader model or other numerical file is opened or hashed.
Numerical file hashes are copied from completed worker receipts and admitted
metadata. The root owns queue registration and every scientific launch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = ROOT / '.private/occupation_bottleneck'
PUBLIC = ROOT / 'results/pcrl_occupation_bottleneck_v1'
METADATA_SUFFIXES = {'.json', '.py', '.md', '.txt', '.csv', '.diff'}


def metadata(path):
    path = Path(path).absolute()
    if path.suffix != '.json' or any(p.is_symlink() for p in (path, *path.parents)):
        raise PermissionError('explicit nonsymlink JSON metadata required')
    return json.loads(path.read_text())


def metadata_pin(path):
    path = Path(path).absolute()
    if path.suffix not in METADATA_SUFFIXES or any(p.is_symlink() for p in (path, *path.parents)):
        raise PermissionError('numerical file hashing refused')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def receipt_pin(path, sha256):
    # Existence is metadata; content validation belongs to the queued replay.
    path = Path(path).absolute()
    if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in '0123456789abcdef' for c in sha256):
        raise ValueError('receipt must supply an exact SHA-256')
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('receipt source absent or symlinked: ' + str(path))
    return {'path': str(path), 'sha256': sha256}


def pins(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield {'path': value['path'], 'sha256': value['sha256']}
        else:
            for child in value.values():
                yield from pins(child)
    elif isinstance(value, list):
        for child in value:
            yield from pins(child)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)
    return metadata_pin(path)


def worker_config(state, receipt):
    unit = state['units'][receipt['unit_id']]
    if unit['status'] != 'done':
        raise ValueError('receipt worker not completed')
    argv = unit['spec']['argv']
    path = argv[argv.index('--spec') + 1]
    config_pin = metadata_pin(path)
    if config_pin not in unit['spec']['sources']:
        raise ValueError('worker config differs from exact registered source')
    return metadata(path)


def frozen_rule():
    # Parse the literal rather than importing geometry.py (which imports numpy).
    tree = ast.parse((ROOT / 'occupation_bottleneck/geometry.py').read_text())
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'RULE' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('unique frozen geometry rule required')
    return ast.literal_eval(assignments[0].value)


def prepare(output, gradient_items=None):
    canonical = metadata(PRIVATE / 'root/CANONICAL_PACKAGES.json')['entries']
    roster = metadata(PRIVATE / 'r1/ROSTER_METADATA.json')
    plan = metadata(PRIVATE / 'root/BANK_PLAN.json')['rows']
    manifest_path = PRIVATE / 'r1/SOURCE_PATHS.json'
    manifest = metadata(manifest_path)
    state = metadata(PRIVATE / 'compute/state.json')
    if len(canonical) != 72 or len(plan) != 72 or len(roster['entries']) != 84:
        raise ValueError('frozen roster dimensions differ')
    full = [entry for entry in canonical if entry['release_variant'] == 'full']
    if len(full) != 66 or len({(e['arm_id'], e['seed']) for e in canonical}) != 72:
        raise ValueError('canonical full or unique package dimensions differ')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    missing = []
    banks, trajectories, development_banks = [], [], []
    bank_by_tag, utility_by_tag = {}, {}
    for entry in canonical:
        tag = f"{entry['arm_id']}_S{entry['seed']}"
        public_path = PUBLIC / 'banks' / (tag + '.json')
        if not public_path.is_file():
            missing.append('bank:' + tag)
            continue
        receipt = metadata(public_path)
        if receipt.get('status') != 'FROZEN_COMPLETE' or receipt.get('arm_id') != entry['arm_id'] or receipt.get('source_seed') != entry['seed']:
            raise ValueError('bank receipt identity/status mismatch: ' + tag)
        config = worker_config(state, receipt)
        if receipt['bindings']['checkpoint_sha256'] != entry['checkpoint_sha256'] or config['checkpoint_sha256'] != entry['checkpoint_sha256']:
            raise ValueError('bank checkpoint binding mismatch: ' + tag)
        directory = Path(config['private_output'])
        freeze = receipt_pin(directory / 'FROZEN_COMPLETE.json', receipt['frozen_complete_sha256'])
        if metadata_pin(freeze['path']) != freeze:
            raise ValueError('bank metadata freeze differs')
        bank_by_tag[tag] = freeze
        banks.append({'arm_id': entry['arm_id'], 'seed': entry['seed'], 'ATT_validation': True,
                      'release': receipt_pin(directory / 'ATT_RELEASE.npz', receipt['att_release_sha256']),
                      'scores': receipt_pin(directory / 'ATT_SCORES.npz', receipt['att_scores_sha256']), 'freeze': freeze})
        development_banks.append({'arm_id': entry['arm_id'], 'seed': entry['seed'],
                                  'release': receipt_pin(directory / 'FIT_SELECTION_RELEASE.npz', receipt['FIT_selection_release_sha256']),
                                  'scores': receipt_pin(directory / 'FIT_SELECTION_SCORES.npz', receipt['FIT_selection_scores_sha256'])})
    for entry in full:
        tag = f"{entry['arm_id']}_S{entry['seed']}"
        public_path = PUBLIC / 'trajectories' / (tag + '.json')
        if not public_path.is_file():
            missing.append('trajectory:' + tag)
            continue
        receipt = metadata(public_path)
        if receipt.get('arm_id') != entry['arm_id'] or receipt.get('seed') != entry['seed'] or receipt.get('checkpoint_sha256') != entry['checkpoint_sha256']:
            raise ValueError('trajectory receipt identity mismatch: ' + tag)
        config = worker_config(state, receipt)
        summary = metadata_pin(public_path)
        utility_by_tag[tag] = summary
        trajectories.append({'inputs': receipt_pin(Path(config['private_output']) / 'FIT_SELECTION_UTILITY.npz', receipt['utility_input_sha256']), 'summary': summary})
    status = {'schema': 'OB1-independent-replay-preparation-v1', 'metadata_only': True,
              'numerical_files_opened_or_hashed': False, 'banks_present': len(banks),
              'trajectories_present': len(trajectories), 'expected_banks': 72,
              'expected_trajectories': 66, 'missing_receipts': missing,
              'gradients_supplied': gradient_items is not None,
              'science_launch_authority': 'ROOT shared queue only'}
    if missing:
        status['status'] = 'INCOMPLETE_NO_RUNNABLE_CONFIG_WRITTEN'
        write(output / 'PREPARATION_STATUS.json', status)
        return status
    paths = []
    core = [e for e in roster['entries'] if e['kind'] == 'CORE_PATH']
    groups = sorted({(e['family'], e['strength'], e['seed']) for e in core})
    for family, strength, seed in groups:
        entries = sorted((e for e in core if (e['family'], e['strength'], e['seed']) == (family, strength, seed)), key=lambda e: e['epoch'])
        if [e['epoch'] for e in entries] != [0, 5, 10, 20]:
            raise ValueError('fixed four-checkpoint path differs')
        points = []
        for entry in entries:
            tag = entry['canonical_model_alias']
            selection = receipt_pin(entry['index'], entry['index_sha256'])
            if metadata_pin(selection['path']) != selection:
                raise ValueError('saved path metadata index differs')
            points.append({'epoch': entry['epoch'], 'checkpoint_sha256': entry['checkpoint_sha256'],
                           'selection': selection, 'bank_freeze': bank_by_tag[tag], 'utility_summary': utility_by_tag[tag]})
        paths.append({'family': family, 'strength': strength, 'seed': seed, 'points': points})
    geometry = []
    rule = frozen_rule()
    for entry in full:
        if entry['arm_id'] not in ('U0', 'RAW_JOINT_R03_E5', 'JOINT_CONDITIONAL_R03_E5'):
            continue
        for role, rolepins in manifest['roles'].items():
            geometry.append({'arm_id': entry['arm_id'], 'seed': entry['seed'], 'role': role,
                             'checkpoint': receipt_pin(entry['checkpoint'], entry['checkpoint_sha256']),
                             'features': rolepins['features'], 'rule': rule})
    if len(paths) != 18 or len(geometry) != 18:
        raise ValueError('fixed path/geometry dimensions differ')
    audit = {'unit_id': 'OB_INDEPENDENT_AUDIT_R7', 'operation': 'audit',
             'public_output': str(PUBLIC / 'r7/INDEPENDENT_AUDIT.json'),
             'source_admission': {'manifest': metadata_pin(manifest_path), 'roster': metadata_pin(PRIVATE / 'r1/ROSTER_METADATA.json')},
             'trajectory': trajectories, 'banks': banks, 'geometry': geometry, 'paths': paths}
    if gradient_items is not None:
        if not set(gradient_items).issubset({'gradients', 'gradient_regeneration'}) or not gradient_items:
            raise ValueError('gradient metadata must contain only registered replay lists')
        audit.update(gradient_items)
    development = {'unit_id': 'OB_INDEPENDENT_C3_R7', 'operation': 'development',
                   'public_output': str(PUBLIC / 'r7/INDEPENDENT_C3.json'),
                   'manifest': metadata_pin(manifest_path), 'banks': development_banks}
    code = [metadata_pin(ROOT / name) for name in ('occupation_bottleneck_independent/replay.py',
            'occupation_bottleneck_independent/__init__.py', 'occupation_bottleneck/scheduler.py',
            'occupation_bottleneck/admission.py', 'occupation_bottleneck/source.py')]
    for name, cfg in (('AUDIT_CONFIG', audit), ('C3_CONFIG', development)):
        config_pin = write(output / (name + '.json'), cfg)
        sources = list(pins(cfg)) + list(pins(manifest)) + code + [config_pin]
        if name == 'AUDIT_CONFIG':
            sources += [{'path': e['checkpoint'], 'sha256': e['checkpoint_sha256']} for e in roster['entries'] if e['exists']]
        unique = {}
        for item in sources:
            path = str(Path(item['path']).absolute())
            if path in unique and unique[path] != item['sha256']:
                raise ValueError('source path has inconsistent hashes')
            unique[path] = item['sha256']
        write(output / (name + '_SOURCES.json'), [{'path': p, 'sha256': s} for p, s in sorted(unique.items())])
    status.update(status='READY_FOR_ROOT_QUEUE_REGISTRATION', paths=len(paths), geometry_role_checks=len(geometry),
                  headline_replay_complete=False, code_pins=code)
    write(output / 'PREPARATION_STATUS.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--gradient-items', help='JSON with gradients and/or gradient_regeneration lists')
    args = parser.parse_args()
    gradient_items = metadata(args.gradient_items) if args.gradient_items else None
    status = prepare(Path(args.output_dir).absolute(), gradient_items)
    print(json.dumps(status, sort_keys=True))
    if status['missing_receipts']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
