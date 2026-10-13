"""Prepare final independent configs from ROOT's completed report metadata."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from build_audit_config import ROOT, PRIVATE, PUBLIC, metadata, metadata_pin, receipt_pin, pins, write, worker_config


def source_list(cfg, config_pin, manifest):
    code = [metadata_pin(ROOT / name) for name in ('occupation_bottleneck_independent/replay.py',
            'occupation_bottleneck_independent/__init__.py', 'occupation_bottleneck/scheduler.py',
            'occupation_bottleneck/admission.py', 'occupation_bottleneck/source.py')]
    unique = {}
    for item in list(pins(cfg)) + list(pins(manifest)) + code + [config_pin]:
        path = str(Path(item['path']).absolute())
        if path in unique and unique[path] != item['sha256']:
            raise ValueError('inconsistent exact source pins')
        unique[path] = item['sha256']
    return [{'path': path, 'sha256': sha} for path, sha in sorted(unique.items())]


def prepare(report_config, output, inference_directory=None):
    report_config = Path(report_config).absolute()
    cfg = metadata(report_config)
    state = metadata(PRIVATE / 'compute/state.json')
    # Treat ROOT's report as complete only when the active queue recorded it.
    worker_config(state, {'unit_id': cfg['unit_id']})
    public = Path(cfg['public_output'])
    index = metadata(Path(cfg['private_output']) / 'REPORT_INPUT_INDEX.json')
    manifest = metadata(cfg['manifest']['path'])
    seal = cfg['reporting_seal']
    if isinstance(seal, dict) and 'roles' in seal:
        operational = seal
    else:
        operational = metadata(seal['path'])
    truth = operational['roles']['INNER_SELECTION']
    if set(truth) != {'income', 'occupation', 'sex'}:
        raise ValueError('exact ROOT-supplied three INNER truth pins required')
    packages = metadata(PRIVATE / 'root/CANONICAL_PACKAGES.json')['entries']
    expected = {(p['arm_id'], p['seed']): p for p in packages}
    if len(cfg['packages']) != 72 or {(p['arm_id'], p['seed']) for p in cfg['packages']} != set(expected):
        raise ValueError('complete ROOT report package roster differs')
    banks = []
    for package in cfg['packages']:
        entry = expected[package['arm_id'], package['seed']]
        freeze = metadata(package['freeze']['path'])
        if metadata_pin(package['freeze']['path']) != package['freeze'] or freeze['bindings']['checkpoint_sha256'] != entry['checkpoint_sha256']:
            raise ValueError('ROOT reporting bank differs from admitted exact checkpoint')
        bank = {key: package[key] for key in ('arm_id', 'seed', 'freeze', 'release', 'scores', 'fit_release', 'fit_scores')}
        bank['release_variant'] = package.get('release_variant', 'full')
        bank['historical_point_ineligible'] = bool(entry.get('historical_point_ineligible', False))
        if bank['release_variant'] == 'head_row_space':
            bank['source_arm'] = entry['source_arm']
        bank['att_release'] = receipt_pin(Path(package['freeze']['path']).parent / 'ATT_RELEASE.npz', freeze['att_release_sha256'])
        bank['att_scores'] = receipt_pin(Path(package['freeze']['path']).parent / 'ATT_SCORES.npz', freeze['att_scores_sha256'])
        bank['reported_metrics'] = index['arrays'][f"metrics__{package['arm_id']}__S{package['seed']}"]
        bank['reported_scores'] = index['arrays'][f"scores__{package['arm_id']}__S{package['seed']}"]
        banks.append(bank)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    reporting_lock = metadata_pin(PUBLIC / 'REPORTING_LOCK.json')
    headline = {'unit_id': 'OB_INDEPENDENT_HEADLINES_R7', 'operation': 'headlines',
                'public_output': str(PUBLIC / 'r7/INDEPENDENT_HEADLINES.json'),
                'manifest': cfg['manifest'], 'reporting_lock': reporting_lock,
                'truth': truth, 'references': {'s'+str(seed): manifest['roles']['INNER_SELECTION']['references_s'+str(seed)] for seed in (0, 1)},
                'banks': banks, 'c3_table': cfg['label_table'],
                'public_report': metadata_pin(public / 'REPORT_PACKAGE_METRICS.json'),
                'c3_summary': metadata_pin(public / 'C3_TASK_ASSOCIATION.json'),
                'decision_aliases': metadata_pin(public / 'DECISION_READER_ALIASES.json'),
                'report_summary': metadata_pin(public / 'REPORT_SUMMARY.json'),
                'reported_truth': index['arrays']['truth'],
                'control_gates': index['arrays']['control_gates']}
    if isinstance(seal, dict) and 'path' in seal:
        headline['operational_seal'] = {'path': seal['path'], 'sha256': seal['sha256']}
    cp = write(output / 'HEADLINE_CONFIG.json', headline)
    write(output / 'HEADLINE_CONFIG_SOURCES.json', source_list(headline, cp, manifest))
    families = []
    if inference_directory is not None:
        for family in ('utility', 'privacy'):
            runner = Path(inference_directory) / (family.upper() + '_INFERENCE.json')
            assessment = {'unit_id': 'OB_INDEPENDENT_'+family.upper()+'_R7', 'operation': 'assessment',
                          'public_output': str(PUBLIC / 'r7' / ('INDEPENDENT_'+family.upper()+'.json')),
                          'private_output': str(output / ('INDEPENDENT_'+family.upper()+'_DRAWS')),
                          'reporting_lock': reporting_lock, 'endpoint_registry': cfg['endpoint_registries'][family],
                          'unit_input': index['unit_input'], 'arrays': index['arrays'], 'runner_summary': metadata_pin(runner)}
            registry = metadata(assessment['endpoint_registry']['path'])
            if registry['K'] != (864 if family == 'utility' else 108):
                raise ValueError('complete family dimensions differ')
            cp = write(output / (family.upper() + '_ASSESSMENT_CONFIG.json'), assessment)
            write(output / (family.upper() + '_ASSESSMENT_CONFIG_SOURCES.json'), source_list(assessment, cp, manifest))
            families.append(family)
    status = {'status': 'READY_FOR_ROOT_QUEUE_REGISTRATION', 'metadata_only': True,
              'numerical_files_opened_or_hashed': False, 'headline_packages': 72,
              'headline_roles': ['FIT2', 'INNER'], 'supported_stratum_banks': 144,
              'assessment_families_prepared': families, 'code_pin': metadata_pin(ROOT / 'occupation_bottleneck_independent/replay.py'),
              'numerical_replay_complete': False}
    write(output / 'PREPARATION_STATUS.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--inference-dir', help='Completed ROOT utility/privacy inference output directory')
    args = parser.parse_args()
    print(json.dumps(prepare(args.report_config, Path(args.output_dir).absolute(), args.inference_dir), sort_keys=True))


if __name__ == '__main__':
    main()
