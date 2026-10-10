"""Queue-only three-claim accounting from pinned public development receipts.

This worker does no endpoint fitting and never opens person-level arrays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .r3_claims import all_claims
from .r3_family import ARMS, K, ordered_clause_ids
from .r3_recovery_intervals import STOCHASTIC
from .r3_selection import GATE_SCHEMA, strongest_control
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def run(config_path):
    active = assert_active_from_env(stage='verification')
    state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT']) / 'state.json').read_text())
    spec = state['units'][active['unit_id']]['spec']
    registered = {Path(x['path']).resolve(): x['sha256'] for x in spec['sources']}

    def pinned(item):
        path = Path(item['path'])
        if path.is_symlink():
            raise PermissionError('claim source symlink refused')
        path = path.resolve(strict=True)
        if registered.get(path) != item['sha256'] or sha(path) != item['sha256']:
            raise PermissionError('claim source not byte-pinned')
        return path

    config = Path(config_path).resolve(strict=True)
    if registered.get(config) != sha(config):
        raise PermissionError('claim config not byte-pinned')
    cfg = json.loads(config.read_text())
    if cfg.get('schema') != 'pcrl-incumbent-r3-claim-config-v1' or cfg.get('unit_id') != active['unit_id'] or cfg.get('K') != K:
        raise PermissionError('claim config identity differs')
    admission = pinned(cfg['admission_lock'])
    science = pinned(cfg['campaign_seal'])
    amendment = pinned(cfg['verification_amendment'])
    if sha(admission) != active['seal_sha256'] or sha(admission) != state['seal']['sha256']:
        raise PermissionError('claim queue admission seal differs')
    if sha(science) not in {a['sha256'] for a in state['amendments'] if 'protection' in a['stages']}:
        raise PermissionError('claim science lock not registered')
    verification = [a for a in state['amendments'] if 'verification' in a['stages']]
    if (not verification or verification[-1]['sha256'] != sha(amendment)
            or spec.get('amendment_sha256') != sha(amendment)
            or json.loads(amendment.read_text()).get('campaign_lock_sha256') != sha(science)):
        raise PermissionError('claim verification amendment does not bind science')

    nomination = json.loads(pinned(cfg['nomination']).read_text())
    utility = json.loads(pinned(cfg['utility_intervals']).read_text())
    recovery = json.loads(pinned(cfg['recovery_intervals']).read_text())
    if (nomination.get('schema') != 'pcrl-incumbent-r3-internal-nomination-v1'
            or nomination.get('final_attack_results_used') is not False
            or utility.get('schema') != 'pcrl-incumbent-r3-utility-intervals-v1'
            or recovery.get('schema') != 'pcrl-incumbent-r3-recovery-inference-v1'
            or utility.get('K') != K or recovery.get('K') != K
            or utility.get('utility_slots') != 192 or recovery.get('recovery_slots') != 756):
        raise PermissionError('fixed nomination or interval scope differs')
    slots = {**utility['slots'], **recovery['slots']}
    if (len(utility['slots']) != 192 or len(recovery['slots']) != 756
            or len(slots) != K or set(slots) != set(ordered_clause_ids())):
        raise PermissionError('K948 slot inventory differs')
    if set(cfg['gates']) != set(ARMS):
        raise PermissionError('all eight original point gates required')
    gates = {}
    for arm in ARMS:
        gate = json.loads(pinned(cfg['gates'][arm]).read_text())
        if gate.get('schema') != GATE_SCHEMA or gate.get('arm_id') != arm or gate.get('original_inner', {}).get('required') != 24:
            raise PermissionError('original utility gate scope differs')
        gates[arm] = gate

    nominee = nomination['nominee']
    slate = [x for x in nomination['slate'] if x['arm_id'] == nominee]
    if nominee is not None and len(slate) != 1:
        raise PermissionError('nominee absent from fixed FIT slate')
    exact_alias = bool(slate[0]['both_seed_exact_p0_alias']) if slate else False
    claims = all_claims(nominee, gates, slots, nominee_exact_p0_alias=exact_alias)

    pair_auc = {}
    for arm in ARMS:
        values = []
        for seed in (0, 1):
            draws = range(3) if arm in STOCHASTIC else (0,)
            keys = [f'{arm}:s{seed}:d{draw}:pair' for draw in draws]
            if any(key not in recovery['point_envelope_aucs'] for key in keys):
                break
            values.append(sum(float(recovery['point_envelope_aucs'][key]) for key in keys) / len(keys))
        if len(values) == 2:
            pair_auc[arm] = sum(values) / 2
    selected_control = ({'status': 'UNTESTED', 'selected': None}
                        if nominee is None else strongest_control(nominee, gates, pair_auc))

    output = Path(cfg['output']).resolve()
    owner = Path(__file__).resolve().parents[1] / 'results/pcrl_incumbent_view_risk_v1/r3'
    if output.parent != owner or output.exists() or output.is_symlink():
        raise PermissionError('claim output must be new in R3')
    report = {'schema': 'pcrl-incumbent-r3-fixed-claim-accounting-v1', 'K': K,
              'nominee': nominee, 'claims': claims, 'strongest_control': selected_control,
              'equal_seed_mean_pair_auc': pair_auc,
              'source_sha256': {'nomination': sha(Path(cfg['nomination']['path'])),
                                'utility_intervals': sha(Path(cfg['utility_intervals']['path'])),
                                'recovery_intervals': sha(Path(cfg['recovery_intervals']['path'])),
                                'gates': {arm: sha(Path(cfg['gates'][arm]['path'])) for arm in ARMS}},
              'queue': active, 'old_assessment_read': False,
              'scope': 'selected on reused FIT development data; intervals are conditional development inference'}
    with output.open('x') as target:
        json.dump(report, target, sort_keys=True, indent=2)
        target.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    result = run(parser.parse_args().config)
    print(json.dumps({'K': result['K'], 'nominee': result['nominee'],
                      'claim_status': {key: result['claims'][key]['status'] for key in ('A', 'B', 'C')}}))


if __name__ == '__main__':
    main()
