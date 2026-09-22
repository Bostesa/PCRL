"""Public inventory only: registered inputs, no ACS/private artifact access."""
import importlib
import json
from pathlib import Path
import shutil

import pytest

from experiments.pcrl_task_directed_release_v1 import config


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.ledger')


@pytest.fixture
def public_inputs(tmp_path):
    for name in ('CONFIGS.json', 'RESOURCE_SCHEDULE.json',
            'EXTRA_CONFIGS.json', 'EXTRA_RESOURCE_SCHEDULE.json', 'ROBUSTNESS_CONFIGS.json',
            'ROBUSTNESS_RESOURCE_SCHEDULE.json', 'BASELINE_SUPPLEMENTS.json', 'BASELINE_SUPPLEMENT_SCHEDULE.json'):
        shutil.copy(config.OUT/name, tmp_path/name)
    (tmp_path/'ALL_RELEASES.json').write_text(json.dumps(config.release_ledger()))
    return tmp_path


def rewrite(path, value, payload_field=None):
    if payload_field:
        value[payload_field] = config.digest({k: v for k, v in value.items() if k != payload_field})
    path.write_text(json.dumps(value))


def test_combines_public_registered_and_scheduled_units_without_mutation(public_inputs):
    before = {p.name: p.read_bytes() for p in public_inputs.iterdir()}
    got = module().build_ledger(study_out=public_inputs)
    assert got['nominal_primary_Q_maps'] == 81
    assert got['nominal_Q_maps'] == 162 and got['scheduled_Q_maps'] == 126
    assert got['nominal_release_anchor_records'] == 333 and got['scheduled_release_anchor_records'] == 297
    assert got['resource_unscheduled_release_anchor_records'] == 36
    assert got['nominal_role_audit_units'] == 333*16 and got['scheduled_role_audit_units'] == 297*16
    assert len({r['id'] for r in got['records']}) == 333
    for record in got['records']:
        assert len(record['roles']) == 16 and len(record['audit_roles'])+len(record['utility_roles']) == 16
        assert record['resource_status'] == ('scheduled' if record['scheduled'] else 'unscheduled_resource')
        assert record['registration_sha256'] and record['schedule_sha256']
    assert before == {p.name: p.read_bytes() for p in public_inputs.iterdir()}


@pytest.mark.parametrize('mutation', ['duplicate_unit', 'unknown_schedule', 'duplicate_schedule', 'wrong_registry_pin'])
def test_invalid_identity_or_schedule_is_rejected(public_inputs, mutation):
    registry_path = public_inputs/'EXTRA_CONFIGS.json'; schedule_path = public_inputs/'EXTRA_RESOURCE_SCHEDULE.json'
    registry = json.loads(registry_path.read_text()); schedule = json.loads(schedule_path.read_text())
    if mutation == 'duplicate_unit':
        registry['controls'].append(registry['controls'][0])
        rewrite(registry_path, registry, 'registration_payload_hash')
        schedule['extra_registration_sha256'] = module().file_hash(registry_path)
    elif mutation == 'unknown_schedule':
        schedule['unit_ids'].append('unknown/anchor_0')
    elif mutation == 'duplicate_schedule':
        schedule['unit_ids'].append(schedule['unit_ids'][0])
    else:
        schedule['extra_registration_sha256'] = '0'*64
    rewrite(schedule_path, schedule, 'schedule_payload_hash')
    with pytest.raises(ValueError):
        module().build_ledger(study_out=public_inputs)


def test_absent_optional_schedule_retains_all_registered_units_as_resource_unscheduled(public_inputs):
    (public_inputs/'EXTRA_RESOURCE_SCHEDULE.json').unlink()
    got = module().build_ledger(study_out=public_inputs)
    branch = [r for r in got['records'] if r['branch'] == 'A']
    assert len(branch) == 123 and all(not r['scheduled'] for r in branch)
    assert all(r['schedule_sha256'] is None for r in branch)


def test_atomic_regeneration_pins_and_preserves_primary_ledger_and_is_idempotent(public_inputs):
    old = (public_inputs/'ALL_RELEASES.json').read_bytes()
    got = module().regenerate(study_out=public_inputs)
    provenance = got['primary_ledger_provenance']
    assert (public_inputs/provenance['preserved_file']).read_bytes() == old
    assert provenance['sha256'] == module().file_hash(public_inputs/provenance['preserved_file'])
    assert got == json.loads((public_inputs/'ALL_RELEASES.json').read_text())
    snapshot = {p.name: p.read_bytes() for p in public_inputs.iterdir()}
    assert module().regenerate(study_out=public_inputs) == got
    assert snapshot == {p.name: p.read_bytes() for p in public_inputs.iterdir()}
