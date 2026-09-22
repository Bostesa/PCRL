"""The evaluation lock must reject changed fitted artifacts before final access."""
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_final_prospective_v1.audit_panel import sha_of, unit_dir
from experiments.pcrl_final_prospective_v1.common import ROOT, sha
from experiments.pcrl_final_prospective_v1.lock import validate_unit


def assert_lock_rejects_changed_fitted_model(tmp_path):
    unit = ('Q', 0, 'attack:A/SEX')
    d = unit_dir(tmp_path, *unit)
    d.mkdir(parents=True)
    joblib.dump({'dataset': 'acs2016', 'release': 'Q', 'anchor': 0,
                 'role': unit[2], 'selection': 'candidate',
                 'independent_selection': 'candidate',
                 'candidates': {'candidate': {'origin': 'independent'}},
                 'validation_scores': {'candidate': {'unweighted': 0.25, 'weighted': 0.25,
                                                     'balanced': 0.25}}}, d/'registry.joblib')
    np.savez_compressed(d/'val_losses.npz', candidate_ids=np.array(['candidate']),
                        losses=np.array([[0.2], [0.3]]), weights=np.array([1., 1.]))
    model = d/'model.bin'
    model.write_bytes(b'original model')
    artifacts = {str(p.relative_to(tmp_path)): sha(p)
                 for p in (d/'registry.joblib', d/'val_losses.npz', model)}
    (d/'ARTIFACTS.json').write_text(json.dumps(artifacts))
    (d/'COMPLETE.json').write_text(json.dumps({
        'dataset': 'acs2016', 'release': 'Q', 'anchor': 0, 'role': unit[2],
        'selection': 'candidate', 'independent_selection': 'candidate',
        'registry_sha256': sha(d/'registry.joblib'),
        'artifacts_sha256': sha_of(artifacts), 'artifact_count': len(artifacts)}))

    validate_unit(tmp_path, unit, 'acs2016')
    model.write_bytes(b'changed model')
    try:
        validate_unit(tmp_path, unit, 'acs2016')
    except ValueError as e:
        assert 'artifact hash' in str(e)
    else:
        raise AssertionError('Changed fitted model passed lock validation')


class LockValidationTest(unittest.TestCase):
    def test_rejects_changed_fitted_model(self):
        with TemporaryDirectory(dir=ROOT) as path:
            assert_lock_rejects_changed_fitted_model(Path(path))
