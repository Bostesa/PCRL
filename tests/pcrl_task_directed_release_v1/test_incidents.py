import json
import numpy as np
from experiments.pcrl_task_directed_release_v1.audits import expected_token_loss
from experiments.pcrl_task_directed_release_v1.incidents import preserve_arrays


def test_invalid_zero_probability_rows_are_preserved_without_repair(tmp_path):
    q=np.zeros((2,2,2));p=np.full((2,2),.5);labels=np.array([0,1])
    try:expected_token_loss(q,p,labels)
    except ValueError as error:result=preserve_arrays(error,tmp_path/'quarantine')
    manifest=json.loads((tmp_path/'quarantine/MANIFEST.json').read_text())
    snapshots=[np.load(tmp_path/'quarantine'/row['file']) for row in manifest['arrays']]
    assert any(a.shape==q.shape and np.array_equal(a,q) for a in snapshots)
    assert result['arrays_preserved']>0
    assert all(row['accepted_for_scoring'] is False for row in manifest['arrays'])
    assert not q.any()
