import numpy as np
import pytest


def test_matrix_has_all_requested_positive_zero_and_unconstrained_slots():
    from experiments.pcrl_task_directed_release_v1.config import configuration
    cfg = configuration()
    rows = cfg['maps']
    assert len(rows) == 81
    assert len({r['id'] for r in rows}) == 81
    assert sum(r['budget'] is not None and r['budget'] > 0 for r in rows) == 54
    assert sum(r['budget'] == 0 for r in rows) == 18
    assert sum(r['budget'] is None for r in rows) == 9
    assert sum(r['input'] == 'T0' and r['budget'] is not None and r['budget'] > 0 for r in rows) == 18


def test_runtime_view_rejects_label_or_other_recipient_features():
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
    x = np.zeros((3, 32)); ha = np.zeros((3, 4))
    allowed = RuntimeInputs(x, ha)
    assert allowed.features().shape == (3, 36)
    with pytest.raises(TypeError):
        RuntimeInputs(x, ha, y=np.zeros(3))
    with pytest.raises(ValueError):
        RuntimeInputs(x, np.zeros((3, 6)))
    with pytest.raises(ValueError):
        RuntimeInputs(x * np.nan, ha)


def test_training_household_split_never_separates_relatives():
    from experiments.pcrl_task_directed_release_v1.data import household_roles
    hh = np.repeat([f'h{i}' for i in range(200)], 3)
    roles = household_roles(hh)
    sets = [set(hh[rows]) for rows in roles.values()]
    assert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i+1:])
    assert sum(len(v) for v in roles.values()) == len(hh)
    assert all(len(v) > 0 for v in roles.values())


def test_label_reader_cannot_open_current_evaluation_before_freeze(tmp_path):
    from experiments.pcrl_task_directed_release_v1.data import read_rows
    raw = tmp_path / 'raw.csv'
    raw.write_text('SERIALNO,SPORDER,PWGTP,MIG\na,1,2,1\nb,1,3,2\nc,1,4,3\n')
    with pytest.raises(PermissionError):
        read_rows(raw, np.array([1]), ['MIG'], pool='test', evaluation_permit=None)
    got = read_rows(raw, np.array([0, 2]), ['SERIALNO','MIG'], pool='representation_fit')
    assert got['MIG'].tolist() == [1, 3]
