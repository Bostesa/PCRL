import pytest
from compressed_income_safeguard.admission import admitted_path
@pytest.mark.parametrize('path',['/tmp/INNER_SELECTION/sex.npz','/tmp/INNER_SELECTION/income.npz','/tmp/custody/INNER_SELECTION_SEX.npz','/tmp/HEAD_VALIDATION/income.npz','/tmp/all_role/source.npz','/tmp/old_assessment/prediction.npz'])
def test_forbidden_before_open(path):
 with pytest.raises(PermissionError):admitted_path(path)
def test_features_permitted():
 from pathlib import Path
 path=Path.cwd()/'fixtures/INNER_SELECTION/features.npz'
 assert admitted_path(path)==path
