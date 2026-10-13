from pathlib import Path
import json,sys,pytest
from .scheduler import assert_active_from_env
assert_active_from_env(stage='admission')
status=pytest.main(['-q','compressed_income_safeguard/test_geometry_gradient.py','compressed_income_safeguard/test_pilot_rounding.py','compressed_income_safeguard/test_access_guards.py'])
if status:raise SystemExit(status)
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as f:json.dump({'status':'PASS','tests_exit':status},f)
