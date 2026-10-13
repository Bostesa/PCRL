from pathlib import Path
import json,sys,pytest
from .scheduler import assert_active_from_env
assert_active_from_env(stage='admission')
status=pytest.main(['-q','occupation_bottleneck/test_geometry_gradient.py','occupation_bottleneck/test_pilot_rounding.py','occupation_bottleneck/test_access_guards.py'])
if status:raise SystemExit(status)
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as f:json.dump({'status':'PASS','tests_exit':status},f)
