import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")  # keep HistGradientBoosting from oversubscribing small fixtures
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from stored_model_eval.config import load_protocol  # noqa: E402
from stored_model_eval.guards import FitAuthorization  # noqa: E402


@pytest.fixture
def cfg():
    c = load_protocol(None)
    c["bootstrap"]["n_boot"] = 400
    c["support"]["min_class_support"] = 20
    return c


@pytest.fixture
def auth():
    return FitAuthorization.synthetic_only()


# --------------------------------------------------------------------------------------------------
# CELL-A pilot: one synthetic end-to-end run through the SAME shell runner / CLI path and the frozen
# EFFECTIVE_PROTOCOL (grids, B = 2000 / 20000, seeds), shared by test_16 and test_17.
# --------------------------------------------------------------------------------------------------
import json  # noqa: E402
import subprocess  # noqa: E402

WT = Path(__file__).resolve().parents[2]
RUNNER = WT / "results/combined_stored_model_pilot_v1/scripts/run_pilot.sh"
E2E_UNITS = ["income_prediction__race", "income_prediction__sex", "education_assessment__income",
             "income_prediction__sex__p0_sigma8_seed0", "income_prediction__sex__p0_sigma8_seed1",
             "income_prediction__sex__p0_sigma8_seed2"]


def branch_extra() -> str:
    from stored_model_eval.effective import BRANCH
    from stored_model_eval.pilot import git_branch
    return "" if git_branch(WT) == BRANCH else " --allow-other-branch-for-tests"


def run_shell(env_extra: dict, check=True):
    env = dict(os.environ, **{k: str(v) for k, v in env_extra.items()})
    env["PY"] = sys.executable
    r = subprocess.run(["sh", str(RUNNER)], env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"runner failed ({r.returncode}):\nSTDOUT\n{r.stdout}\nSTDERR\n{r.stderr[-4000:]}")
    return r


@pytest.fixture(scope="session")
def pilot_e2e(tmp_path_factory):
    from stored_model_eval.fixtures import make_pilot_world
    from stored_model_eval.pilot_inputs import build_inputs
    root = tmp_path_factory.mktemp("pilot_e2e")
    w = make_pilot_world(root)
    run = root / "run_v1"
    build_inputs(w["orig"], run / "inputs", w["source"], w["checkpoint"], log=lambda m: None)
    out = root / "tables"
    out.mkdir()
    env = {"RUN_DIR": run, "LOCK": root / "PILOT_LOCK_v2.json", "OUT": out, "FEATURES": w["features"],
           "PILOT_EXTRA_ARGS": "--synthetic" + branch_extra()}
    logs = {"lock": run_shell({**env, "STAGE": "lock"})}
    logs["dry"] = run_shell({**env, "STAGE": "run"})
    logs["exec"] = run_shell({**env, "STAGE": "run", "EXECUTE": 1, "UNITS": ",".join(E2E_UNITS)})
    complete_before = {u: (run / "units" / u / "COMPLETE.json").read_text() for u in E2E_UNITS}
    logs["resume"] = run_shell({**env, "STAGE": "run", "EXECUTE": 1, "RESUME": 1, "UNITS": ",".join(E2E_UNITS)})
    logs["infer"] = run_shell({**env, "STAGE": "infer"})
    logs["report"] = run_shell({**env, "STAGE": "report"})
    return {"root": root, "world": w, "run": run, "units": run / "units", "out": out, "env": env, "logs": logs,
            "infer": json.loads((run / "infer" / "PILOT_INFER.json").read_text()),
            "complete_before_resume": complete_before}
