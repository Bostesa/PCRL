#!/usr/bin/env python3
"""Own mutation self-check of replay_lra.py (role E). Each mutant breaks ONE registered lra rule in a temporary copy
(written to the directory given as argv[1], then deleted); the PHASE_0 self-tests of every mutant must not PASS.
Imports nothing of the study. Run under the semaphore:
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -P <WORKTREE>/lra/sema.py --label E:phase0-mutation -- \\
        env OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python <this file> <scratch dir>
Writes MUTATION_SELFCHECK.json next to this file."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "replay_lra.py"
M = {
    "R1_missing_record_as_infeasible": ('    if not isinstance(rec, dict):\n        return LRA_FIT_RECORD_TECH, None',
                                        '    if not isinstance(rec, dict):\n        return None, False'),
    "R2_independent_minima": ('invent_pairing=False):\n    """Exact release aliases', 'invent_pairing=True):\n    """Exact release aliases'),
    "R3_untrained_credited": ('"privacy_training_credited": lra_arm(rep) in LRA_PRIVACY_TRAINED_ARMS and not untrained,',
                              '"privacy_training_credited": lra_arm(rep) in LRA_PRIVACY_TRAINED_ARMS,'),
    "R4_missing_guard_drops_fallback": (
        "def my_selection_lra(rows, reverse=False, private_extra=(), guard_overrides=None, drop_missing_guard_fallback=False,",
        "def my_selection_lra(rows, reverse=False, private_extra=(), guard_overrides=None, drop_missing_guard_fallback=True,"),
    "R6_any_count_accepted": ('    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and int(v) > 0',
                              '    return v is not None'),
    "R7_aliases_by_config_id": ('def lra_role_aliases(rows, resolved, by_config_id=False):',
                                'def lra_role_aliases(rows, resolved, by_config_id=True):'),
    "R10_gate_accepts_GATE_MET": ('LRA_GATE_VERDICTS = ("ENGINEERING_READY", "ENGINEERING_BLOCKED")',
                                  'LRA_GATE_VERDICTS = ("ENGINEERING_READY", "ENGINEERING_BLOCKED", "GATE_MET", "GATE_NOT_MET")'),
    "R13_acceptance_escape": (
        "def lra_assessment_open_ok(technical_valid, controls_ok, admission_ok, gate, science_lock_ok, evaluation_lock_ok,\n"
        "                           accept_technical_failure=False):",
        "def lra_assessment_open_ok(technical_valid, controls_ok, admission_ok, gate, science_lock_ok, evaluation_lock_ok,\n"
        "                           accept_technical_failure=True):"),
    "R14_incomplete_before_Q": (
        '        elif q == "PASS":\n            head = "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"\n'
        '        elif any(v == "INCOMPLETE_OR_INVALID" for v in shown.values()):\n            head = "INCOMPLETE_OR_INVALID"',
        '        elif any(v == "INCOMPLETE_OR_INVALID" for v in shown.values()):\n            head = "INCOMPLETE_OR_INVALID"\n'
        '        elif q == "PASS":\n            head = "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"'),
    "mean_seed_eligibility": ('def lra_config_row(cid, per_seed, seed_average=False):',
                              'def lra_config_row(cid, per_seed, seed_average=True):'),
    "class_d1_not_in_T_pool": ('L_d0("CLASS"), L_d1("CLASS"), "REF|F0"]', 'L_d0("CLASS"), "REF|F0"]'),
    "infeasible_fit_in_C_pair_pool": (
        '    L["C_pair*"] = [c_ for c_ in L["C_pair*"] if (rows.get(c_) or {}).get("fit_feasible") is not False]\n    st = {}',
        '    st = {}'),
    "old_bootstrap_seed": ('B_BOOT, BOOT_SEED = 1999, 20261010', 'B_BOOT, BOOT_SEED = 1999, 20261009'),
    "cbp_headroom_reintroduced": (
        '    return task_check(u, uU, headroom={"ll_excess": GATE["ll_excess"], "brier_excess": GATE["brier_excess"]})',
        '    r = task_check(u, uU)\n    r["ordinary_ok"] = r["ordinary_ok"] and r["headroom_ok"]\n    return r'),
    "z_bonferroni_wrong": ('Z_LRA = 3.2048452050105634', 'Z_LRA = 3.0233414397'),
    "trace_search_margin_zero": ('TR_BUDGET_MARGIN = 1e-10 ', 'TR_BUDGET_MARGIN = 0.0 '),
    "trace_no_strict_improvement": ('                if not nobj < obj:\n', '                if False:\n'),
    "trace_no_stats_hash_check": ('                        if rh[0] != sa or rh[1] != sb:\n', '                        if False:\n'),
    "trace_partner_held_to_budget": ('return [("seq1", [a], [a] if k else []), ("seq2", [b], [b] if k else [])]',
                                     'return [("seq1", [a], [1, 2] if k else []), ("seq2", [b], [b] if k else [])]'),
}


def main():
    outdir = Path(sys.argv[1])
    src = SRC.read_text()
    res = {}
    for name, (a, b) in M.items():
        if src.count(a) != 1:
            res[name] = {"status": "MUTANT_NOT_APPLICABLE"}
            continue
        p = outdir / f"mut_{name}.py"
        p.write_text(src.replace(a, b))
        r = subprocess.run([sys.executable, str(p), "--phase", "0", "--no-write"], capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        try:
            out = json.loads(r.stdout)
            st = out["summary"]["status_by_check"]["selftests"]
            fl = [f["path"] for f in out["flagged"]][:4]
        except Exception:  # noqa: BLE001
            st, fl = "CRASHED", [r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "no output"]
        res[name] = {"selftests": st, "killed": st != "PASS", "flagged": fl}
        p.unlink()
    rep = {"schema": "lra-verifier-mutation-selfcheck-v1", "verifier_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "mutants": len(M), "killed": sum(1 for v in res.values() if v.get("killed")), "results": res,
           "status": "PASS" if all(v.get("killed") for v in res.values()) else "FAIL"}
    (HERE / "MUTATION_SELFCHECK.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("mutants", "killed", "status")}))


if __name__ == "__main__":
    main()
