"""Host reports on a synthetic units root (real fit_nm formats, fake runner files)."""
import json

import numpy as np

from experiments.pcrl_shared_context_release_v1 import fit_nm, reports
from tests.pcrl_shared_context_release_v1.test_fit_nm import synthetic_roles


def test_capacity_and_manifest_are_aggregate(tmp_path):
    units = tmp_path / "private" / "units"
    roles, d17, hist = synthetic_roles(seed=3)
    fit_nm.build_bank(0, roles, d17, hist, "0" * 64, units / "a0_bank", target_roles=("A/SEX", "AB/SEX"))
    fit_nm.run_unit("NM1_U", 0, roles, d17, hist, units / "a0_bank", units / "a0_NM1_U", rounds=1)
    fit_nm.run_unit("T32_U", 0, roles, d17, hist, units / "a0_bank", units / "a0_T32_U", rounds=0)
    fit_nm.run_det_sel("DET_SEL1", units / "a0_NM1_U", units / "a0_DET_SEL1", role_dict=roles, q_ref=d17)
    (units / "a0_RD_PRIV").mkdir()
    (units / "a0_RD_PRIV" / "SELECTED.json").write_text(json.dumps({
        "variant": "RD_PRIV", "flag": "WITNESS_FALLBACK", "witness_selected": True, "selected_round": None,
        "rounds": [{"round": 0, "tau": .002, "census_coefficient_split": {"fraction_changed_vs_D17": .2},
                    "final_bank_check": {"feasible": False, "max_violation": .01}}]}))
    coef = roles["coefficient_split"]
    rows = {0: {**{k: coef[k] for k in ("x", "ha", "token_codes", "teacher_p", "residual", "risk",
                                        "households", "weights")}, "d17": d17}}
    cap = reports.capacity(units, rows_by_anchor=rows)
    a0 = cap["anchors"]["0"]
    assert set(a0["units"]) == {"NM1_U", "T32_U", "DET_SEL1", "RD_PRIV"}
    nm = a0["units"]["NM1_U"]
    assert len(nm["rounds"]) == 2 and nm["closing_refit"]["round_index"] == 2
    assert nm["selection_status"] in ("SELECTED_ROUND", "WITNESS_SELECTED", "WITNESS_FALLBACK")
    assert "A" in nm["rounds"][0] and "final_bank" in nm["rounds"][0]
    assert "final_release_nonalias" in nm and "final_release_nonalias" in a0["units"]["DET_SEL1"]
    assert a0["units"]["T32_U"]["rounds"][0]["eta"] == 0.
    levels = cap["summary_by_family_anchor"]
    assert levels["RD_PRIV"]["0"]["RD_PRIV"] == {"trained": True, "used": None, "selected": False}
    assert set(levels["NM1"]["0"]["NM1_U"]) == {"trained", "used", "selected"}
    assert "policies" in a0["bank"] and a0["bank"]["alias_ledger"]["retained"][0] == "D17"
    text = json.dumps(cap)
    assert "inner_check_fixed_decoder_task" not in text and "labels" not in text
    assert not any(str(i) in text for i in coef["ids"][:20])
    # runner files
    queue = tmp_path / "QUEUE.json"
    queue.write_text(json.dumps({"schema": "pcrl-sc-queue-v1", "units": [
        {"id": "a0_bank"}, {"id": "a0_NM1_U", "depends_on": ["a0_bank"]}, {"id": "a0_RD_PRIV"}]}))
    (units / "_receipts").mkdir()
    (units / "_receipts" / "a0_bank.json").write_text(json.dumps({
        "started_utc": "t0", "completed_utc": "t1", "wall_seconds": 5.0, "attempt": 2,
        "code_commit": "c" * 40, "outputs_sha256": {"BANK_COMPLETE.json": "ab"}}))
    (units / ".attempts" / "a0_bank" / "attempt-1").mkdir(parents=True)
    (units / ".attempts" / "a0_bank" / "attempt-2").mkdir(parents=True)
    (units / "STATUS.json").write_text(json.dumps({"code_commit": "c" * 40, "units": {
        "a0_bank": {"state": "COMPLETE"}, "a0_NM1_U": {"state": "RUNNING"},
        "a0_RD_PRIV": {"state": "FAILED", "reason": "technical: exit 1"}}}))
    man = reports.manifest(units, queue)
    assert man["queue_units"] == 3 and man["state_counts"] == {"COMPLETE": 1, "RUNNING": 1, "FAILED": 1}
    bank_slot = man["slots"][0]
    assert bank_slot["attempt"] == 2 and bank_slot["output_files"] == 1
    failed = {f["id"] for f in man["technical_failures_and_retries"]}
    assert failed == {"a0_bank", "a0_RD_PRIV"}
    assert any(t["id"] == "a0_RD_PRIV" for t in man["triggers"]["witness"])
    out = tmp_path / "ENCODER_CAPACITY.json"
    reports._write(out, cap)
    assert json.loads(out.read_text())["schema"] == "pcrl-sc-encoder-capacity-v1"
    assert np.isfinite(nm["rounds"][0]["nonalias_coefficient_split"]["within_t32"]["V_weighted"])
