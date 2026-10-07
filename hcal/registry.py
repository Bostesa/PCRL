"""Pre-science registries of the held-out calibration study (hcal; role A): writes CALIBRATION_RULES.json,
SELECTION_RULES.json, PRIMARY_FAMILY.json, LABEL_TRUTH_TABLE.json and QUEUE_MANIFEST.json from the code that executes
them (single source of truth). No data is loaded.

    PYTHONPATH=. <python> -m hcal.registry
"""
from __future__ import annotations

import json

from hcal import family as FAM
from hcal import ids as I
from hcal import select as SEL


def calibration_rules():
    from hcal import calib as C
    return {"schema": "hcal-calibration-rules-v1", "module": "hcal/calib.py",
            "rows": {"H-*": "CALIBRATION_HELDOUT representatives (2,000; one per group, smallest row id)",
                     "T-TOKEN32": "CALIBRATION_TRAIN_MATCHED representatives (2,000; inside OSF_DEFENSE_FIT)",
                     "labels": "task labels only; SEX never enters a calibration objective"},
            "H-TOKEN32": {"objective": "sum_i [-log q(Y_i) + 0.5 ||q - onehot(Y_i)||^2] + 32 KL(mu_t || q) over the "
                                       "class-dominant simplex, q = (u + eps 1 + eps e_d) / (1 + (K+1) eps), eps 1e-12",
                          "prior": "mu_t = admitted token_S / token_n (unsmoothed U mean on OSF_DEFENSE_FIT), fixed",
                          "solver": "lra.decoder internals wrapped unchanged with A = y + 32 mu_t (hcal.calib."
                                    "token32_solve); bitwise equal to lra.decoder.solve_batch when mu_t = S / n",
                          "fallbacks": {"n_cal = 0": "q0_t exactly, NO_CALIBRATION_OBSERVATIONS",
                                        "reserved empty token (n_fit = 0)": "q0_t exactly, "
                                                                            "RESERVED_EMPTY_ORIGINAL_FALLBACK"},
                          "kappa": C.KAPPA, "eps": C.EPS, "certificate_tolerances": C.TOLERANCES["token32"]},
            "T-TOKEN32": {"same as": "H-TOKEN32", "partitions": list(I.DIAGNOSTIC_PARTITIONS),
                          "units": "4 partitions x 3 seeds = 12 partition-pair fits", "role": "diagnostic only; never "
                                                                                           "a nominee or T*"},
            "H-GLOBAL-TEMP": {"map": "q_alpha(t)_k = softmax_k(alpha log q0_tk)", "bounds": [C.TEMP_LO, C.TEMP_HI],
                              "objective": "mean unclipped NLL on the calibration representatives (stable LSE)",
                              "solver": f"bounded bisection on the convex NLL derivative (<= {C.TEMP_BISECT_ITERS} "
                                        "halvings or adjacent floats; the bracket end with the smaller |g|)",
                              "certificate": {"grad_tol_rel": C.TEMP_GRAD_TOL, "nll_vs_identity_tol_rel":
                                              C.TEMP_NLL_TOL, "boundary": "KKT sign at 0.25 / 4"},
                              "identity": "alpha == 1 returns q0 exactly", "scope": "one alpha per task, map and seed"},
            "H-CLASS-TEMP": {"map": "the same scalar per predicted class, shared over its tokens", "min_rows":
                             C.CLASS_MIN, "fallback": "alpha = 1 for a class with fewer than 50 representatives "
                                                      "(including absent classes)", "max_parameters": {"income": 2,
                                                                                                        "occupation": 6}},
            "continuous U": {"families": ["H-GLOBAL-TEMP", "H-CLASS-TEMP"], "log_input": "alpha != 1: p' = max(p, "
                             "1e-12) renormalised, softmax(alpha log p'); alpha == 1 returns p exactly",
                             "classes": "U's predicted class (teacher decision)", "decision_check": "argmax(q) == "
                             "teacher decision on every row; any mismatch raises"},
            "scoring": {"clip": C.LOSS_CLIP, "loss": "natural log", "brier": "multiclass sum_k (p_k - 1[y = k])^2",
                        "recorded": "unclipped fitting NLL, clipped scoring NLL and the clipped-probability count"},
            "no_grid": "no kappa, temperature-bound, lambda, capacity or sample-split sweep",
            "prior_art": "temperature scaling: Guo, Pleiss, Sun and Weinberger, ICML 2017 (not a new algorithm)"}


def queue_manifest():
    P = I.partitions()
    return {"schema": "hcal-queue-manifest-v1",
            "logical_units": {
                "partition_seed_units": {"count": len(P) * 3, "rule": "57 partition pairs x 3 source seeds"},
                "new_heldout_decoder_pair_units": {"count": 3 * len(P) * 3, "rule": "3 held-out families x 171"},
                "train_matched_diagnostic_units": {"count": len(I.DIAGNOSTIC_PARTITIONS) * 3},
                "u_calibration_pair_units": {"count": 2 * 3},
                "admitted_original_release_controls": {"count": len(I.original_ids()) * 3, "rule": "84 x 3"},
                "mean_decoded_lra_controls": {"count": len(I.lra_partitions()) * 3,
                                              "note": "legacy means are aliases of the admitted D0 releases"},
                "privacy_bank_fits": {"count": "one fresh bank per AUDITED partition/seed (audit plan, after the "
                                               "utility stage); never per decoder variant", "max": len(P) * 3}},
            "physical_units": {
                "cal__s{k}__<partition>": {"stage": "calibrate", "count": len(P) * 3, "depends_on": "admission bank"},
                "calU__s{k}": {"stage": "calibrate", "count": 3, "depends_on": "admitted teacher"},
                "util__s{k}": {"stage": "utility", "count": 3, "depends_on": "every cal__ / calU__ unit of the seed"},
                "audit_plan.json": {"stage": "utility", "count": 1, "depends_on": "util__s0-2 (Ucal*, gates)"},
                "fam__s{k}__<partition>, com__s{k}__<partition>": {"stage": "audit", "count": "2 x audited x 3",
                                                                   "depends_on": "audit plan, cal__ unit, admitted "
                                                                                 "lra aud__ units"},
                "com__s{k}__SRC_U": {"stage": "compose", "count": 3, "depends_on": "every com__ / fam__ unit of the "
                                                                                    "seed, admitted aud__tea__s{k}__U"},
                "controls.json": {"stage": "controls", "count": 1, "depends_on": "CONTROL_PLAN.json, cal__ units"},
                "selection.json": {"stage": "select", "count": 1, "depends_on": "util, com, controls"},
                "replay.json": {"stage": "replay", "count": 1, "depends_on": "selection"},
                "oprob__s{k}__<release>, oatt__s{k}__<partition key>": {"stage": "assess (after EVALUATION_LOCK)",
                                                                       "count": "scored releases x 3, scored "
                                                                                "partitions x 3"}},
            "shards": "calibrate / audit run as two shards under the 2-slot semaphore; every unit atomic and "
                      "resumable (jcv.finalize.save_unit)",
            "attacker_and_control_counts": "generated from the frozen slate at run time (records), never invented "
                                           "here"}


def control_plan():
    from hcal import controls as CT
    return {"schema": "hcal-control-plan-v1", "module": "hcal/controls.py", "plan": CT.control_plan(),
            "jobs": [list(j) for j in CT.control_jobs(CT.control_plan())], "limits": CT.CONTROL_LIMITS,
            "realised_null_threshold_required": CT.SOURCE_NULL_THRESHOLD, "threshold_tol": CT.THRESHOLD_TOL,
            "views": {"codes": "the fresh complete view of each control partition with its FULL registered table set "
                               "(hcal.bank.fresh_views); fit ATTACK_FIT_NEW with permuted SEX, select on INNER half A, "
                               "evaluate on half B", "rot": "SRC|U interface through lra.audit.controls_for_release "
                                                            "unchanged (fit AUDIT_FIT)"},
            "units": "one control audit per (job, check); planted releases in memory / temporary directories only, "
                     "never saved as units, never in a candidate bank",
            "seeds": {"control_seed": CT.CONTROL_SEED, "null_calibration": "CONTROL_SEED + 100 + rep, rep 0-4"},
            "timing": "after SCIENCE_LOCK; must pass before nomination / EVALUATION_LOCK; a failure is a technical "
                      "validity issue, never a negative method outcome; synthetic tests never replace these"}


def main():
    out = {"CONTROL_PLAN.json": control_plan(), "CALIBRATION_RULES.json": calibration_rules(), "SELECTION_RULES.json": SEL.selection_rules(),
           "PRIMARY_FAMILY.json": {"schema": "hcal-primary-family-v1", "executable": "hcal/family.py, hcal/infer.py",
                                   "size": FAM.PRIMARY_SIZE, "alpha": FAM.ALPHA, "z": FAM.Z_PRIMARY,
                                   "z_definition": "statistics.NormalDist().inv_cdf(1 - 0.05 / (2 * 23))",
                                   "B": FAM.B, "bootstrap_seed": FAM.BOOT_SEED,
                                   "bootstrap": "paired multinomial bootstrap over OSF_DEVELOPMENT_ASSESSMENT "
                                                "exact-record groups; identical draws for every statistic, arm and "
                                                "model seed; SE = sd (ddof 1); interval = point +- z SE",
                                   "aggregation": "equal-weight mean over model seeds 0-2 of the per-seed paired "
                                                  "statistic; recovery = SEX AUC of the inner-AUC-selected common-bank "
                                                  "attacker, mean over attacker seeds 0-2",
                                   "slots": FAM.PRIMARY, "supplementary": {
                                       "partitions": list(FAM.SUPPLEMENTARY_PARTITIONS),
                                       "contrasts": [list(c) for c in FAM.SUPPLEMENTARY_CONTRASTS],
                                       "inference": "nominal 95%, descriptive, counted separately"}},
           "LABEL_TRUTH_TABLE.json": {"schema": "hcal-label-truth-table-v1", "executable": "hcal.family.overall_label",
                                      "precedence": list(FAM.LABELS), "cases": FAM.truth_table()},
           "QUEUE_MANIFEST.json": queue_manifest()}
    for name, body in out.items():
        (I.PKG / name).write_text(json.dumps(body, indent=1, default=str) + "\n")
    print("wrote", sorted(out))


if __name__ == "__main__":
    main()
