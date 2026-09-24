"""Aggregate, non-identifying host reports over a finished (or running) units root.

capacity : ENCODER_CAPACITY.json -- per anchor/unit policy disagreement with D17,
           per-round eta/A/within-T32 spread/stochastic rows/re-score/final-bank
           feasibility, closing refit, final selection status, and the
           trained / used / selected summary per family and anchor.
manifest : RUN_MANIFEST.json -- every queue slot with its runner receipt
           (timestamps, wall time, attempts, commit, output hashes), unique vs
           alias laws (audit ALIAS_LEDGER), triggers and technical failures.

Only fields already present in unit receipts are copied (no person rows, ids or
label values; nothing from inner_check or outer).  `capacity --index` optionally
recomputes the final-release nonalias summary for every unit that has a
`release.json`, from the legal inputs of coefficient_split only (no labels).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

TOL = 1e-12
FAMILIES = ("NM1", "NM4", "T32", "DET_SEL", "RD_TASK", "RD_PRIV", "ADV")
_UNIT = re.compile(r"^a([012])_(.+)$")


def _json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _family(name):
    for fam in ("NM1", "NM4", "T32"):
        if name.startswith(fam + "_"):
            return fam
    if name.startswith("DET_SEL"):
        return "DET_SEL"
    if name in ("RD_TASK", "RD_PRIV"):
        return name
    if name.startswith("ADV"):
        return "ADV"
    return name


def _gate(nonalias):
    """Compact copy of channel.nonalias_diagnostic (drops the per-state list)."""
    if not nonalias:
        return None
    within = {k: v for k, v in nonalias["within_t32"].items() if k != "per_state"}
    return {"tv_to_d17": nonalias["tv_to_d17"], "within_t32": within,
            "stochastic_rows": nonalias["deterministic_emission"]["stochastic_rows"],
            "all_rows_one_hot": nonalias["deterministic_emission"]["all_rows_one_hot"],
            "stochastic_weight_fraction": nonalias.get("stochastic_weight_fraction")}


def _differs(gate):
    return None if gate is None else bool(gate["tv_to_d17"]["max"] > TOL)


def _varies(gate):
    return None if gate is None else bool(gate["within_t32"]["max_pairwise_tv"] > TOL
                                          or gate["within_t32"]["max_tv_to_state_mean"] > TOL)


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------

def bank_summary(root):
    root = Path(root)
    pol = _json(root / "policies" / "POLICIES.json") or {}
    ctx = _json(root / "contexts" / "CONTEXTS.json") or {}
    manifest = _json(root / "BANK_COMPLETE.json") or {}
    disagreement = pol.get("disagreement", {}).get("coefficient_split", {})
    return {"status": manifest.get("status"), "smoke": manifest.get("smoke"),
            "policies": {name: {k: d[k] for k in (
                "fraction_differs_from_D17_unweighted", "fraction_differs_from_D17_weighted",
                "within_T32_disagreement_quantiles", "states_with_within_T32_token_variation",
                "affected_unique_households") if k in d} for name, d in disagreement.items()},
            "alias_ledger": pol.get("alias_ledger"),
            "price_groups": pol.get("price_groups"),
            "context_census_coefficient_split": ctx.get("census_coefficient_split")}


def nm_summary(root):
    root = Path(root)
    receipt = _json(root / "COMPLETE.json") or {}
    selection = _json(root / "FINAL_BANK_SELECTION.json") or {}
    checks = {c.get("member", f"round_{c['round']:02d}" if c.get("round") is not None else "D17_witness"): c
              for c in selection.get("final_bank_checks", [])}
    rounds = []
    for path in sorted(root.glob("round_r*/ROUND.json")):
        rec = _json(path)
        r = rec["round"]
        with np.load(path.parent / "PARAMS.npz") as params:
            A = params["A"].tolist()
        sol = rec.get("solution", {})
        check = checks.get(f"round_{r:02d}", {})
        rounds.append({
            "round": r, "eta": sol.get("sparsity", {}).get("eta"), "A": A,
            "sparsity": sol.get("sparsity"), "tau": sol.get("tau"),
            "lp_status": sol.get("status"),
            "nonalias_coefficient_split": _gate(rec.get("nonalias", {}).get("coefficient_split")),
            "nonalias_inner_selection": _gate(rec.get("nonalias", {}).get("inner_selection")),
            "within_t32_projection_rescore": rec.get("within_t32_projection_rescore"),
            "inner_selection_fixed_decoder_task": rec.get("inner_selection_fixed_decoder_task"),
            "final_bank": {k: check.get(k) for k in ("feasible", "maximum_cut_violation",
                                                    "violated_cuts", "worst_cut", "ab_sex_slack_vs_rho")}})
    closing = _json(root / "closing" / "CLOSING.json")
    selected = selection.get("selected_round")
    return {"variant": receipt.get("variant"), "K": receipt.get("K"),
            "complete": receipt.get("status") == "COMPLETE",
            "selection_status": selection.get("status"), "selected_round": selected,
            "selected_member": selection.get("selected_member"),
            "selection_rule": selection.get("selection_rule"),
            "witness_final_bank": checks.get("D17_witness"),
            "closing_refit": None if closing is None else {
                k: closing.get(k) for k in ("round_index", "new_attack_count", "cut_count", "bank_sha256")},
            "rounds": rounds}


def det_summary(root):
    rec = _json(Path(root) / "DET_SEL.json") or {}
    return {"selection_status": "ALL_D17_ASSIGNMENT" if (rec.get("selected") or {}).get("non_d17_contexts") == 0
            else ("DETERMINISTIC_ASSIGNMENT" if rec else None),
            "selected_assignment": (rec.get("selected") or {}).get("assignment"),
            "non_d17_contexts": (rec.get("selected") or {}).get("non_d17_contexts"),
            "assignments_evaluated": rec.get("assignments_evaluated"),
            "feasible_assignments": rec.get("feasible_assignments"),
            "all_d17_assignment_included": rec.get("all_d17_assignment_included"),
            "bank_source": rec.get("bank_source"),
            "nonalias_coefficient_split": _gate(rec.get("nonalias_coefficient_split"))}


def baseline_summary(root):
    """RD / ADV: status and any coefficient_split census already in SELECTED.json."""
    rec = _json(Path(root) / "SELECTED.json") or {}
    rounds = [{"round": c.get("round"), "tau": c.get("tau"),
               "census_coefficient_split": c.get("census_coefficient_split"),
               "final_bank_check": {k: v for k, v in (c.get("final_bank_check") or {}).items()
                                    if k in ("feasible", "max_violation", "maximum_cut_violation")}}
              for c in (rec.get("rounds") or rec.get("candidates") or [])]
    witness = rec.get("witness_selected")
    flag = rec.get("flag")
    return {"variant": rec.get("variant"), "flag": flag, "witness_selected": witness,
            "selection_status": ("WITNESS_FALLBACK" if flag == "WITNESS_FALLBACK" else
                                 "WITNESS_SELECTED" if witness else
                                 "SELECTED_ROUND" if rec else None),
            "selected_round": rec.get("selected_round", rec.get("selected_epoch")),
            "selected_tau": rec.get("selected_tau"), "rounds": rounds}


def release_gate(unit_dir, rows):
    """Nonalias summary of the unit's pinned final law on coefficient_split legal inputs."""
    from . import channel, laws, policies
    law = laws.load(unit_dir)
    q = law(policies.legal_inputs(rows))
    gate, _ = channel.nonalias_diagnostic(q, rows["token_codes"], rows["d17"], rows["weights"],
                                          rows["households"])
    return _gate(gate)


def _levels(family, summary, bank):
    """trained / used / selected (tolerance 1e-12 TV) from what the unit recorded."""
    final_gate = summary.get("final_release_nonalias")
    if family in ("NM1", "NM4", "T32"):
        gates = [r["nonalias_coefficient_split"] for r in summary["rounds"]]
        policy_trained = any(p.get("fraction_differs_from_D17_unweighted", 0) > 0
                             for n, p in (bank or {}).get("policies", {}).items() if n != "D17")
        trained = policy_trained or any(_differs(g) for g in gates if g)
        used = any(_varies(g) for g in gates if g)
        if summary["selection_status"] is None:
            selected = None
        elif summary["selected_round"] is None:
            selected = False
        else:
            chosen = next(r for r in summary["rounds"] if r["round"] == summary["selected_round"])
            selected = _differs(chosen["nonalias_coefficient_split"])
    elif family == "DET_SEL":
        g = summary["nonalias_coefficient_split"]
        trained, used = _differs(g), _varies(g)
        selected = trained
    else:
        census = [r["census_coefficient_split"] for r in summary["rounds"] if r.get("census_coefficient_split")]
        trained = any(c.get("fraction_changed_vs_D17", 0) > 0 for c in census) if census else None
        used = None
        selected = (False if summary["witness_selected"] else
                    (_differs(final_gate) if final_gate else None))
    if final_gate is not None and family not in ("NM1", "NM4", "T32"):
        used = _varies(final_gate) if used is None else (used or _varies(final_gate))
        if selected is None:
            selected = _differs(final_gate)
    return {"trained": trained, "used": used, "selected": selected}


def _unit_dirs(units_root):
    for path in sorted(Path(units_root).iterdir()):
        match = _UNIT.match(path.name)
        if path.is_dir() and match:
            yield int(match.group(1)), match.group(2), path


def capacity(units_root, *, rows_by_anchor=None):
    """Build ENCODER_CAPACITY.json content. rows_by_anchor[a] = coefficient_split legal rows
    plus households, weights and d17 (optional; enables the uniform final-release gate)."""
    anchors = {}
    for anchor, name, path in _unit_dirs(units_root):
        entry = anchors.setdefault(str(anchor), {"units": {}})
        if name == "bank":
            entry["bank"] = bank_summary(path)
            continue
        family = _family(name)
        if family in ("NM1", "NM4", "T32"):
            summary = nm_summary(path)
        elif family == "DET_SEL":
            summary = det_summary(path)
        elif family in ("RD_TASK", "RD_PRIV", "ADV"):
            summary = baseline_summary(path)
        else:
            continue
        if rows_by_anchor and anchor in rows_by_anchor and (path / "release.json").is_file():
            try:
                summary["final_release_nonalias"] = release_gate(path, rows_by_anchor[anchor])
            except Exception as error:  # noqa: BLE001 - recorded, report still written
                summary["final_release_nonalias_error"] = f"{type(error).__name__}: {error}"
        summary["family"] = family
        entry["units"][name] = summary
    levels = {}
    for anchor, entry in anchors.items():
        for name, summary in entry["units"].items():
            levels.setdefault(summary["family"], {}).setdefault(anchor, {})[name] = _levels(
                summary["family"], summary, entry.get("bank"))
    return {"schema": "pcrl-sc-encoder-capacity-v1", "tolerance_tv": TOL,
            "definitions": {"trained": "a fitted policy column or round law differs from D17 on coefficient_split",
                            "used": "a fitted round law varies within a T32 state (within-T32 TV > tol)",
                            "selected": "the final selected release differs from D17 (witness = not selected)"},
            "rows": "coefficient_split and inner_selection aggregates only; no inner_check/outer",
            "anchors": anchors, "summary_by_family_anchor": levels}


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def _outputs_digest(outputs):
    digest = hashlib.sha256()
    for key in sorted(outputs):
        digest.update(f"{key}\0{outputs[key]}\n".encode())
    return digest.hexdigest()


def manifest(units_root, queue_path):
    root = Path(units_root)
    queue = _json(queue_path)
    status = _json(root / "STATUS.json") or {}
    slots, failures, triggers, ledgers = [], [], {}, {}
    for unit in queue["units"]:
        uid = unit["id"]
        receipt = _json(root / "_receipts" / f"{uid}.json")
        state = status.get("units", {}).get(uid, {})
        attempts_dir = root / ".attempts" / uid
        attempts = sorted(p.name for p in attempts_dir.glob("attempt-*")) if attempts_dir.is_dir() else []
        slot = {"id": uid, "depends_on": unit.get("depends_on", []),
                "release": unit.get("release"), "state": state.get("state"),
                "attempt_dirs": attempts, "reason": state.get("reason")}
        if receipt:
            outputs = receipt.get("outputs_sha256", {})
            slot.update({k: receipt.get(k) for k in ("started_utc", "completed_utc", "wall_seconds",
                                                      "attempt", "code_commit", "code_tree_sha256",
                                                      "inputs_sha256")},
                        output_files=len(outputs), outputs_digest_sha256=_outputs_digest(outputs),
                        release_json_sha256=outputs.get("release.json"),
                        complete_json_sha256=outputs.get("COMPLETE.json"))
        slots.append(slot)
        if len(attempts) > (1 if receipt else 0) or state.get("state") in ("FAILED", "REFUSED", "BLOCKED", "RETRYING"):
            failures.append({"id": uid, "state": state.get("state"), "attempt_dirs": attempts,
                             "reason": state.get("reason"), "completed": receipt is not None})
        unit_dir = root / uid
        if uid == "decide_k":
            triggers["K_decision"] = _json(unit_dir / "DECISION.json")
        match = _UNIT.match(uid)
        if match and unit_dir.is_dir():
            name = match.group(2)
            for fname, key in (("FINAL_BANK_SELECTION.json", "status"), ("SELECTED.json", "flag")):
                rec = _json(unit_dir / fname)
                if rec and rec.get(key) in ("WITNESS_FALLBACK", "WITNESS_SELECTED", "WITNESS_SELECTED_BY_RULE"):
                    triggers.setdefault("witness", []).append({"id": uid, "status": rec[key]})
            if name == "bank":
                groups = (_json(unit_dir / "policies" / "POLICIES.json") or {}).get("price_groups", {})
                used = sorted(g for g, v in groups.items() if v.get("fallback_used"))
                if used:
                    triggers.setdefault("price_group_fallbacks", []).append({"id": uid, "groups": used})
            ledger = _json(unit_dir / "ALIAS_LEDGER.json")
            if ledger:
                ledgers[uid] = {"declared_name_count": ledger.get("declared_name_count"),
                                "canonical_release_count": ledger.get("canonical_release_count"),
                                "canonical_ids": ledger.get("canonical_ids"),
                                "declared": ledger.get("declared")}
    states = {}
    for slot in slots:
        states[slot["state"]] = states.get(slot["state"], 0) + 1
    return {"schema": "pcrl-sc-run-manifest-v1", "queue_units": len(slots), "state_counts": states,
            "runner": {k: status.get(k) for k in ("started_utc", "updated_utc", "finished_utc",
                                                  "code_commit", "code_tree_sha256", "workers")},
            "slots": slots, "unique_vs_alias": ledgers, "triggers": triggers,
            "technical_failures_and_retries": failures}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _coefficient_rows(index_path):
    from . import fit_nm
    out = {}
    for anchor in (0, 1, 2):
        role_dict, q_ref, _, _ = fit_nm.load_roles(anchor, index_path)
        rows = role_dict["coefficient_split"]
        out[anchor] = {**{k: rows[k] for k in ("x", "ha", "token_codes", "teacher_p", "residual",
                                               "risk", "households", "weights")}, "d17": q_ref}
        del role_dict
    return out


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=float) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("capacity", "manifest"))
    parser.add_argument("--units-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--queue")
    parser.add_argument("--index", help="capacity: also recompute final-release gates from legal inputs")
    args = parser.parse_args(argv)
    if args.command == "capacity":
        rows = _coefficient_rows(args.index) if args.index else None
        _write(args.out, capacity(args.units_root, rows_by_anchor=rows))
    else:
        if not args.queue:
            parser.error("manifest requires --queue")
        _write(args.out, manifest(args.units_root, args.queue))
    print(json.dumps({"written": args.out}))


if __name__ == "__main__":
    main()
