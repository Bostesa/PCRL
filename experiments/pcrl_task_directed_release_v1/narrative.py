"""Render final prose from explicit, accepted public aggregates; never load data.

This module does not run inference, validate private artifacts, or discover result
files. The caller supplies accepted JSON objects and their original-file hashes.
The evidence producer remains responsible for replaying the registered endpoint
checks. Here we check source pins and formula consistency and project a small
public schema, so unknown/private fields never enter the prose or manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


REPORT_NAMES = ("RESEARCH_DECISION.md", "PAPER_ADDENDUM.md", "VALIDATION.md")
INPUT_NAMES = {"evidence": "EVIDENCE.json", "prediction_assessment": "PREDICTION_ASSESSMENT.json",
               "equivalence": "EXACT_EQUIVALENCE.json", "numerical_versions": "NUMERICAL_VERSION_INDEX.json",
               "verification": "PARALLEL_VERIFICATION.json"}
WEIGHTS = ("unweighted", "PWGTP")
ROUTES = ("utility_first", "protection_first")
VERSION_HASHES = ("original_map_receipt_sha256", "original_solution_sha256", "original_audit_receipt_sha256",
                  "original_registry_sha256", "claim_sha256", "retry_receipt_sha256", "installation_receipt_sha256",
                  "current_map_receipt_sha256", "current_solution_sha256", "current_audit_receipt_sha256",
                  "current_registry_sha256")
VERSION_COUNTS = tuple(p + k for p in ("original_", "current_")
                       for k in ("audit_role_units", "new_role_fits", "reused_role_audits"))
EXECUTION_COUNTS = ("nominal_map_units", "nominal_release_anchor_units", "scheduled_release_anchor_units",
    "scheduled_role_audit_units", "resource_unscheduled_release_anchor_units", "scheduled_evaluation_units",
    "accepted_map_units", "accepted_audit_units", "accepted_evaluation_units",
    "incomplete_audit_units", "incomplete_evaluation_units", "accepted_role_audit_units", "new_role_fit_units",
    "reused_role_audit_units", "registered_numerical_retry_slots", "completed_numerical_retry_attempts",
    "installed_numerical_replacements", "retained_original_after_rejected_retry", "preserved_original_map_versions",
    "preserved_original_audit_versions", "repeated_audit_units", "total_accepted_map_versions",
    "total_accepted_audit_versions", "total_role_audit_units_with_repeats", "total_new_role_fit_units_with_repeats")
EQUIVALENCE_COUNTS = ("nominal_map_units", "accepted_verified_map_units", "unique_literal_kernel_groups",
                      "unique_reduced_kernel_groups", "invalid_map_units", "unfinished_map_units")


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected an accepted SHA256 pin")
    return value


def _name(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,240}", value) is None:
        raise ValueError("Invalid public identifier")
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite aggregate number")
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative aggregate count")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pointer(*parts):
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _fmt(value):
    return "unavailable" if value is None else f"{value:.9g}" if isinstance(value, float) else str(value)


def _table(headers, rows):
    if not rows:
        return "Pending: no accepted aggregate rows supplied.\n"
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "".join(
        "| " + " | ".join(_fmt(v) for v in row) + " |\n" for row in rows)


def _formula(value, bounds):
    if not isinstance(value, dict) or type(value.get("passed")) is not bool:
        raise ValueError("A supplied adjusted formula needs a boolean verdict")
    passed = value["passed"]
    if "endpoint" in value:
        endpoint = _name(value["endpoint"])
        _name(value["check"])
        if endpoint not in bounds:
            raise ValueError("Adjusted formula references a missing endpoint")
        return passed, {endpoint}, "passed" if passed else "did_not_pass"
    kind = value.get("kind")
    if kind == "blocked":
        if passed:
            raise ValueError("Blocked adjusted formula cannot pass")
        return False, set(), "blocked"
    if kind not in ("all", "any") or not isinstance(value.get("clauses"), list) or not value["clauses"]:
        raise ValueError("Invalid adjusted formula tree")
    children = [_formula(c, bounds) for c in value["clauses"]]
    if passed != (all(c[0] for c in children) if kind == "all" else any(c[0] for c in children)):
        raise ValueError("Inconsistent adjusted formula verdict")
    return passed, set().union(*(c[1] for c in children)), "passed" if passed else "did_not_pass"


def _evidence(value):
    result = {"selection_sha256": None, "claims": [], "bounds": {}, "tasks": [], "sensitive": [],
              "nominees": [], "execution": {}, "complete_configurations": 0, "incomplete_configurations": 0,
              "inference_supplied": False, "family_size": None}
    if value is None:
        return result
    if value.get("phase") != "evaluation" or value.get("pool") != "test" or value.get("selection_frozen") is not True:
        raise ValueError("Narratives require frozen evaluation evidence")
    if "cross_anchor_scope" in value:
        expected = {"globally_unseen_people": False, "globally_untouched_labels": False,
                    "retraining_uncertainty_included": False, "bounds_conditional_on_fitted_models": True}
        if any(value["cross_anchor_scope"].get(k) is not v for k, v in expected.items()):
            raise ValueError("Evidence scope contradicts the disclosed cross-anchor design")
    result["selection_sha256"] = _hash(value["selection_sha256"])
    for name, row in sorted(value.get("configurations", {}).items()):
        _name(name)
        if row.get("status") != "complete":
            result["incomplete_configurations"] += 1
            continue
        result["complete_configurations"] += 1
        task = row["task"]
        for w in WEIGHTS:
            record = {"configuration": name, "weighting": w,
                      "pointer": _pointer("configurations", name, "task")}
            for field in ("fixed", "independent", "selected", "loss_delta_H", "loss_delta_J"):
                metric = task.get(field)
                record[field] = None if metric is None else _number(metric[w])
            result["tasks"].append(record)
        for role, by_weight in sorted(row.get("primary_sensitive", {}).items()):
            if role not in ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P"):
                raise ValueError("Unexpected primary sensitive role")
            for w in WEIGHTS:
                record = {"configuration": name, "role": role, "weighting": w,
                          "pointer": _pointer("configurations", name, "primary_sensitive", role, w)}
                record.update({k: None if by_weight[w][k] is None else _number(by_weight[w][k]) for k in
                               ("selected_ce", "recovery_over_H", "recovery_increment_over_J")})
                result["sensitive"].append(record)
    for key in EXECUTION_COUNTS:
        if key in value.get("execution", {}):
            result["execution"][key] = _count(value["execution"][key])
    routes = value.get("selected_routes", {})
    result["nominees"] = sorted({_name(r["nominee"]) for r in routes.values() if r.get("nominee") is not None})
    inference = value.get("selected_inference", {})
    if inference.get("status") in (None, "not_supplied"):
        return result
    if inference["status"] != "supplied_adjusted_results":
        raise ValueError("Unknown adjusted inference status")
    if inference.get("provenance", {}).get("selection_sha256") != result["selection_sha256"]:
        raise ValueError("Adjusted results do not match the frozen selection")
    bounds = inference["bounds"]
    result["family_size"] = _count(inference["family_size"])
    if len(bounds) != result["family_size"]:
        raise ValueError("Adjusted family size differs from supplied endpoints")
    for name, interval in bounds.items():
        _name(name)
        row = {k: _number(interval[k]) for k in ("estimate", "lower", "upper")}
        if not row["lower"] <= row["estimate"] <= row["upper"]:
            raise ValueError("Inconsistent adjusted interval")
        result["bounds"][name] = row
    for group in ("main", "attribution", "diagnostic"):
        for route, claims in inference.get(group, {}).items():
            _name(route)
            for name, formula in claims.items():
                _name(name)
                passed, endpoints, status = _formula(formula, bounds)
                if group == "main" and name == "competitive" and passed:
                    selected = routes.get(route, {})
                    if (selected.get("screen_passed") is not True
                        or selected.get("competitive_validation_eligible") is not True
                        or selected.get("descriptive_only") is True
                        or selected.get("incomplete_required_families") != []
                        or not selected.get("eligible_comparator_families")):
                        raise ValueError("Competitive pass conflicts with selection eligibility/completeness")
                result["claims"].append({"group": group, "route": route, "claim": name, "passed": passed,
                    "status": status, "endpoints": sorted(endpoints),
                    "pointer": _pointer("selected_inference", group, route, name)})
    result["inference_supplied"] = True
    return result


def _predictions(value):
    if value is None:
        return []
    if value.get("scientific_gate") is not False or value.get("inference_performed") is not False:
        raise ValueError("Prediction assessments must remain descriptive, outside scientific gates")
    records = value["predictions"]
    if set(records) != {f"P{i}" for i in range(1, 8)}:
        raise ValueError("Retain all seven subjective predictions")
    rows = []
    for name in sorted(records):
        row = records[name]; probability = _number(row["probability"])
        status = row["status"]; expected = {"supported": True, "refuted": False, "unassessed": None}
        if not 0 <= probability <= 1 or status not in expected or row.get("value") is not expected[status]:
            raise ValueError("Inconsistent subjective prediction assessment")
        complete = row["coverage"].get("complete")
        if type(complete) is not bool or (status == "refuted" and not complete):
            raise ValueError("Incomplete prediction coverage cannot refute a bet")
        rows.append({"prediction": name, "probability": probability, "status": status, "coverage_complete": complete,
                     "pointer": _pointer("predictions", name)})
    return rows


def _versions(value):
    if value is None:
        return None
    if (value.get("schema") != 1 or type(value.get("registered")) is not bool
        or value.get("coverage_complete") is not True or value.get("new_nominal_configurations") != 0
        or not isinstance(value.get("records"), list)
        or _count(value["registered_slots"]) != len(value["records"])):
        raise ValueError("Invalid completed numerical-version receipt summary")
    registered = value["registered"]
    pin = _hash(value["registration_sha256"]) if registered else None
    if not registered and (value["records"] or value.get("registration_sha256") is not None):
        raise ValueError("Unregistered numerical version records")
    records = []; seen = set()
    for row in value["records"]:
        name = _name(row["configuration"]); anchor = _count(row["anchor"])
        if anchor not in (0, 1, 2) or (name, anchor) in seen:
            raise ValueError("Invalid or duplicate numerical slot")
        seen.add((name, anchor)); installed = row["installation_decision"] == "installed_accepted_retry"
        if (row["installation_decision"] not in ("installed_accepted_retry", "retained_original")
            or row.get("attempts") != 1 or row.get("retry_completed") is not True
            or row.get("retry_accepted") is not installed or row.get("reaudit_completed") is not installed):
            raise ValueError("Inconsistent numerical retry/installation/re-audit status")
        clean = {"configuration": name, "anchor": anchor, "installation_decision": row["installation_decision"],
                 "reaudit_completed": installed}
        clean.update({key: _hash(row[key]) for key in VERSION_HASHES})
        clean.update({key: _count(row[key]) for key in VERSION_COUNTS})
        for prefix in ("original_", "current_"):
            if (clean[prefix + "audit_role_units"] != 16
                or clean[prefix + "new_role_fits"] + clean[prefix + "reused_role_audits"] != 16):
                raise ValueError("Numerical-version audit coverage must retain all sixteen roles")
        records.append(clean)
    return {"registered": registered, "registration_sha256": pin, "registered_slots": len(records),
            "new_nominal_configurations": 0, "records": records}


def _verification(value, selection_sha):
    if value is None:
        return {"status": "pending", "passed": False}
    counts = {k: _count(value[k]) for k in ("planned_units", "passed_units", "failed_or_incomplete_units")}
    pin = value.get("global_selection", {}).get("selection_sha256")
    if pin is not None:
        _hash(pin)
        if selection_sha is not None and pin != selection_sha:
            raise ValueError("Verification uses another frozen selection")
    passed = value.get("passed") is True
    if passed and (value.get("all_passed") is not True or value.get("source_selection_closure_unchanged") is not True
                   or counts["planned_units"] == 0 or counts["passed_units"] != counts["planned_units"]
                   or counts["failed_or_incomplete_units"] != 0 or pin is None):
        raise ValueError("Verification pass has incomplete or unpinned coverage")
    return {"status": "passed" if passed else "failed_or_incomplete", "passed": passed, **counts,
            "selection_sha256": pin}


SCOPE = """This is a supervised residence release study. The teacher and finite-channel costs use residence labels; the risk refinement additionally fits protected-label predictors. J is an inherited representation without that residence supervision. A gain over J alone therefore does not isolate a channel-design benefit. Current label-matched controls, including the original teacher-pool and the mechanism40/union88 supervised LEACE and SPLINCE supplements, determine the registered competitive comparison. The union pool describes available fitting rows, not a proved superset of every method's actual protected-label observations.

Ttask and Trisk both refine the 32-cell T0 code and use at most 64 input cells. Coarse row copying proves family inclusion under the same action dictionary, empirical laws, costs and budgets; it does not establish a held-out improvement. Equal-size task/risk refinements are not nested. Attribute a risk-specific effect only to the separate registered comparison against both T0 and Ttask with its recovery caps. The learned teacher and quantized code are not asserted to be exact sufficient statistics. The exact theoretical fixture and empirical teacher/quantization effects answer different questions.

The modeled quantities are CMI under fitted finite conditional laws, with A and AB and both unweighted and PWGTP measures retained. These constraints are not guarantees conditional on the recipient's full H. Full-H attacker CE improvement over the same H is measured predictive recovery; it is neither a CMI estimate nor a CMI lower or upper bound for these restricted model slates. A zero fitted budget allows pre-existing disclosure through H. It is not differential privacy, worst-case protection, joint intersectional protection, or a population guarantee. Marginal SEX/RAC1P constraints remain separate.

One token Z is drawn from Q(.|T). The fixed decoder is sigmoid(logit(b(H_A)) + a_Z). Report expected cross-entropy across tokens, not the cross-entropy of a mixture prediction; publishing a Q row or a risk score would change the release. Fixed decoder, independent deployment learner and validation-selected deployment learner losses remain separate. The same selected predictions are scored under both weightings.

The three anchor pools are disjoint within each anchor but overlap heavily across anchors. The evaluation seal is anchor-specific: these are not globally unseen people or globally untouched labels. The late [DATED_SPLIT_CLARIFICATION.md](DATED_SPLIT_CLARIFICATION.md) records this existing design and does not create a retrospective independent holdout. Shared-household adjustment handles the registered overlap structure for conditional development diagnostics; model-fitting and selection dependencies remain. Fresh-year confirmation remains prospective and unexecuted here.
"""


REFERENCES = """The executable contract and limitations are in [PROTOCOL.md](PROTOCOL.md), [METHOD.md](METHOD.md), [RELEASE_CONTRACT.md](RELEASE_CONTRACT.md), [INTERPRETATION_ADDENDUM_2026-09-22.md](INTERPRETATION_ADDENDUM_2026-09-22.md), and [DATED_SPLIT_CLARIFICATION.md](DATED_SPLIT_CLARIFICATION.md). [OWNERSHIP_CORRECTION.md](OWNERSHIP_CORRECTION.md) locates the predecessor's planned coverage-containing local conditioner; it does not claim that an offending ACS fit was executed.

[NOVELTY_AND_ASSUMPTIONS.md](NOVELTY_AND_ASSUMPTIONS.md) and [LITERATURE_SOURCES.json](LITERATURE_SOURCES.json) give primary-source section/theorem references for perfect privacy, privacy funnels, randomized preprocessing, LEACE and SPLINCE. No novelty is asserted for convexity, nullspaces, quantization, randomization, multi-view constraints or nested refinements. A task-posterior sufficiency separation can be useful without being a new general privacy-funnel theorem.

[HISTORICAL_SCORE_INDEX.json](HISTORICAL_SCORE_INDEX.json) and its [reading guide](HISTORICAL_SCORE_INDEX.md) preserve published H/J, LEACE-on-A0, SPLINCE-on-A0 and executed OptNet scores with exact source commits, paths and hashes. Their supervision, fitting/slate and development data use differ from the current-host matched audits. In particular, historical expanded/kernel-expanded audit scores are separate from the current common audit slate; they are not the same attacker benchmark. Those historical scores are a separate evidence stratum and are not substituted into current matched comparisons or current inference. The optional neural-adversarial baseline was not executed; no comparison with it is established.

The inherited work's service/release ownership framing, explicit recipient views, byte-preserving H baseline and reproducible empirical comparisons remain useful contributions to examine on their own terms. This extension does not silently replace the original manuscript or establish all of its claims. [REVIEW_INDEX.md](REVIEW_INDEX.md) is an internal evidence-navigation aid: actual venue reviews have not been adjudicated by this generator. Any final response to actual reviews must quote and answer those reviews faithfully and separately; current venue prior-review disclosure requirements must be satisfied in any later authorized submission. No manuscript edit, registration, paper submission or contact is performed by this reporting step.

[ORIGINAL_WORK_HANDOFF.md](ORIGINAL_WORK_HANDOFF.md) locates the original dominant-axis/rare-class-sensitive audit, correctly scoped composition correction and historical rebuttal measurements at the full pinned review commit, with exact paths, lines and hashes. Preserve their demonstrated observations and the coverage of any actual verification; a source index does not newly verify historical outcomes. Do not revive the invalid R²-to-classification-accuracy guarantee or describe shared-union erasure as retaining conflicting purposes. The pinned Gaussian comparison distinguishes successful per-purpose retention from destruction by the shared union. Historical rebuttal measurements also do not prove an architecture-wide impossibility theorem.
"""


def render_reports(*, evidence=None, prediction_assessment=None, equivalence=None,
                   numerical_versions=None, verification=None, provenance):
    """Return public documents and a traceable whitelist manifest, with no I/O.

    ``provenance`` maps accepted input basenames to their original-file SHA256.
    The caller verifies those bytes before parsing; object digests below bind
    the exact supplied parsed content but cannot replace that file verification.
    Missing inputs explicitly remain pending, including verification.
    """
    supplied = {"evidence": evidence, "prediction_assessment": prediction_assessment, "equivalence": equivalence,
                "numerical_versions": numerical_versions, "verification": verification}
    if not isinstance(provenance, dict):
        raise ValueError("Input provenance is required")
    pins = {}
    for name, pin in provenance.items():
        if name not in INPUT_NAMES.values():
            raise ValueError("Only declared public aggregate input basenames may be pinned")
        pins[name] = _hash(pin)
    for key, value in supplied.items():
        if value is not None and INPUT_NAMES[key] not in pins:
            raise ValueError("Every supplied aggregate needs its accepted-file SHA256")
    e = _evidence(evidence); p = _predictions(prediction_assessment)
    embedded_versions = None if evidence is None else evidence.get("numerical_versions")
    if numerical_versions is not None and embedded_versions is not None and numerical_versions != embedded_versions:
        raise ValueError("Standalone and evidence numerical-version summaries disagree")
    versions = _versions(numerical_versions if numerical_versions is not None else embedded_versions)
    verified = _verification(verification, e["selection_sha256"])
    eq = {}
    if equivalence is not None:
        if equivalence.get("independent_evidence_claimed") is not False:
            raise ValueError("Kernel groups are not independent evidence")
        eq = {key: _count(equivalence["counts"][key]) for key in EQUIVALENCE_COUNTS if key in equivalence["counts"]}
    passes = [r for r in ROUTES if any(c["group"] == "main" and c["route"] == r
               and c["claim"] == "competitive" and c["passed"] for c in e["claims"])]
    readiness = "evidence_pending" if evidence is None else "inference_pending" if not e["inference_supplied"] else "adjusted_results_supplied"
    if passes:
        headline = ("The registered adjusted competitive criterion passed for " + ", ".join(passes)
                    + ", within this supervised, conditional development analysis. This is a scoped criterion result, not a general method-success or population-privacy claim.")
    elif e["inference_supplied"]:
        headline = ("No adjusted competitive pass is established among the supplied frozen routes. Evaluated failures and blocked comparisons are separated below. "
                    "This is not an impossibility result for other encoders, distributions or privacy–utility trade-offs.")
    else:
        headline = "Final scientific decision pending: adjusted inference has not been supplied. Favorable point estimates cannot establish a method-success claim."
    claims = _table(["Group", "Route", "Claim", "Status", "EVIDENCE.json pointer"],
                    [[c[k] for k in ("group", "route", "claim", "status", "pointer")] for c in e["claims"]])
    selected_tasks = [r for r in e["tasks"] if r["configuration"] in e["nominees"]]
    task_table = _table(["Configuration", "Weight", "Fixed CE", "Independent CE", "Selected CE", "CE − H", "CE − J"],
        [[r[k] for k in ("configuration", "weighting", "fixed", "independent", "selected", "loss_delta_H", "loss_delta_J")] for r in selected_tasks])
    sensitive_table = _table(["Configuration", "Role", "Weight", "Selected CE", "Recovery over H", "Recovery increment over J"],
        [[r[k] for k in ("configuration", "role", "weighting", "selected_ce", "recovery_over_H", "recovery_increment_over_J")]
         for r in e["sensitive"] if r["configuration"] in e["nominees"]])
    prediction_table = _table(["Bet", "Prospective probability", "Assessment", "Complete pertinent coverage", "Source pointer"],
        [[r[k] for k in ("prediction", "probability", "status", "coverage_complete", "pointer")] for r in p])
    # Extrema are deterministic descriptive summaries over complete supplied configurations, not new tests or nominees.
    balanced = {}
    for name in sorted({r["configuration"] for r in e["tasks"]}):
        values = [r["loss_delta_J"] for r in e["tasks"] if r["configuration"] == name]
        if len(values) == 2 and all(v is not None for v in values):
            balanced[name] = math.fsum(values) / 2
    extrema = []
    if balanced:
        for label, fn in (("Lowest balanced selected CE − J", min), ("Highest balanced selected CE − J", max)):
            val = fn(balanced.values())
            extrema.append({"quantity": label, "value": val, "configurations": sorted(k for k, v in balanced.items() if v == val)})
    extrema_table = _table(["Descriptive extremum", "Value", "Configurations"],
                           [[r["quantity"], r["value"], ", ".join(r["configurations"])] for r in extrema])
    source_table = _table(["Accepted input", "SHA256"], sorted(pins.items()))
    if versions is None:
        version_text = "Numerical-version receipt coverage pending. The registered four retry slots must be reported individually; do not assume that a retry was accepted or a replacement re-audited."
    else:
        version_text = _table(["Configuration", "Anchor", "Version decision", "Replacement re-audited"],
            [[r[k] for k in ("configuration", "anchor", "installation_decision", "reaudit_completed")] for r in versions["records"]])
        if not versions["registered"]:
            version_text = "The supplied receipt summary declares no numerical retry registration. No retry completion is inferred.\n"
    version_text += ("\nThe [NUMERICAL_RECOVERY_PLAN.md](NUMERICAL_RECOVERY_PLAN.md) permits one normalized-entropy solver retry per registered slot with the same objective, role laws, support rules, witness bound and tolerances. "
        "Original fallback maps and their audits remain preserved. An accepted retry replaces its map before selection and requires the same audit slate again; a rejected retry retains its original. "
        "Acceptance must not depend on a favorable validation or evaluation score. These are numerical versions of existing configurations, not new nominal methods. Solver-reported optimality, numerical feasibility and an independent exact optimality proof are different claims.\n")
    verification_text = "Independent artifact verification pending; no restore, replay or archive-completeness claim is made by this report."
    if verification is not None:
        verification_text = ("Supplied independent verification status: " + verified["status"] + ".\n\n" +
            _table(["Coverage quantity", "Count"], [[k, verified[k]] for k in ("planned_units", "passed_units", "failed_or_incomplete_units")]) +
            "\nThis status covers the verification plan actually run, not every unselected fitted artifact or the entire encrypted archive. Archive upload, checksum/read-back and clean restore evidence must be reported separately.")
    research = ("# Research decision\n\n" + headline + "\n\n## Frozen adjusted claims\n\n" + claims +
        "\nMain competitive, historical-J, attribution and stricter diagnostic formulas are separate. The adjusted family is not enlarged by descriptive full-grid tables. "
        "The generator checks formula-tree consistency and endpoint presence; accepted upstream claim evaluation must recompute the registered bound checks.\n\n" +
        "## Selected residence and disclosure results\n\n" + task_table +
        "\nCE is in nats; lower is better for prediction. CE − H and CE − J are signed loss differences: negative is improved utility.\n\n" + sensitive_table +
        "\nRecovery over H is CE(H) − CE(candidate), so positive means additional measured disclosure. Recovery increment over J is CE(J) − CE(candidate), so positive means more recovery than J. "
        "These are full-H predictive comparisons, not modeled CMI. Means require all three anchors; missing configurations are not silently averaged.\n\n" +
        "## Descriptive range and subjective bets\n\n" + extrema_table +
        "\nThese extrema are drawn only from complete supplied configurations, use equal U/PWGTP weighting, and carry no new significance or selection claim. Full-grid point estimates remain descriptive.\n\n" +
        prediction_table + "\nP1–P7 preserve their prospective probabilities and machine assessments. A supported existential bet may have incomplete coverage; refutation requires complete pertinent attempts. "
        "Unassessed is not refuted. These bets are not scientific gates.\n\n## Interpretation and handoff\n\n" + SCOPE + "\n" + REFERENCES)
    paper = ("# Paper addendum\n\n" + headline + "\n\n" + SCOPE +
        "\nThe empirical question is whether distinctions lost by a task-only code improve the constrained trade-off for this supervised service. "
        "Evidence for a strict refined-code advantage, matched-baseline competitiveness and coalition protection must each use its own frozen formula; one cannot stand in for another. "
        "Continuous teacher → code/action → protected-channel loss differences are a controlled descriptive decomposition, not a causal decomposition. "
        "Finite empirical zero-privacy certificates establish supported-state feasibility of a nonconstant channel when an exact witness is supplied; they do not establish useful held-out utility.\n\n" +
        "## Machine-grounded findings\n\n" + claims + "\n" + task_table +
        "\nThe strongest defensible positive or negative statement is the scoped adjusted result above, with the supplied point values and prediction assessments. "
        "No favorable descriptive cell can promote a blocked or failed registered claim.\n\n## Numerical versions\n\n" + version_text +
        "\n## Original work, related work and review handoff\n\n" + REFERENCES)
    validation = ("# Validation and provenance\n\n" + verification_text +
        "\n\n## Accepted aggregate inputs\n\n" + source_table +
        "\nThis generator performs no fitting, inference, private model replay, filesystem discovery or sealed evaluation access. "
        "The caller supplies hash-verified aggregate files; manifest content digests bind the supplied parsed objects. An input SHA alone is not a claim that this generator re-read the file.\n\n" +
        "## Execution and exact-equivalence counts\n\n" + _table(["Execution count", "Value"], sorted(e["execution"].items())) + "\n" +
        _table(["Exact-equivalence count", "Value"], sorted(eq.items())) +
        "\nNominal release/anchor units, active accepted artifacts, learner/role-fit units, numerical versions and exact kernel groups are distinct counts. "
        "Equal hashes or repeated audits are not independent observations. Exact equivalence requires matching action dictionaries and code-construction identities, with only certified child-row copying or global constants reduced across codes. "
        "Near equality and equality on observed predictions alone do not establish exact channel identity.\n\n## Numerical retry provenance\n\n" + version_text +
        "\n## Verification scope still required\n\nIndependent checks must name actual coverage of objective/CMI replay, supported-state ties, refinement embeddings, zero-rank/witness certificates, H parity, "
        "runtime deployment inputs, selected prediction identity, all 16 role losses, source/selection pins, expected token losses and wire encoding. "
        "A finite empirical certificate is not population privacy or proof of optimizer optimality. Missing comparisons, numerical failures and resource closeout remain visible. "
        "B-only reuse is counted explicitly; weighted and unweighted scores must use the same selected predictions.\n\n" +
        "Public artifacts contain aggregate evidence and hashes, not person IDs, protected-label arrays, row-level predictions or model weights. "
        "Private fitted artifacts require the separate encrypted archive, read-back and restore checks described in [REPRODUCE.md](REPRODUCE.md). "
        "No all-files privacy or successful archival claim follows merely from this whitelist renderer.\n\n" + SCOPE + "\n" + REFERENCES)
    documents = dict(zip(REPORT_NAMES, (research, paper, validation)))
    manifest = {"schema": 1, "readiness": readiness, "competitive_pass_established": bool(passes),
        "competitive_pass_routes": passes, "selection_sha256": e["selection_sha256"],
        "complete_configurations": e["complete_configurations"], "incomplete_configurations": e["incomplete_configurations"],
        "family_size": e["family_size"], "claims": e["claims"], "adjusted_bounds": e["bounds"],
        "task_points": e["tasks"], "primary_sensitive_points": e["sensitive"], "descriptive_extrema": extrema,
        "prediction_assessments": p, "execution_counts": e["execution"], "equivalence_counts": eq,
        "numerical_versions": versions, "verification": verified, "provenance": pins,
        "input_content_digests": {INPUT_NAMES[k]: _digest(v) for k, v in supplied.items() if v is not None},
        "document_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in documents.items()},
        "scope": "accepted aggregate reporting only; conditional development diagnostics; no new inference or independent-holdout claim"}
    return {"documents": documents, "manifest": manifest}


def write_reports(report, *, out_dir):
    """Exclusively create the three documents and traceability manifest.

    Existing files are never replaced. Call in a fresh final-report directory;
    a failed partial write remains explicit and is not silently repaired here.
    """
    if set(report.get("documents", {})) != set(REPORT_NAMES):
        raise ValueError("Unexpected report filenames")
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    outputs = {**report["documents"], "NARRATIVE_MANIFEST.json": json.dumps(report["manifest"], indent=2, sort_keys=True, allow_nan=False) + "\n"}
    for name in outputs:
        if (root / name).exists():
            raise FileExistsError(root / name)
    for name, text in outputs.items():
        with (root / name).open("x", encoding="utf-8") as handle:
            handle.write(text)
    return {name: str(root / name) for name in outputs}
