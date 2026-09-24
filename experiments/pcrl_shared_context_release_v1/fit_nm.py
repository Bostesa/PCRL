"""Nested-mixture (NM) unit runner mirroring the AR Branch A alternation.

Subcommands
-----------
build-bank : per anchor, fit the round-0 decoder and attack bank exactly as
             AR round 0, solve the round-0 T32 LP, extract its duals, fit the
             frozen nuisance, 2-fold cross-fitted task decoders, the M=5
             D17-anchored policy bank and the hard context rules.  Written
             once and shared by every variant of that anchor.
fit        : one variant (NM1_U, NM1_P, NM4_U, NM4_P, T32_U, T32_P) from a
             frozen bank: rounds 0..R of LP -> decoder refit on the per-person
             coverage law -> best-response attacks -> rebase -> re-solve.
             DET_SEL1 / DET_SEL4: exhaustive one-policy-per-context selector
             evaluated on the final round of an NM1_U / NM4_U unit.
decide-k   : registered K=4 -> K=2 fallback from three anchors' bank censuses.

All fitted objects and person-level arrays stay in a private unit directory
(a path containing `private`).  The outer role is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.pcrl_adaptive_release_v1 import alternate, fit_a, fit_b, roles
from experiments.pcrl_adaptive_release_v1 import nuisance as ar_nuisance
from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit
from experiments.pcrl_task_aligned_cuts_v1 import data, method
from experiments.pcrl_task_directed_release_v1.audits import fit_slate, load_candidate
from . import channel, contexts, laws, policies

SCHEMA_BANK = "pcrl-sc-policy-bank-v1"
SCHEMA_UNIT = "pcrl-sc-nested-unit-v1"
SCHEMA_RELEASE = "pcrl-sc-nested-release-v1"
DELTA = 0.001
MAX_ROUNDS = 6
SEED_BASE = 20260924              # AR round seeds: SEED_BASE + 10000*anchor + 100*round
SCIENCE_ROLES = fit_a.SCIENCE_ROLES
PROTECTED_ROLES = fit_a.PROTECTED_ROLES
VARIANTS = {"NM1_U": ("NM1", "U"), "NM1_P": ("NM1", "P"),
            "NM4_U": ("NM4", "U"), "NM4_P": ("NM4", "P"),
            "T32_U": ("T32", "U"), "T32_P": ("T32", "P")}
DET_VARIANTS = {"DET_SEL1": "NM1", "DET_SEL4": "NM4"}
SMOKE_FRACTION = 0.3
FOLD_SALT = "pcrl_shared_context_release_v1|crossfit_fold|"
SMOKE_SALT = "pcrl_shared_context_release_v1|smoke_subsample|"
LEGAL_ROLES_FOR_LAW = ("nuisance_train", "audit_fit", "coefficient_split",
                       "inner_selection", "inner_check")

_write_json = fit_a._write_or_verify_json
_save_npz = fit_a._save_or_verify_npz
_sha_array = fit_a._sha_array
_sha_file = data.sha256_file


def _inventory(root):
    """SHA-256 of every artifact except the completion receipts themselves."""
    root = Path(root)
    return {str(path.relative_to(root)): _sha_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name not in ("COMPLETE.json", "BANK_COMPLETE.json", laws.DESCRIPTOR)}


def _write_release_descriptor(root, variant, anchor):
    """Infrastructure descriptor (`laws.py`, kind "nested") pinning spec, params, receipt."""
    root = Path(root)
    spec = json.loads((root / "RELEASE_SPEC.json").read_text())
    pins = {name: _sha_file(root / name)
            for name in ("RELEASE_SPEC.json", spec["params_relative"], "COMPLETE.json")}
    return laws.write_descriptor(root, "nested", release_id=f"a{int(anchor)}_{variant}",
                                 anchor=int(anchor), pins=pins,
                                 extra={"variant": variant, "K": spec["K"],
                                        "eta_fixed_zero": spec["eta_fixed_zero"],
                                        "bank_manifest_sha256": spec["bank_manifest_sha256"]})


def _log(event, **fields):
    print(json.dumps({"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      **fields}, sort_keys=True, default=str), flush=True)


def _private_root(path) -> Path:
    root = Path(path)
    if "private" not in root.parts:
        raise ValueError("fitted objects require a private output directory")
    return root


def _unit_hash(households, salt) -> np.ndarray:
    return np.asarray([int.from_bytes(hashlib.sha256((salt + str(h)).encode()).digest()[:8], "big") / 2**64
                       for h in households])


def subset_rows(rows, mask):
    """Row subset of a pooled-role dictionary (labels included, private)."""
    mask = np.asarray(mask, dtype=bool)
    out = {key: np.asarray(value)[mask] for key, value in rows.items() if key != "labels"}
    out["labels"] = {key: np.asarray(value)[mask] for key, value in rows["labels"].items()}
    return out


def smoke_subsample(role_dict, fraction=SMOKE_FRACTION):
    return {name: subset_rows(rows, _unit_hash(rows["households"], SMOKE_SALT) < fraction)
            for name, rows in role_dict.items()}


def load_roles(anchor, index_path, *, smoke=False):
    """Hash-verified sanitized 2018 inner roles (outer never loaded)."""
    sanitized = Path(__file__).resolve().parents[2] / data.SANITIZED_RELATIVE / "SANITIZATION.json"
    if not sanitized.is_file():
        raise RuntimeError("sanitized 2018 receipt required; refusing raw prepared fallback")
    value = data.index(index_path)
    prepared = data.load_prepared(value, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise RuntimeError("outer assessment labels appeared before lock")
    role_dict = {name: roles.pooled_role(prepared, name) for name in SCIENCE_ROLES}
    del prepared
    q_ref = data.load_map(value, anchor, "D17")
    historical_q = data.load_map(value, anchor, "Q")
    encoder_sha = data.member_record(value, anchor, "encoder")["sha256"]
    if smoke:
        role_dict = smoke_subsample(role_dict)
    return role_dict, q_ref, historical_q, encoder_sha


def _code_sha256():
    module = Path(__file__).resolve().parent
    return {name: _sha_file(module / name)
            for name in ("channel.py", "policies.py", "contexts.py", "fit_nm.py")}


# ---------------------------------------------------------------------------
# Laws, decoders and attack banks on per-person laws
# ---------------------------------------------------------------------------

def coverage_law_person(law, codes, q_ref):
    """(current per-person law + D17[t] + uniform)/3, AR coverage rule per person."""
    law = channel.validate_law(law)
    q_ref = method.validate_channel(q_ref, n_tokens=17)
    t = np.asarray(codes, dtype=np.int64)
    return channel.validate_law((law + q_ref[t] + 1 / 17) / 3)


def task_scores_law(decoder, rows, law):
    """Exact expected-token task CE of a frozen decoder under a per-person law."""
    valid, losses = fit_a.task_person_losses(decoder, rows)
    person = np.einsum("nz,nz->n", channel.validate_law(law)[valid], losses)
    w = np.asarray(rows["weights"], dtype=np.float64)[valid]
    return {"U": float(person.mean()), "W": float(np.dot(w / w.sum(), person)),
            "n_original_people": int(len(person))}


def fit_decoder_per_person(role_dict, laws, q_ref, output_dir, seed):
    """AR `fit_frozen_decoder` with the per-person coverage law (write-once)."""
    root = Path(output_dir)
    train, selected = role_dict["nuisance_train"], role_dict["inner_selection"]
    mask_train = fit_a._valid_mask(train, "same_residence")
    mask_select = fit_a._valid_mask(selected, "same_residence")
    inherited_audit.assert_household_disjoint(
        np.asarray(train["households"])[mask_train],
        np.asarray(selected["households"])[mask_select])
    fit_law = coverage_law_person(laws["nuisance_train"], train["token_codes"], q_ref)
    select_law = coverage_law_person(laws["inner_selection"], selected["token_codes"], q_ref)
    expected = {"schema": "pcrl-sc-decoder-v1", "seed": int(seed),
                "fit_law_sha256": _sha_array(fit_law),
                "selection_law_sha256": _sha_array(select_law),
                "train_input_sha256": fit_a._decoder_input_hash(train, mask_train),
                "selection_input_sha256": fit_a._decoder_input_hash(selected, mask_select),
                "training_law": "(current per-person law + D17[T0] + uniform_17)/3",
                "fit_original_people": int(mask_train.sum()),
                "selection_original_people": int(mask_select.sum())}
    receipt_path = root / "FIT_DECODER.json"
    slate = root / "slate"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("existing decoder receipt differs from input contract")
    else:
        if not (slate / "slate.json").exists():
            if slate.exists() and any(slate.iterdir()):
                raise FileExistsError("partial task slate retained; quarantine before technical retry")
            fit_slate(np.asarray(train["ha"])[mask_train], fit_law[mask_train],
                      np.asarray(train["labels"]["same_residence"])[mask_train],
                      np.asarray(train["weights"])[mask_train],
                      np.asarray(selected["ha"])[mask_select], select_law[mask_select],
                      np.asarray(selected["labels"]["same_residence"])[mask_select],
                      np.asarray(selected["weights"])[mask_select],
                      2, int(seed), slate, slate="standard")
        meta = json.loads((slate / "slate.json").read_text())
        model_dir = slate / meta["selection"]
        receipt = {**expected, "selected_candidate": meta["selection"],
                   "selected_model_relative_directory": str(model_dir.relative_to(root)),
                   "selected_model_sha256": inherited_audit.model_directory_hash(model_dir),
                   "slate_sha256": _sha_file(slate / "slate.json"),
                   "candidate_count": len(meta["candidate_ids"])}
        _write_json(receipt_path, receipt)
    return _load_decoder_receipt(root)


def _load_decoder_receipt(root):
    root = Path(root)
    receipt = json.loads((root / "FIT_DECODER.json").read_text())
    if _sha_file(root / "slate" / "slate.json") != receipt["slate_sha256"]:
        raise ValueError("frozen task slate hash differs")
    model_dir = root / receipt["selected_model_relative_directory"]
    if inherited_audit.model_directory_hash(model_dir) != receipt["selected_model_sha256"]:
        raise ValueError("frozen task decoder model hash differs")
    return {**receipt, "model_directory": str(model_dir.resolve()),
            "decoder": load_candidate(model_dir),
            "receipt_sha256": _sha_file(root / "FIT_DECODER.json")}


def build_attack_bank_per_person(role_dict, source, laws, output_dir, seed,
                                 *, target_roles=PROTECTED_ROLES):
    """Best-response attack slates on a per-person source law (AR recipe).

    Mirrors `AR/fit_a.build_attack_bank` for a single released source: views A
    and AB per target, standard slate, fit audit_fit, validate inner_selection;
    every candidate becomes a cut (A roles from view A; AB roles from A, AB).
    """
    root = Path(output_dir)
    fit_rows, selection_rows = role_dict["audit_fit"], role_dict["inner_selection"]
    inherited_audit.assert_household_disjoint(
        fit_rows["households"], selection_rows["households"],
        role_dict["coefficient_split"]["households"])
    fit_law = channel.validate_law(laws["audit_fit"])
    select_law = channel.validate_law(laws["inner_selection"])
    source_sha = hashlib.sha256((_sha_array(fit_law) + _sha_array(select_law)).encode()).hexdigest()
    fitted = {}
    counter = 0
    for target in sorted({role.split("/")[1] for role in target_roles}):
        need_a = f"A/{target}" in target_roles or f"AB/{target}" in target_roles
        need_ab = f"AB/{target}" in target_roles
        for view in ("A", "AB"):
            if (view == "A" and not need_a) or (view == "AB" and not need_ab):
                continue
            fitted[(view, target)] = inherited_audit.fit_role_slate(
                fit_rows, fit_law, selection_rows, select_law, f"attack:{view}/{target}",
                root / "slates" / f"{view}_{target}" / source, int(seed) + 1000 * counter,
                release_id=source, slate="standard")
            counter += 1
    specs = []
    for role in target_roles:
        role_view, target = role.split("/")
        for source_view in (("A",) if role_view == "A" else ("A", "AB")):
            registry = fitted[(source_view, target)]
            for cid, model in sorted(registry["models"].items()):
                model_dir = Path(model["model_directory"]).resolve()
                specs.append({"id": f"{role}/from_{source_view}/{source}/{cid}",
                              "role": role, "target": target, "source_view": source_view,
                              "wire": "release", "training_source": source,
                              "source_channel_sha256": source_sha,
                              "model_relative_directory": str(model_dir.relative_to(root.resolve())),
                              "model_directory": str(model_dir),
                              "model_sha256": model["model_sha256"],
                              "class_order": list(range(fit_a.CLASS_COUNT[target])),
                              "fit_class_counts": registry["fit_class_counts"],
                              "fit_missing_classes": registry["fit_missing_classes"]})
    _write_json(root / "NESTED_ATTACK_BANK.json", {
        "schema": "pcrl-sc-attack-bank-v1", "seed": int(seed), "source": source,
        "source_law_sha256": source_sha, "target_roles": list(target_roles),
        "attack_specs": [{k: v for k, v in s.items() if k != "model_directory"} for s in specs],
        "fit_slate_count": len(fitted)})
    return specs


def nested_cuts(specs, coef_rows, contexts_coef, tokens_coef, n_contexts):
    """Both coefficient blocks for every attack spec, U and W, on coefficient_split."""
    cuts = []
    codes = np.asarray(coef_rows["token_codes"], dtype=np.int64)
    w = np.asarray(coef_rows["weights"], dtype=np.float64)
    for spec in specs:
        valid, losses = fit_a.person_loss_from_attack(spec, coef_rows)
        pair = channel.coefficient_pair_blocks(losses, codes[valid], contexts_coef[valid],
                                               tokens_coef[valid], w[valid], n_contexts=n_contexts)
        pool = fit_a._population_hash(coef_rows, spec["target"], valid)
        for v in channel.WEIGHTINGS:
            cuts.append({"id": f"{spec['id']}/{v}", "role": spec["role"], "weighting": v,
                         "coeff_B": pair[v]["B"], "coeff_A": pair[v]["A"],
                         "coefficient_pool_sha256": pool, "class_order": spec["class_order"],
                         "weight_normalization": "1/n" if v == "U" else "PWGTP/sum(PWGTP)",
                         "predictor_sha256": spec["model_sha256"],
                         "source_view": spec["source_view"],
                         "source_release": spec["training_source"],
                         "model_candidate": spec["id"].rsplit("/", 1)[1]})
    return cuts


def task_blocks(decoder, coef_rows, contexts_coef, tokens_coef, n_contexts):
    valid, losses = fit_a.task_person_losses(decoder, coef_rows)
    return channel.coefficient_pair_blocks(
        losses, np.asarray(coef_rows["token_codes"], dtype=np.int64)[valid],
        contexts_coef[valid], tokens_coef[valid],
        np.asarray(coef_rows["weights"], dtype=np.float64)[valid], n_contexts=n_contexts)


def _save_cut_blocks(path, cuts):
    _save_npz(path, **{f"cut_{i:04d}_{b}": cut[f"coeff_{b}"]
                       for i, cut in enumerate(cuts) for b in ("B", "A")})


def _load_cut_blocks(npz_path, meta):
    with np.load(npz_path, allow_pickle=False) as arrays:
        return [{**m, "coeff_B": arrays[f"cut_{i:04d}_B"].copy(), "coeff_A": arrays[f"cut_{i:04d}_A"].copy()}
                for i, m in enumerate(meta)]


def _meta(cuts):
    return [{k: v for k, v in cut.items() if k not in ("coeff_B", "coeff_A")} for cut in cuts]


# ---------------------------------------------------------------------------
# build-bank
# ---------------------------------------------------------------------------

def _crossfit_task_losses(role_dict, historical_q, q_ref, root, seed):
    """2-fold household-grouped cross-fitted task CE on nuisance_train (M1.2)."""
    train = role_dict["nuisance_train"]
    valid = fit_a._valid_mask(train, "same_residence")
    fold = (_unit_hash(train["households"], FOLD_SALT) < 0.5).astype(int)
    full = np.full((len(valid), 17), np.nan)
    receipts = []
    for f in (0, 1):
        sub = {"nuisance_train": subset_rows(train, fold != f),
               "inner_selection": role_dict["inner_selection"]}
        record = fit_a.fit_frozen_decoder(sub, historical_q, q_ref,
                                          root / f"crossfit_decoder_f{f}", seed + f)
        held = np.flatnonzero(fold == f)
        vmask, losses = fit_a.task_person_losses(record["decoder"], subset_rows(train, fold == f))
        full[held[vmask]] = losses
        receipts.append({k: v for k, v in record.items() if k not in ("decoder", "model_directory")})
    u = full[valid]
    if not np.isfinite(u).all():
        raise AssertionError("cross-fitted task losses incomplete")
    return valid, u, {"folds": receipts, "fold_rule": "SHA256(salt|household) < 0.5",
                      "fold_people": [int((fold == f).sum()) for f in (0, 1)]}


def build_bank(anchor, role_dict, q_ref, historical_q, encoder_sha256, output_dir, *, smoke=False,
               target_roles=PROTECTED_ROLES):
    root = _private_root(output_dir)
    if set(role_dict) != set(SCIENCE_ROLES):
        raise ValueError("bank receives exactly the five inner roles")
    inherited_audit.assert_household_disjoint(*(np.asarray(role_dict[n]["households"]) for n in SCIENCE_ROLES))
    q_ref = method.validate_channel(q_ref, n_states=32, n_tokens=17)
    historical_q = method.validate_channel(historical_q, n_states=32, n_tokens=17)
    expected = {"schema": SCHEMA_BANK, "anchor": int(anchor), "delta": DELTA,
                "seed_base": SEED_BASE, "smoke": bool(smoke), "tau": policies.TAU,
                "target_roles": list(target_roles),
                "encoder_sha256": encoder_sha256, "q_ref_sha256": _sha_array(q_ref),
                "historical_q_sha256": _sha_array(historical_q),
                "role_input_sha256": {n: fit_a._role_fingerprint(role_dict[n]) for n in SCIENCE_ROLES},
                "outer_labels_accessed": False}
    done = root / "BANK_COMPLETE.json"
    if done.exists():
        return load_bank_dir(root, expected=expected)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "INPUTS.json", {**expected, "code_sha256": _code_sha256()})
    seed0 = SEED_BASE + 10000 * int(anchor)
    t0 = time.time()
    decoder = fit_a.fit_frozen_decoder(role_dict, historical_q, q_ref, root / "decoder_r00", seed0)
    _log("bank_decoder_r00", seconds=round(time.time() - t0, 1))
    sources = {"H": None, "D17": q_ref, "Q": historical_q,
               "coverage": (q_ref + historical_q + 1 / 17) / 3}
    bank = fit_a.build_attack_bank(role_dict, sources, root / "bank_r00", seed0 + 50000,
                                   target_roles=tuple(target_roles))
    _log("bank_attacks_r00", seconds=round(time.time() - t0, 1), cuts=len(bank["cuts"]))
    # Round-0 T32 LP, identical to AR round 0 (one retained decoder).
    task_pair = fit_a.task_cost_pair(decoder["decoder"], role_dict["coefficient_split"])
    update = alternate.channel_update(task_pair, q_ref, bank["cuts"], DELTA)
    solution = update["solution"]
    if not solution.get("feasible"):
        raise RuntimeError("round-0 T32 LP unresolved")
    cost = 0.5 * (task_pair["U"] + task_pair["W"])
    duals = fit_b.fixed_bank_dual_prices(cost, update["cuts"])
    if duals["status"] != "optimal":
        raise RuntimeError("round-0 T32 dual extraction failed")
    t32 = root / "round0_t32"
    _save_npz(t32 / "Q.npz", Q=solution["Q"])
    _save_npz(t32 / "COSTS.npz", cost_U=task_pair["U"], cost_W=task_pair["W"], cost=cost)
    _write_json(t32 / "DUALS.json", {"multipliers": duals["multipliers"],
                                     "objective": duals["objective"],
                                     "dual_lower_bound": duals["dual_lower_bound"],
                                     "bank_sha256": update["bank_sha256"],
                                     "rho": update["rho"],
                                     "lp_objective": solution["objective"],
                                     "scope": duals["scope"]})
    # Frozen nuisance (contexts) on nuisance_train.
    train = role_dict["nuisance_train"]
    nuis, nuis_receipt = ar_nuisance.fit_frozen_nuisance(train, seed=seed0 + 70000)
    nuis_receipt = ar_nuisance.save_frozen_nuisance(root / "nuisance", nuis, nuis_receipt)
    # Cross-fitted task costs + priced attack costs on nuisance_train.
    valid, u, crossfit = _crossfit_task_losses(role_dict, historical_q, q_ref, root, seed0 + 80000)
    _log("bank_crossfit", seconds=round(time.time() - t0, 1))
    groups = policies.price_groups(update["cuts"], duals["multipliers"])
    needed = sorted({cid for g in groups.values() for cid in g["multipliers"]})
    specs_by_id = {spec["id"]: spec for spec in bank["attack_specs"]}
    priced_rows, _ = fit_b._attack_loss_rows([c for c in update["cuts"] if c["id"] in needed],
                                             specs_by_id, train)
    legal = policies.legal_inputs(train)
    features = policies.policy_features(legal)
    standardizer = policies.FeatureStandardizer.fit(features)
    z = standardizer.transform(features)[valid]
    d17_tok = channel.d17_token_map(q_ref)
    members = [policies.D17Policy("D17", d17_tok)]
    fit_records = {}
    for index, name in enumerate(policies.POLICY_NAMES[1:], start=1):
        group = groups[name]
        g = policies.priced_person_costs(u, priced_rows, group["multipliers"], valid)
        oracle, record = policies.fit_paired_oracle(
            z, g, d17_tok[np.asarray(train["token_codes"])[valid]],
            np.asarray(train["weights"])[valid], np.asarray(train["households"])[valid],
            seed=seed0 + 90000 + index)
        pricing = {k: v for k, v in group.items() if k != "multipliers"}
        pricing["priced_cuts"] = len(group["multipliers"])
        members.append(policies.switched_policy_from_oracle(name, d17_tok, standardizer, oracle,
                                                            tau=policies.TAU, pricing=pricing))
        fit_records[name] = {**record, "pricing": pricing,
                             "target_mean": g.mean(0).tolist()}
        _log("bank_policy", name=name, seconds=round(time.time() - t0, 1))
    del priced_rows
    bank_obj = policies.PolicyBank(policies.POLICY_NAMES, tuple(members))
    saved = policies.save_bank(root / "policies", bank_obj)
    coef = role_dict["coefficient_split"]
    tokens = {name: bank_obj.predict(policies.legal_inputs(role_dict[name]))
              for name in ("nuisance_train", "coefficient_split")}
    ledger = policies.alias_ledger(tokens["coefficient_split"], list(policies.POLICY_NAMES))
    diagnostics = {name: policies.disagreement_diagnostics(
        tokens[name], list(policies.POLICY_NAMES), role_dict[name]["token_codes"],
        role_dict[name]["weights"], role_dict[name]["households"]) for name in tokens}
    _write_json(root / "policies" / "POLICIES.json", {
        "schema": SCHEMA_BANK, **saved, "tau": policies.TAU,
        "features": "[X_A(32), H_A(4), logit(p), r, risk(11)] standardized on nuisance_train",
        "cost_target": "paired Delta_i(z) = g_i(z) - g_i(D17(x_i)) (M2), g = u - sum_j lambda_j a_ij; u cross-fitted (2-fold households)",
        "alias_ledger": ledger, "fits": fit_records, "crossfit": crossfit,
        "price_groups": {k: {kk: vv for kk, vv in v.items() if kk != "multipliers"}
                         for k, v in groups.items()},
        "disagreement": diagnostics})
    rules = contexts.fit_context_rules(train, nuis, nuis_receipt["model_sha256"])
    rules_saved = contexts.save_rules(root / "contexts", rules)
    legal_coef = policies.legal_inputs(coef)
    census = {str(k): contexts.support_census(rule.assign(legal_coef), coef["households"],
                                              coef["weights"], k) for k, rule in rules.items()}
    _write_json(root / "contexts" / "CONTEXTS.json", {**rules_saved, "census_coefficient_split": census,
                                                      "nuisance_sha256": nuis_receipt["model_sha256"]})
    manifest = {**expected, "status": "COMPLETE", "retained_policies": ledger["retained"],
                "policy_bank_sha256": saved["sha256"], "contexts_sha256": rules_saved["sha256"],
                "artifact_sha256": _inventory(root)}
    _write_json(done, manifest)
    _log("bank_complete", seconds=round(time.time() - t0, 1))
    return load_bank_dir(root, expected=expected)


def load_bank_dir(bank_dir, *, expected=None):
    root = Path(bank_dir).resolve()
    manifest = json.loads((root / "BANK_COMPLETE.json").read_text())
    if expected is not None and any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("completed bank differs from requested inputs")
    if manifest.get("artifact_sha256") != _inventory(root):
        raise ValueError("bank artifact inventory differs")
    decoder = _load_ar_decoder(root / "decoder_r00")
    attack = fit_a.load_frozen_bank(root / "bank_r00")
    policy_bank = policies.load_bank(root / "policies", manifest["policy_bank_sha256"])
    rules = contexts.load_rules(root / "contexts", manifest["contexts_sha256"])
    ledger = json.loads((root / "policies" / "POLICIES.json").read_text())["alias_ledger"]
    return {"root": root, "manifest": manifest, "decoder": decoder, "attack": attack,
            "policy_bank": policy_bank.subset(ledger["retained"]),
            "full_policy_bank": policy_bank, "rules": rules,
            "retained_policies": ledger["retained"],
            "manifest_sha256": _sha_file(root / "BANK_COMPLETE.json")}


def _load_ar_decoder(directory):
    receipt = json.loads((Path(directory) / "FIT_DECODER.json").read_text())
    spec = {"model_directory": str((Path(directory) / receipt["selected_model_relative_directory"]).resolve()),
            "selected_model_sha256": receipt["selected_model_sha256"]}
    return {**receipt, **spec, "decoder": fit_a.load_frozen_decoder(spec)}


# ---------------------------------------------------------------------------
# fit (one variant)
# ---------------------------------------------------------------------------

def _variant_setup(variant, nm4_k):
    family, form = VARIANTS[variant]
    if family == "NM4":
        if nm4_k not in (2, 4):
            raise ValueError("NM4 variants require the registered K decision (--nm4-k 4 or 2)")
        return family, form, int(nm4_k), False
    return family, form, 1, family == "T32"


def _solve(form, cost, calibrated, q_ref, fix):
    if form == "U":
        return channel.solve_utility(cost, calibrated["cuts"], q_ref, fix_eta_zero=fix)
    return channel.solve_privacy_first(cost, calibrated["cuts"], q_ref, fix_eta_zero=fix)


def _jsonable_solution(solution):
    return {k: v for k, v in solution.items() if k not in ("params", "cut_losses", "cut_multipliers")}


def run_unit(variant, anchor, role_dict, q_ref, historical_q, bank_dir, output_dir,
             *, rounds=MAX_ROUNDS, nm4_k=None, smoke=False):
    """Fit one NM/T32 variant from a frozen bank; write-once checkpoints."""
    family, form, K, fix = _variant_setup(variant, nm4_k)
    if not isinstance(rounds, int) or not 0 <= rounds <= MAX_ROUNDS:
        raise ValueError("round count outside registration")
    if set(role_dict) != set(SCIENCE_ROLES):
        raise ValueError("unit receives exactly the five inner roles")
    root = _private_root(output_dir)
    q_ref = method.validate_channel(q_ref, n_states=32, n_tokens=17)
    historical_q = method.validate_channel(historical_q, n_states=32, n_tokens=17)
    bank = load_bank_dir(bank_dir)
    if bank["manifest"]["smoke"] != bool(smoke) or bank["manifest"]["q_ref_sha256"] != _sha_array(q_ref):
        raise ValueError("bank smoke flag / D17 differs from this unit")
    fingerprints = {n: fit_a._role_fingerprint(role_dict[n]) for n in SCIENCE_ROLES}
    if fingerprints != bank["manifest"]["role_input_sha256"]:
        raise ValueError("unit role inputs differ from the frozen bank")
    rule = bank["rules"][K]
    policy_bank = bank["policy_bank"]
    M = len(policy_bank.names)
    expected = {"schema": SCHEMA_UNIT, "variant": variant, "family": family, "form": form,
                "anchor": int(anchor), "delta": DELTA, "rounds": rounds, "K": K,
                "eta_fixed_zero": fix, "policy_columns": list(policy_bank.names),
                "bank_manifest_sha256": bank["manifest_sha256"], "smoke": bool(smoke),
                "q_ref_sha256": _sha_array(q_ref), "historical_q_sha256": _sha_array(historical_q),
                "role_input_sha256": fingerprints, "outer_labels_accessed": False}
    complete = root / "COMPLETE.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("completed unit differs from requested scientific unit")
        if receipt.get("artifact_sha256") != _inventory(root):
            raise ValueError("completed unit artifact inventory differs")
        _write_release_descriptor(root, variant, anchor)
        return receipt
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "INPUTS.json", {**expected, "bank_dir": str(bank["root"]),
                                       "code_sha256": _code_sha256()})
    t_start = time.time()
    legal = {n: policies.legal_inputs(role_dict[n]) for n in SCIENCE_ROLES}
    tokens = {n: policy_bank.predict(legal[n]) for n in SCIENCE_ROLES}
    ctx = {n: rule.assign(legal[n]) for n in SCIENCE_ROLES}
    codes = {n: np.asarray(role_dict[n]["token_codes"], dtype=np.int64) for n in SCIENCE_ROLES}
    coef = role_dict["coefficient_split"]
    # Round-0 bank blocks; the B block must equal the AR 32x17 coefficient.
    cut_store = nested_cuts(bank["attack"]["attack_specs"], coef, ctx["coefficient_split"],
                            tokens["coefficient_split"], K)
    ar_coeff = {cut["id"]: cut["coeff"] for cut in bank["attack"]["cuts"]}
    parity = max(float(np.max(np.abs(cut["coeff_B"] - ar_coeff[cut["id"]]))) for cut in cut_store)
    if set(ar_coeff) != {cut["id"] for cut in cut_store} or parity > 1e-12:
        raise AssertionError("nested B block differs from inherited AR T32 coefficients")
    _save_cut_blocks(root / "blocks_r00.npz", cut_store)
    _write_json(root / "blocks_r00.json", {"cuts": _meta(cut_store), "ar_b_block_max_abs_difference": parity})
    decoders = {f"round_00/{bank['decoder']['selected_candidate']}": bank["decoder"]}
    laws = {n: historical_q[codes[n]] for n in SCIENCE_ROLES}   # AR: current_Q starts at historical Q
    records, params_by_round = [], []
    final = None
    for r in range(rounds + 1):
        round_root = root / f"round_r{r:02d}"
        seed = SEED_BASE + 10000 * int(anchor) + 100 * r
        if r > 0:
            rec = fit_decoder_per_person(role_dict, laws, q_ref, root / f"decoder_r{r:02d}", seed)
            decoders[f"round_{r:02d}/{rec['selected_candidate']}"] = rec
            specs = build_attack_bank_per_person(role_dict, f"candidate_r{r-1:02d}", laws,
                                                 root / f"bank_r{r:02d}", seed + 50000,
                                                 target_roles=tuple(bank["manifest"]["target_roles"]))
            fresh = nested_cuts(specs, coef, ctx["coefficient_split"], tokens["coefficient_split"], K)
            _save_cut_blocks(root / f"blocks_r{r:02d}.npz", fresh)
            _write_json(root / f"blocks_r{r:02d}.json", {"cuts": _meta(fresh)})
            cut_store = cut_store + fresh
        scores = {cid: task_scores_law(rec["decoder"], role_dict["inner_selection"], laws["inner_selection"])
                  for cid, rec in decoders.items()}
        chosen = min(scores, key=lambda cid: ((scores[cid]["U"] + scores[cid]["W"]) / 2, cid))
        decoder = decoders[chosen]["decoder"]
        cost = task_blocks(decoder, coef, ctx["coefficient_split"], tokens["coefficient_split"], K)
        calibrated = channel.calibrate_nested(q_ref, cut_store, DELTA)
        solution = _solve(form, cost, calibrated, q_ref, fix)
        if not solution.get("feasible") or solution.get("params") is None:
            raise RuntimeError(f"round {r} nested LP unresolved; preserve private unit")
        p = solution["params"]
        _save_npz(round_root / "PARAMS.npz", B=p["B"], A=p["A"], eta=np.asarray(p["eta"]))
        _save_npz(round_root / "TASK_BLOCKS.npz", U_B=cost["U"]["B"], U_A=cost["U"]["A"],
                  W_B=cost["W"]["B"], W_A=cost["W"]["A"])
        _save_cut_blocks(round_root / "CALIBRATED_BLOCKS.npz", calibrated["cuts"])
        _write_json(round_root / "CALIBRATED_BANK.json", {
            "bank_sha256": calibrated["bank_sha256"], "rho": calibrated["rho"],
            "cuts": _meta(calibrated["cuts"]),
            "witness_eta1_d17_column_max_abs_difference": calibrated.get("witness_eta1_d17_column_max_abs_difference")})
        q_law = {n: channel.person_law(p["B"], p["A"], p["eta"], codes[n], ctx[n], tokens[n])
                 for n in SCIENCE_ROLES}
        nonalias, proj = {}, {}
        for n in ("coefficient_split", "inner_selection"):
            nonalias[n], proj[n] = channel.nonalias_diagnostic(
                q_law[n], codes[n], q_ref, role_dict[n]["weights"], role_dict[n]["households"])
        rescore = channel.projection_rescore(proj["coefficient_split"], cost, calibrated["cuts"])
        proj_sel = {v: proj["coefficient_split"][v][codes["inner_selection"]] for v in ("U", "W")}
        rescore["inner_selection_task_U_under_projection"] = task_scores_law(
            decoder, role_dict["inner_selection"], proj_sel["U"])["U"]
        rescore["inner_selection_task_W_under_projection"] = task_scores_law(
            decoder, role_dict["inner_selection"], proj_sel["W"])["W"]
        record = {"round": r, "status": "FIXED_BANK_FEASIBLE", "cut_count": len(cut_store),
                  "selected_decoder_id": chosen, "decoder_selection_scores": scores,
                  "solution": _jsonable_solution(solution),
                  "params_sha256": _sha_file(round_root / "PARAMS.npz"),
                  "inner_selection_fixed_decoder_task": task_scores_law(decoder, role_dict["inner_selection"], q_law["inner_selection"]),
                  "inner_check_fixed_decoder_task": task_scores_law(decoder, role_dict["inner_check"], q_law["inner_check"]),
                  "nonalias": nonalias, "within_t32_projection_rescore": rescore,
                  "elapsed_seconds": round(time.time() - t_start, 1),
                  "outer_labels_accessed": False,
                  "scope": "fixed-bank/fixed-decoder training diagnostics; refit audits are separate"}
        if fix and r == 0 and form == "U":
            q_bank = np.load(bank["root"] / "round0_t32" / "Q.npz")["Q"]
            record["t32_round0_max_abs_difference_vs_bank_lp"] = float(np.max(np.abs(p["B"] - q_bank)))
        _write_json(round_root / "ROUND.json", record)
        records.append(record)
        params_by_round.append(p)
        final = (cost, calibrated["cuts"])
        laws = {n: q_law[n] for n in SCIENCE_ROLES}
        _log("unit_round", variant=variant, round=r, seconds=record["elapsed_seconds"],
             objective=solution["replay"].get("objective"), tau=solution.get("tau"),
             eta=p["eta"], tv_d17=nonalias["coefficient_split"]["tv_to_d17"]["weighted_mean"])
    closing = _closing_refit(root, anchor, rounds, role_dict, laws, q_ref, final[0], cut_store,
                             ctx, tokens, K, bank)
    selection = select_final_round(params_by_round, records, final[0], closing["cuts"])
    selection["closing_refit"] = closing["record"]
    _write_json(root / "FINAL_BANK_SELECTION.json", selection)
    chosen_round = selection["selected_round"]
    release_spec = _release_spec(variant, anchor, K, fix, policy_bank.names, bank, root,
                                 f"round_r{chosen_round:02d}/PARAMS.npz")
    _write_json(root / "RELEASE_SPEC.json", release_spec)
    _write_json(root / "SELECTED.json", {"selected_round": chosen_round,
                                         "selection_rule": selection["selection_rule"],
                                         "release_spec_sha256": _sha_file(root / "RELEASE_SPEC.json")})
    receipt = {**expected, "status": "COMPLETE", "selected_round": chosen_round,
               "closing_refit": {k: closing["record"][k] for k in
                                 ("round_index", "new_attack_count", "cut_count", "bank_sha256")},
               "rounds_summary": [{"round": x["round"], "objective": x["solution"]["replay"].get("objective"),
                                   "tau": x["solution"].get("tau"),
                                   "eta": x["solution"]["sparsity"]["eta"]} for x in records],
               "artifact_sha256": _inventory(root)}
    _write_json(complete, receipt)
    _write_release_descriptor(root, variant, anchor)
    return receipt


def _closing_refit(root, anchor, rounds, role_dict, laws, q_ref, final_cost, cut_store,
                   ctx, tokens, K, bank):
    """Amendment M3.1: best responses on the LAST round's own law, then rebase.

    Same AR best-response routine/roles; seed convention round = last + 1. The
    enlarged, rebased bank is the final bank for the AR final-round rule and for
    DET_SEL.  No new decoder is fitted; the last round's task blocks are kept.
    """
    r_close = rounds + 1
    seed = SEED_BASE + 10000 * int(anchor) + 100 * r_close
    closing_root = Path(root) / "closing"
    specs = build_attack_bank_per_person(role_dict, f"candidate_r{rounds:02d}", laws,
                                         closing_root / "bank", seed + 50000,
                                         target_roles=tuple(bank["manifest"]["target_roles"]))
    coef = role_dict["coefficient_split"]
    fresh = nested_cuts(specs, coef, ctx["coefficient_split"], tokens["coefficient_split"], K)
    _save_cut_blocks(closing_root / "blocks.npz", fresh)
    _write_json(closing_root / "blocks.json", {"cuts": _meta(fresh)})
    calibrated = channel.calibrate_nested(q_ref, cut_store + fresh, DELTA)
    _save_npz(closing_root / "TASK_BLOCKS.npz", U_B=final_cost["U"]["B"], U_A=final_cost["U"]["A"],
              W_B=final_cost["W"]["B"], W_A=final_cost["W"]["A"])
    _save_cut_blocks(closing_root / "CALIBRATED_BLOCKS.npz", calibrated["cuts"])
    _write_json(closing_root / "CALIBRATED_BANK.json", {
        "bank_sha256": calibrated["bank_sha256"], "rho": calibrated["rho"],
        "cuts": _meta(calibrated["cuts"])})
    record = {"amendment": "M3.1", "round_index": r_close, "seed": seed + 50000,
              "source": f"candidate_r{rounds:02d}", "new_attack_count": len(specs),
              "cut_count": len(calibrated["cuts"]), "bank_sha256": calibrated["bank_sha256"],
              "rho": calibrated["rho"],
              "task_blocks": "last alternation round's selected decoder (no decoder refit)"}
    _write_json(closing_root / "CLOSING.json", record)
    return {"cuts": calibrated["cuts"], "record": record}


def select_final_round(params_by_round, records, final_cost, final_cuts):
    """AR rule: feasible on the last rebased bank, then min inner-selection task CE."""
    checks = []
    for r, p in enumerate(params_by_round):
        check = channel.check_feasible(p, final_cost, final_cuts)
        checks.append({"round": r, "feasible": check["feasible"],
                       "maximum_cut_violation": check["maximum_cut_violation"]})
    eligible = [rec for rec, c in zip(records, checks) if c["feasible"]]
    if not eligible:
        raise RuntimeError("no round satisfies the final rebased bank")
    chosen = min(eligible, key=lambda rec: ((rec["inner_selection_fixed_decoder_task"]["U"] +
                                             rec["inner_selection_fixed_decoder_task"]["W"]) / 2, rec["round"]))
    return {"selected_round": chosen["round"], "final_bank_checks": checks,
            "selection_rule": "minimum inner-selection balanced task CE among final-bank-feasible rounds; earlier round tie"}


def _release_spec(variant, anchor, K, fix, names, bank, unit_root, params_relative):
    unit_root = Path(unit_root).resolve()
    return {"schema": SCHEMA_RELEASE, "kind": "nested", "variant": variant, "anchor": int(anchor),
            "K": int(K), "eta_fixed_zero": bool(fix), "policy_columns": list(names),
            "params_relative": params_relative,
            "params_sha256": _sha_file(unit_root / params_relative),
            "bank_dir": str(bank["root"]),
            "bank_dir_relative": os.path.relpath(bank["root"], unit_root),
            "bank_manifest_sha256": bank["manifest_sha256"],
            "policy_bank_sha256": bank["manifest"]["policy_bank_sha256"],
            "contexts_sha256": bank["manifest"]["contexts_sha256"],
            "legal_inputs": list(policies.LEGAL_INPUTS),
            "wire": "H_A unchanged plus one keyed-persistent 17-token draw"}


# ---------------------------------------------------------------------------
# DET_SEL (amendment M1.3)
# ---------------------------------------------------------------------------

def deterministic_selector(cost, cuts, n_contexts, names):
    """Exhaustive one-policy-per-context assignment on fixed blocks."""
    M = len(names)
    rows = []
    for assignment in itertools.product(range(M), repeat=n_contexts):
        A = np.zeros((n_contexts, M))
        A[np.arange(n_contexts), assignment] = 1.
        params = {"B": np.zeros((32, 17)), "A": A, "eta": 1.}
        check = channel.check_feasible(params, cost, cuts)
        rows.append({"assignment": [names[m] for m in assignment], "index": list(assignment),
                     "objective": check["objective"], "objective_U": check["objective_U"],
                     "objective_W": check["objective_W"], "feasible": check["feasible"],
                     "maximum_cut_violation": check["maximum_cut_violation"],
                     "non_d17_contexts": int(sum(m != 0 for m in assignment))})
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("no deterministic assignment feasible (all-D17 witness should be)")
    best = min(feasible, key=lambda row: (row["objective"], row["non_d17_contexts"], row["index"]))
    return best, rows


def run_det_sel(variant, from_dir, output_dir, *, role_dict=None, q_ref=None):
    family = DET_VARIANTS[variant]
    src = Path(from_dir).resolve()
    receipt = json.loads((src / "COMPLETE.json").read_text())
    if receipt.get("family") != family or receipt.get("form") != "U":
        raise ValueError(f"{variant} must be evaluated from a completed {family}_U unit")
    if receipt.get("artifact_sha256") != _inventory(src):
        raise ValueError("source NM unit inventory differs")
    root = _private_root(output_dir)
    last = receipt["rounds"]
    round_root = src / "closing" if (src / "closing" / "CLOSING.json").is_file() else src / f"round_r{last:02d}"
    K, names = receipt["K"], receipt["policy_columns"]
    expected = {"schema": "pcrl-sc-det-sel-v1", "variant": variant, "anchor": receipt["anchor"],
                "K": K, "source_unit_complete_sha256": _sha_file(src / "COMPLETE.json"),
                "source_round": last, "smoke": receipt["smoke"], "outer_labels_accessed": False}
    if (root / "COMPLETE.json").exists():
        done = json.loads((root / "COMPLETE.json").read_text())
        if any(done.get(k) != v for k, v in expected.items()) or done.get("artifact_sha256") != _inventory(root):
            raise ValueError("completed DET_SEL unit differs")
        _write_release_descriptor(root, variant, done["anchor"])
        return done
    with np.load(round_root / "TASK_BLOCKS.npz") as t:
        cost = {v: {"B": t[f"{v}_B"].copy(), "A": t[f"{v}_A"].copy()} for v in ("U", "W")}
    meta = json.loads((round_root / "CALIBRATED_BANK.json").read_text())["cuts"]
    cuts = _load_cut_blocks(round_root / "CALIBRATED_BLOCKS.npz", meta)
    best, rows = deterministic_selector(cost, cuts, K, names)
    root.mkdir(parents=True, exist_ok=True)
    A = np.zeros((K, len(names)))
    A[np.arange(K), best["index"]] = 1.
    _save_npz(root / "PARAMS.npz", B=np.zeros((32, 17)), A=A, eta=np.asarray(1.))
    record = {**expected, "selected": best, "assignments_evaluated": len(rows),
              "feasible_assignments": int(sum(r["feasible"] for r in rows)),
              "assignments": rows,
              "rule": "lowest 0.5U+0.5W task loss among bank-feasible assignments; tie fewer non-D17 contexts, then lexicographic",
              "bank_source": str(round_root.relative_to(src)),
              "scope": "final-round decoder and final (M3.1 closing, if present) rebased bank of the source NM unit (coefficient_split)"}
    if role_dict is not None:
        bank = load_bank_dir(json.loads((src / "INPUTS.json").read_text())["bank_dir"])
        rows_c = role_dict["coefficient_split"]
        legal = policies.legal_inputs(rows_c)
        law = channel.person_law(np.zeros((32, 17)), A, 1., rows_c["token_codes"],
                                 bank["rules"][K].assign(legal), bank["policy_bank"].predict(legal))
        record["nonalias_coefficient_split"], _ = channel.nonalias_diagnostic(
            law, rows_c["token_codes"], q_ref, rows_c["weights"], rows_c["households"])
    _write_json(root / "DET_SEL.json", record)
    bank_info = {"root": Path(json.loads((src / "INPUTS.json").read_text())["bank_dir"]),
                 "manifest_sha256": receipt["bank_manifest_sha256"],
                 "manifest": json.loads((Path(json.loads((src / "INPUTS.json").read_text())["bank_dir"]) / "BANK_COMPLETE.json").read_text())}
    _write_json(root / "RELEASE_SPEC.json", _release_spec(variant, receipt["anchor"], K, False,
                                                          names, bank_info, root, "PARAMS.npz"))
    done = {**expected, "status": "COMPLETE", "selected_assignment": best["assignment"],
            "artifact_sha256": _inventory(root)}
    _write_json(root / "COMPLETE.json", done)
    _write_release_descriptor(root, variant, done["anchor"])
    return done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _check_smoke_path(path, smoke):
    worktree = Path(__file__).resolve().parents[2]
    target = Path(path).resolve()
    if smoke and target.is_relative_to(worktree):
        raise ValueError("--smoke outputs must stay outside the repository (use the scratchpad)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="fit", choices=("fit", "build-bank", "decide-k"))
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2))
    parser.add_argument("--variant", choices=tuple(VARIANTS) + tuple(DET_VARIANTS))
    parser.add_argument("--out")
    parser.add_argument("--bank")
    parser.add_argument("--from", dest="from_dir")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--nm4-k", type=int, choices=(2, 4))
    parser.add_argument("--index", default=str(Path(__file__).resolve().parents[2] / data.INDEX_RELATIVE))
    parser.add_argument("--banks", nargs=3, help="decide-k: bank dirs for anchors 0 1 2")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "decide-k":
        census = {str(a): json.loads((Path(d) / "contexts" / "CONTEXTS.json").read_text())["census_coefficient_split"]["4"]
                  for a, d in enumerate(args.banks)}
        print(json.dumps(contexts.fallback_decision(census), sort_keys=True))
        return
    if args.out is None or args.anchor is None:
        parser.error("--anchor and --out are required")
    _check_smoke_path(args.out, args.smoke)
    if args.smoke:
        args.rounds = min(args.rounds, 1)
    if args.command == "build-bank":
        role_dict, q_ref, hist, enc = load_roles(args.anchor, args.index, smoke=args.smoke)
        bank = build_bank(args.anchor, role_dict, q_ref, hist, enc, args.out, smoke=args.smoke)
        print(json.dumps({"status": "COMPLETE", "retained": bank["retained_policies"],
                          "manifest_sha256": bank["manifest_sha256"]}, sort_keys=True))
        return
    if args.variant is None:
        parser.error("--variant required")
    if args.variant in DET_VARIANTS:
        if not args.from_dir:
            parser.error("DET_SEL variants require --from <NM*_U unit dir>")
        role_dict, q_ref, _, _ = load_roles(args.anchor, args.index, smoke=args.smoke)
        done = run_det_sel(args.variant, args.from_dir, args.out, role_dict=role_dict, q_ref=q_ref)
        print(json.dumps({"status": done["status"], "selected": done["selected_assignment"]}, sort_keys=True))
        return
    if not args.bank:
        parser.error("fit requires --bank <frozen bank dir>")
    role_dict, q_ref, hist, _ = load_roles(args.anchor, args.index, smoke=args.smoke)
    receipt = run_unit(args.variant, args.anchor, role_dict, q_ref, hist, args.bank, args.out,
                       rounds=args.rounds, nm4_k=args.nm4_k, smoke=args.smoke)
    print(json.dumps({"status": receipt["status"], "selected_round": receipt["selected_round"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
