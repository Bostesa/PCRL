"""Richer deterministic comparators RD_TASK and RD_PRIV (shared-context release v1).

Owner: baseline/audit agent. These are *competitors* of the nested mixture (NM).
They are built to be as strong as the registered budget allows, never tuned to
lose. Every released object is a deterministic map from legal per-person inputs
{x (X_A 32), ha (H_A 4), teacher_p, residual, risk (11), token_codes (stored T0)}
to one of the 17 tokens of the common codebook. Never hb, labels, weights,
ids, households or roles.

Shared machinery (also imported by ``adv.py``):
  * role loading through the predecessor sanitized loaders (outer never opened);
  * the legal feature builder [X_A, H_A, logit(p), r, risk(11)] standardized on
    nuisance_train (DESIGN_SPEC §2);
  * a round-0 bank adapter (frozen decoder + frozen attack bank + round-0 T32
    dual prices) reading the NM ``fit_nm build-bank`` directory, or an AR-layout
    directory (``decoder_r00/FIT_DECODER.json`` + ``bank_r*/ATTACK_BANK.json``);
  * ``PersonBank``: per-person replay of every frozen attacker on
    coefficient_split, D17-calibrated floors rho-delta per role|weighting group
    (the AR ``calibrate_reference`` convention, evaluated person-wise so that it
    also applies to per-person laws), and feasibility checks;
  * best-response attacker refits on a candidate law (audit_fit, validated on
    inner_selection, standard slate), added to the bank and rebased (M1(d)).

All policies are built through the method implementer's stable M2 API in
``policies.py`` (paired oracle Delta(z) = g(z) - g(D17), ``fit_paired_oracle``,
``switched_policy_from_oracle``; ties and |gain| <= tau go to D17), so RD and
the NM policy columns share one oracle recipe.

RD_TASK (DESIGN_SPEC §4): d_1 (the NM bank's task_only column), then 3 Lloyd
rounds (cross-fitted decoders refit on the deterministic law under the coverage
rule (M1.2), paired task oracle refit, policy re-derived). Candidates are
(round, tau in TAU_GRID); inner_selection picks by a receiver slate fit on the
candidate's pure law.

RD_PRIV (DESIGN_SPEC §4): switched argmin of Delta_u - mu * Delta_p, where
Delta_p is the paired round-0 dual-price-weighted attacker loss. mu is the
smallest multiplier (mu=0, geometric bracket, <=12 bisection steps) whose policy
satisfies every current-bank cut on coefficient_split; if none does, the
registered fallback bisects the D17-switch threshold tau instead (D17 is the
feasible limit). 3 rounds of best-response attacker refit + rebase +
re-bisection follow; the last policy also receives a closing best response, so
every round policy is judged against attackers refit on itself (M1(d)).
inner_selection picks the round: lowest receiver task CE among final-bank-
feasible rounds; if none, RD_PRIV is D17 (flagged alias).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import policies

N_TOKENS = 17
DELTA = 0.001
PRIMAL_TOL = 1e-7
PROTECTED_ROLES = ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P")
CLASS_COUNT = {"SEX": 2, "RAC1P": 9, "same_residence": 2}
SCIENCE_ROLES = ("nuisance_train", "audit_fit", "coefficient_split",
                 "inner_selection", "inner_check")
LEGAL_INPUTS = ("x", "ha", "teacher_p", "residual", "risk", "token_codes")
FEATURE_NAMES = ([f"x_a_{j}" for j in range(32)] + [f"h_a_{j}" for j in range(4)]
                 + ["logit_teacher_p", "residual"] + [f"risk_{j}" for j in range(11)])
LLOYD_ROUNDS = 3
PRIV_ROUNDS = 3
BISECTION_STEPS = 12
TAU_GRID = (0.002, 0.01, 0.03, 0.1, 0.3)  # RD_TASK D17-shrinkage grid; first entry = policies.TAU
SEED_BASE = {"RD_TASK": 51000, "RD_PRIV": 52000, "ADV": 64000, "BR": 56000}  # disjoint from audit 26000+ and positive-control 27000+ ranges
SMOKE_FRACTION = 0.3


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def sha_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value) -> str:
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(str(a.shape).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_json(path, value) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return sha_file(path)


def valid_mask(rows, target) -> np.ndarray:
    y = np.asarray(rows["labels"][target])
    return (y >= 0) & (y < CLASS_COUNT[target])


def household_fold(households, salt: str, n_folds: int = 2) -> np.ndarray:
    return np.asarray([int.from_bytes(hashlib.sha256((salt + str(h)).encode()).digest()[:8], "big") % n_folds
                       for h in households], dtype=np.int64)


def onehot(tokens, n_tokens=N_TOKENS) -> np.ndarray:
    t = np.asarray(tokens, dtype=np.int64)
    out = np.zeros((len(t), n_tokens), dtype=np.float64)
    out[np.arange(len(t)), t] = 1.0
    return out


def d17_tokens(d17_map: np.ndarray, codes) -> np.ndarray:
    d17_map = np.asarray(d17_map, dtype=np.float64)
    if d17_map.shape != (32, N_TOKENS) or not np.allclose(d17_map.max(1), 1.0):
        raise ValueError("D17 must be the deterministic 32x17 historical map")
    return np.argmax(d17_map, axis=1)[np.asarray(codes, dtype=np.int64)]


def subset_rows(rows: Mapping, mask: np.ndarray) -> dict:
    mask = np.asarray(mask)
    out = {}
    for key, value in rows.items():
        if key == "labels":
            out[key] = {k: np.asarray(v)[mask] for k, v in value.items()}
        else:
            out[key] = np.asarray(value)[mask]
    return out


# ---------------------------------------------------------------------------
# data (sanitized 2018 inner roles only)
# ---------------------------------------------------------------------------

def _torch_threads():
    import torch
    torch.set_num_threads(1)


def study_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_roles(anchor: int, *, smoke: bool = False):
    """Five inner roles + D17 + historical Q, via ``fit_nm.load_roles`` (same rows as NM).

    Using the method implementer's loader guarantees the identical sanitized
    rows and the identical --smoke household subsample as the NM bank; the
    bank's role fingerprints are re-checked in ``check_bank_roles``.
    """
    from experiments.pcrl_task_aligned_cuts_v1 import data
    from . import fit_nm

    return fit_nm.load_roles(anchor, str(study_root() / data.INDEX_RELATIVE), smoke=smoke)


def check_bank_roles(bank, role_dict):
    """Refuse a bank fitted on different rows (fit_a role fingerprints)."""
    from experiments.pcrl_adaptive_release_v1 import fit_a

    expected = (bank.manifest or {}).get("role_input_sha256")
    if not expected:
        return {"checked": False}
    for name in SCIENCE_ROLES:
        if fit_a._role_fingerprint(role_dict[name]) != expected[name]:
            raise ValueError(f"bank was fitted on different {name} rows")
    return {"checked": True}


# ---------------------------------------------------------------------------
# legal features (reused from the method implementer's policies.py)
# ---------------------------------------------------------------------------

def raw_features(inputs: Mapping) -> np.ndarray:
    """[X_A(32), H_A(4), logit(p), r, risk(11)]; ``policies.policy_features`` (legal-key guard)."""
    from . import policies
    return policies.policy_features(inputs)


def legal_inputs(rows: Mapping) -> dict:
    from . import policies
    return policies.legal_inputs(rows)


class Scaler:
    """Thin callable wrapper of ``policies.FeatureStandardizer``."""

    def __init__(self, mean, scale):
        from . import policies
        self.std = policies.FeatureStandardizer(np.asarray(mean, dtype=np.float64),
                                                np.asarray(scale, dtype=np.float64))
        self.mean, self.scale = self.std.mean, self.std.scale

    @classmethod
    def fit(cls, z):
        from . import policies
        std = policies.FeatureStandardizer.fit(z)
        return cls(std.mean, std.scale)

    def __call__(self, z):
        return self.std.transform(z)


def feature_builder(nuisance_rows):
    scaler = Scaler.fit(raw_features(legal_inputs(nuisance_rows)))
    return scaler, "policies.FeatureStandardizer(policies.policy_features) fitted on nuisance_train"


# ---------------------------------------------------------------------------
# round-0 bank adapter
# ---------------------------------------------------------------------------

class Round0Bank:
    """Frozen round-0 decoder, attack specs/cuts and (optional) dual prices."""

    def __init__(self, decoder, decoder_record, specs, cut_meta, multipliers,
                 crossfit, source, policy_bank=None, manifest=None):
        self.policy_bank = policy_bank
        self.manifest = manifest
        self.decoder = decoder
        self.decoder_record = decoder_record
        self.specs = specs
        self.cut_meta = cut_meta
        self.multipliers = multipliers
        self.crossfit = crossfit
        self.source = source


def _load_decoder_receipt(receipt_path: Path):
    from experiments.pcrl_adaptive_release_v1 import fit_a

    receipt = json.loads(receipt_path.read_text())
    model_dir = (receipt_path.parent / receipt["selected_model_relative_directory"]).resolve()
    decoder = fit_a.load_frozen_decoder({"model_directory": str(model_dir),
                                         "selected_model_sha256": receipt["selected_model_sha256"]})
    return decoder, {"receipt": str(receipt_path), "receipt_sha256": sha_file(receipt_path),
                     "selected_model_sha256": receipt["selected_model_sha256"],
                     "model_directory": str(model_dir)}


def load_bank(bank_dir) -> Round0Bank:
    """Read a frozen round-0 bank directory without refitting anything.

    Primary layout: the NM ``fit_nm build-bank`` directory (BANK_COMPLETE.json,
    inventory-verified by ``fit_nm.load_bank_dir``): decoder_r00, bank_r00,
    round0_t32/DUALS.json, crossfit_decoder_f{0,1}, policies/. Fallback: an
    AR-layout directory (decoder_r00 + bank_r*), used by --smoke/test banks.
    """
    from experiments.pcrl_adaptive_release_v1 import fit_a

    root = Path(bank_dir).resolve()
    if (root / "BANK_COMPLETE.json").is_file():
        from . import fit_nm
        b = fit_nm.load_bank_dir(root)
        record = {k: v for k, v in b["decoder"].items() if k != "decoder"}
        specs = {spec["id"]: spec for spec in b["attack"]["attack_specs"]}
        cut_meta = [{"id": c["id"], "attack_id": c["id"].rsplit("/", 1)[0], "role": c["role"],
                     "weighting": c["weighting"]} for c in b["attack"]["cuts"]]
        multipliers = json.loads((root / "round0_t32" / "DUALS.json").read_text())["multipliers"]
        cross = [_load_decoder_receipt(root / f"crossfit_decoder_f{f}" / "FIT_DECODER.json") for f in (0, 1)]
        source = {"layout": "fit_nm BANK_COMPLETE", "dir": str(root),
                  "manifest_sha256": b["manifest_sha256"]}
        return Round0Bank(b["decoder"]["decoder"], record, specs, cut_meta, multipliers, cross, source,
                          policy_bank=b["full_policy_bank"], manifest=b["manifest"])
    receipts = sorted(root.glob("decoder_r00/FIT_DECODER.json"))
    if not receipts:
        raise FileNotFoundError("no frozen decoder receipt in bank directory")
    bank_dirs = sorted(p.parent for p in root.glob("bank_r*/ATTACK_BANK.json"))
    if not bank_dirs:
        raise FileNotFoundError("no frozen attack bank in bank directory")
    decoder, decoder_record = _load_decoder_receipt(receipts[0])
    specs, cut_meta, source = {}, [], {"layout": "AR-discovered (smoke/test)", "dir": str(root)}
    for bdir in bank_dirs:
        frozen = fit_a.load_frozen_bank(bdir)
        specs.update({spec["id"]: spec for spec in frozen["attack_specs"]})
        cut_meta += [{"id": c["id"], "attack_id": c["id"].rsplit("/", 1)[0], "role": c["role"],
                      "weighting": c["weighting"]} for c in frozen["cuts"]]
    dual_file = root / "round0_t32" / "DUALS.json"
    multipliers = json.loads(dual_file.read_text())["multipliers"] if dual_file.is_file() else None
    cross = [_load_decoder_receipt(root / f"crossfit_decoder_f{f}" / "FIT_DECODER.json")
             for f in (0, 1) if (root / f"crossfit_decoder_f{f}" / "FIT_DECODER.json").is_file()]
    return Round0Bank(decoder, decoder_record, specs, cut_meta, multipliers, cross, source)


def build_smoke_bank(role_dict, d17, q_hist, out_dir, seed, *, sources=("H", "D17")):
    """NOT a science bank: AR round-0 decoder + reduced-source attack bank for smoke/tests."""
    from experiments.pcrl_adaptive_release_v1 import fit_a

    out = Path(out_dir)
    fit_a.fit_frozen_decoder(role_dict, q_hist, d17, out / "decoder_r00", seed)
    initial = {"H": None, "D17": d17, "Q": q_hist, "coverage": (d17 + q_hist + 1 / 17) / 3}
    fit_a.build_attack_bank(role_dict, {s: initial[s] for s in sources},
                            out / "bank_r00", seed + 50000)
    write_json(out / "SMOKE_BANK_NOT_SCIENCE.json", {"sources": list(sources), "seed": seed})
    return load_bank(out)


# ---------------------------------------------------------------------------
# person-level bank
# ---------------------------------------------------------------------------

def attack_losses(spec, rows):
    from experiments.pcrl_adaptive_release_v1 import fit_a
    valid, losses = fit_a.person_loss_from_attack(spec, rows)
    return valid, np.asarray(losses, dtype=np.float64)


class PersonBank:
    """Frozen attackers replayed person-wise on one role; D17-calibrated floors.

    Cut value for law q: L_j(q) = sum_i c_ij sum_z q_iz loss_ij(z) on the
    target-valid rows (c = 1/n (U) or PWGTP/sum (W)). rho_g = min_j in g L_j(D17),
    floor = rho_g - delta; identical to AR ``calibrate_reference`` for T32 laws.
    """

    def __init__(self, rows, d17_map, delta=DELTA):
        self.rows = rows
        self.delta = float(delta)
        self.d17_law = onehot(d17_tokens(d17_map, rows["token_codes"]))
        self.weights = np.asarray(rows["weights"], dtype=np.float64)
        self.attacks = {}   # attack_id -> dict(valid, losses, role, target)
        self.cuts = []      # dicts id, attack_id, role, weighting, origin

    def add(self, specs: Mapping[str, Mapping], cut_meta: Sequence[Mapping], origin: str):
        for cut in cut_meta:
            aid = cut["attack_id"]
            if aid not in self.attacks:
                valid, losses = attack_losses(specs[aid], self.rows)
                if np.allclose(losses, losses[:, :1], atol=1e-12, rtol=0):
                    losses = losses[:, :1].copy()  # token-invariant (H-only) route
                w = self.weights[valid]
                self.attacks[aid] = {"valid": valid, "losses": losses,
                                     "role": specs[aid]["role"], "w": w / w.sum()}
            if any(c["id"] == cut["id"] for c in self.cuts):
                raise ValueError(f"duplicate cut {cut['id']}")
            self.cuts.append({**cut, "origin": origin})
        self._rho = None

    def attack_values(self, law) -> dict:
        law = np.asarray(law, dtype=np.float64)
        out = {}
        for aid, a in self.attacks.items():
            loss = a["losses"]
            if loss.shape[1] == 1:
                e = loss[:, 0]
            else:
                e = np.einsum("nz,nz->n", law[a["valid"]], loss)
            out[aid] = {"U": float(e.mean()), "W": float(np.dot(a["w"], e))}
        return out

    def cut_values(self, law) -> np.ndarray:
        values = self.attack_values(law)
        return np.array([values[c["attack_id"]][c["weighting"]] for c in self.cuts])

    def calibrate(self) -> dict:
        ref = self.cut_values(self.d17_law)
        rho = {}
        for value, cut in zip(ref, self.cuts):
            g = f"{cut['role']}|{cut['weighting']}"
            rho[g] = min(rho.get(g, np.inf), float(value))
        self._rho = rho
        self._floors = np.array([rho[f"{c['role']}|{c['weighting']}"] - self.delta for c in self.cuts])
        return rho

    def check(self, law) -> dict:
        if self._rho is None:
            self.calibrate()
        values = self.cut_values(law)
        violation = self._floors - values
        worst = int(np.argmax(violation))
        by_group = {}
        for v, cut in zip(violation, self.cuts):
            g = f"{cut['role']}|{cut['weighting']}"
            by_group[g] = max(by_group.get(g, -np.inf), float(v))
        return {"feasible": bool(violation.max() <= PRIMAL_TOL),
                "max_violation": float(violation.max()),
                "worst_cut": self.cuts[worst]["id"],
                "group_max_violation": by_group,
                "n_cuts": len(self.cuts), "rho": dict(self._rho)}


# ---------------------------------------------------------------------------
# decoders, cross-fitting, task losses
# ---------------------------------------------------------------------------

def coverage(law, d17_map, codes) -> np.ndarray:
    return (np.asarray(law, dtype=np.float64) + onehot(d17_tokens(d17_map, codes)) + 1 / N_TOKENS) / 3


def fit_decoder_on_law(fit_rows, fit_law, sel_rows, sel_law, out_dir, seed):
    """AR standard slate on (H_A, law) with inner_selection selection."""
    from experiments.pcrl_task_directed_release_v1.audits import fit_slate, load_candidate
    from experiments.pcrl_task_aligned_cuts_v1 import audit as tac_audit

    out = Path(out_dir)
    receipt = out / "FIT_DECODER.json"
    if not receipt.exists():
        mf, ms = valid_mask(fit_rows, "same_residence"), valid_mask(sel_rows, "same_residence")
        tac_audit.assert_household_disjoint(np.asarray(fit_rows["households"])[mf],
                                            np.asarray(sel_rows["households"])[ms])
        fit_slate(np.asarray(fit_rows["ha"])[mf], np.asarray(fit_law)[mf],
                  np.asarray(fit_rows["labels"]["same_residence"])[mf], np.asarray(fit_rows["weights"])[mf],
                  np.asarray(sel_rows["ha"])[ms], np.asarray(sel_law)[ms],
                  np.asarray(sel_rows["labels"]["same_residence"])[ms], np.asarray(sel_rows["weights"])[ms],
                  2, int(seed), out / "slate", slate="standard")
        meta = json.loads((out / "slate" / "slate.json").read_text())
        cid = meta["selection"]
        write_json(receipt, {"schema": 1, "seed": int(seed), "selected_candidate": cid,
                             "selected_model_relative_directory": f"slate/{cid}",
                             "selected_model_sha256": tac_audit.model_directory_hash(out / "slate" / cid),
                             "fit_law_sha256": sha_array(np.asarray(fit_law)),
                             "fit_people": int(mf.sum()), "selection_people": int(ms.sum())})
    return _load_decoder_receipt(receipt)


def task_losses(decoder, rows):
    from experiments.pcrl_adaptive_release_v1 import fit_a
    valid, losses = fit_a.task_person_losses(decoder, rows)
    return valid, np.asarray(losses, dtype=np.float64)


def task_score(decoder, rows, law) -> dict:
    valid, losses = task_losses(decoder, rows)
    e = np.einsum("nz,nz->n", np.asarray(law)[valid], losses)
    w = np.asarray(rows["weights"], dtype=np.float64)[valid]
    u, wt = float(e.mean()), float(np.dot(w / w.sum(), e))
    return {"U": u, "W": wt, "balanced": 0.5 * (u + wt), "n": int(valid.sum())}


def crossfit_folds(households) -> np.ndarray:
    """fit_nm fold rule (SHA256(FOLD_SALT|household) < 0.5 -> fold 1), so bank decoders line up."""
    from . import fit_nm
    return (fit_nm._unit_hash(households, fit_nm.FOLD_SALT) < 0.5).astype(np.int64)


def crossfit_task_losses(train_rows, train_law, sel_rows, sel_law, out_dir, seed,
                         provided=None):
    """2-fold household-grouped cross-fitted per-person task CE on nuisance_train (M1.2).

    Returns (valid mask, losses (n_valid,17)). Each fold's decoder is fit on the
    other fold with the same slate/selection rule (inner_selection).
    """
    folds = crossfit_folds(train_rows["households"])
    valid = valid_mask(train_rows, "same_residence")
    full = np.full((len(folds), N_TOKENS), np.nan)
    records = []
    for k in (0, 1):
        if provided:  # bank crossfit_decoder_f{k} was fitted on fold != k (fit_nm rule)
            decoder, record = provided[k]
        else:
            fit_idx = folds != k
            decoder, record = fit_decoder_on_law(
                subset_rows(train_rows, fit_idx), np.asarray(train_law)[fit_idx],
                sel_rows, sel_law, Path(out_dir) / f"crossfit_fold{k}", seed + 7 * (k + 1))
        score_idx = folds == k
        v, loss = task_losses(decoder, subset_rows(train_rows, score_idx))
        block = np.full((int(score_idx.sum()), N_TOKENS), np.nan)
        block[v] = loss
        full[score_idx] = block
        records.append(record)
    return valid, full[valid], records


# ---------------------------------------------------------------------------
# cost oracle (amendment M2: policies.fit_paired_oracle, per-person nats)
# ---------------------------------------------------------------------------

def paired_oracle(z, costs, base_tokens, households, seed, pwgtp):
    """``policies.fit_paired_oracle`` (paired Delta over the D17 token; M2), one thread."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return policies.fit_paired_oracle(z, costs, base_tokens, pwgtp, households, seed=int(seed))


@dataclass(frozen=True)
class MuPairedOracle(policies.PairedOracle):
    """Delta_hat(z) = Delta_u_hat(z) - mu * Delta_p_hat(z) (RD_PRIV Lagrangian, paired against D17).

    ``models`` regress the paired task cost, ``price_models`` the paired
    dual-price-weighted attack cost; both use the registered M2 recipe. Paired
    targets are linear in mu, so two fits replace one refit per bisection step.
    """
    price_models: tuple = ()
    mu: float = 0.0

    def predict_delta(self, standardized_features, base_tokens) -> np.ndarray:
        task = policies.PairedOracle(self.models).predict_delta(standardized_features, base_tokens)
        if self.mu == 0.0:
            return task
        price = policies.PairedOracle(self.price_models).predict_delta(standardized_features, base_tokens)
        return task - self.mu * price


def price_direction(cut_meta, multipliers) -> tuple[dict, dict]:
    """All-cut round-0 dual direction (scale 1) with the policies.py group fallback.

    Equal to ``policies.price_groups(...)['all_priced_x2']`` divided by 2, so RD_PRIV
    at mu=2 on the round-0 bank reproduces the d_4 pricing direction.
    """
    full = {c["id"]: float(multipliers.get(c["id"], 0.0)) for c in cut_meta}
    groups = policies.price_groups([{"id": c["id"], "role": c["role"]} for c in cut_meta], full)
    g = groups["all_priced_x2"]
    return {cid: v / g["scale"] for cid, v in g["multipliers"].items()}, \
        {"fallback_used": g["fallback_used"], "group_cuts": g["group_cuts"], "dual_sum": g["dual_sum"]}


def person_price_targets(n_rows, attacks: Mapping, direction: Mapping[str, float], cut_meta) -> np.ndarray:
    """p_i(z) = sum_j lambda_j a_ij(z) in per-person nats (policies.py convention)."""
    out = np.zeros((n_rows, N_TOKENS))
    for cut in cut_meta:
        lam = float(direction.get(cut["id"], 0.0))
        if lam <= 0:
            continue
        valid, losses = attacks[cut["attack_id"]]
        out[valid] += lam * (losses if losses.shape[1] == N_TOKENS else np.repeat(losses, N_TOKENS, 1))
    return out


def round0_dual_prices(bank: Round0Bank, coef_rows, d17_map) -> dict:
    """fit_b.fixed_bank_dual_prices on the round-0 T32 LP (only if the bank lacks DUALS.json)."""
    from experiments.pcrl_adaptive_release_v1 import fit_a, fit_b

    pb = PersonBank(coef_rows, d17_map)
    pb.add(bank.specs, bank.cut_meta, "round0")
    pb.calibrate()
    cost_pair = fit_a.task_cost_pair(bank.decoder, coef_rows)
    cuts = []
    for cut, floor in zip(pb.cuts, pb._floors):
        a = pb.attacks[cut["attack_id"]]
        losses = a["losses"] if a["losses"].shape[1] > 1 else np.repeat(a["losses"], N_TOKENS, 1)
        codes = np.asarray(coef_rows["token_codes"])[a["valid"]]
        pair = fit_a.aggregate_loss_coefficients(codes, losses,
                                                 np.asarray(coef_rows["weights"])[a["valid"]], 32)
        cuts.append({"id": cut["id"], "coeff": pair[cut["weighting"]], "floor": float(floor)})
    solved = fit_b.fixed_bank_dual_prices(0.5 * (cost_pair["U"] + cost_pair["W"]), cuts)
    return {"status": solved["status"], "multipliers": solved.get("multipliers") or {},
            "objective": solved.get("objective")}


# ---------------------------------------------------------------------------
# deterministic policies (frozen policies.SwitchedCostPolicy objects)
# ---------------------------------------------------------------------------

def switch_tokens(delta, base_tokens, tau) -> np.ndarray:
    """policies.switched_argmin on paired Delta: deviate iff min_z Delta < -tau; ties to D17."""
    return policies.switched_argmin(delta, base_tokens, tau)


class TokenPolicyAdapter:
    """Wrap a policy with ``.predict(inputs) -> tokens`` as a one-hot law callable."""

    def __init__(self, policy, label):
        self.policy = policy
        self.label = label

    def tokens(self, inputs):
        return np.asarray(self.policy.predict(inputs), dtype=np.int64)

    predict = tokens

    def __call__(self, inputs):
        return onehot(self.tokens(inputs))


def make_policy(name, d17, standardizer, oracle, tau, pricing=None) -> TokenPolicyAdapter:
    pol = policies.switched_policy_from_oracle(name, np.argmax(np.asarray(d17), axis=1), standardizer,
                                               oracle, tau=tau, pricing=pricing)
    return TokenPolicyAdapter(pol, name)


class _D17Law(TokenPolicyAdapter):
    """Exact D17 (reference receiver; RD_PRIV fallback), via policies.D17Policy."""

    def __init__(self, d17):
        super().__init__(policies.D17Policy("D17", np.argmax(np.asarray(d17), axis=1)), "D17")


def save_policy(policy, path) -> dict:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy, path, compress=3)
    return {"path": path.name, "sha256": sha_file(path)}


def load_policy(path, expected_sha):
    import joblib
    if sha_file(path) != expected_sha:
        raise ValueError("frozen deterministic policy hash mismatch")
    return joblib.load(path)


def load_law(unit_dir) -> Callable[[Mapping], np.ndarray]:
    """Pinned selected RD law: callable(legal inputs dict) -> one-hot (n,17)."""
    root = Path(unit_dir)
    selected = json.loads((root / "SELECTED.json").read_text())
    policy = load_policy(root / selected["policy"]["path"], selected["policy"]["sha256"])

    def law(inputs: Mapping) -> np.ndarray:
        extra = set(inputs) - set(LEGAL_INPUTS)
        if extra:
            raise ValueError(f"illegal release inputs {sorted(extra)}")
        return policy(inputs)
    law.policy = policy
    law.selected = selected
    return law


def policy_census(tokens, rows, d17_map) -> dict:
    ref = d17_tokens(d17_map, rows["token_codes"])
    changed = tokens != ref
    within = {}
    codes = np.asarray(rows["token_codes"])
    for t in np.unique(codes):
        m = codes == t
        within[int(t)] = int(len(np.unique(tokens[m])))
    return {"fraction_changed_vs_D17": float(changed.mean()),
            "affected_unique_households": int(len(np.unique(np.asarray(rows["households"])[changed]))),
            "token_histogram": np.bincount(tokens, minlength=N_TOKENS).tolist(),
            "distinct_tokens_within_T32_max": int(max(within.values())),
            "T32_states_with_within_state_variation": int(sum(v > 1 for v in within.values()))}


# ---------------------------------------------------------------------------
# best-response attackers on a candidate law (M1(d))
# ---------------------------------------------------------------------------

def fit_best_response(role_dict, law_fn: Callable[[Mapping], np.ndarray], out_dir, seed, tag):
    """A/AB x SEX/RAC1P release slates on audit_fit (validated on inner_selection).

    Mirrors AR round r>0 (fit_a.build_attack_bank with one candidate source): A/S
    cuts come from the A slate, AB/S cuts from the A and AB slates. Returns
    (specs by id, cut metadata).
    """
    from experiments.pcrl_task_aligned_cuts_v1 import audit as tac_audit

    fit_rows, sel_rows = role_dict["audit_fit"], role_dict["inner_selection"]
    fit_law = law_fn(legal_inputs(fit_rows))
    sel_law = law_fn(legal_inputs(sel_rows))
    root = Path(out_dir)
    registries = {}
    counter = 0
    for target in ("RAC1P", "SEX"):
        for view in ("A", "AB"):
            registries[(view, target)] = tac_audit.fit_role_slate(
                fit_rows, fit_law, sel_rows, sel_law, f"attack:{view}/{target}",
                root / "slates" / f"{view}_{target}", int(seed) + 1000 * counter,
                release_id=tag, slate="standard")
            counter += 1
    specs, cuts = {}, []
    for role in PROTECTED_ROLES:
        role_view, target = role.split("/")
        for source_view in (("A",) if role_view == "A" else ("A", "AB")):
            registry = registries[(source_view, target)]
            for cid, model in sorted(registry["models"].items()):
                aid = f"{role}/from_{source_view}/{tag}/{cid}"
                specs[aid] = {"id": aid, "role": role, "target": target, "source_view": source_view,
                              "wire": "release", "training_source": tag,
                              "model_directory": model["model_directory"],
                              "model_sha256": model["model_sha256"],
                              "class_order": list(range(CLASS_COUNT[target]))}
                for weighting in ("U", "W"):
                    cuts.append({"id": f"{aid}/{weighting}", "attack_id": aid,
                                 "role": role, "weighting": weighting})
    write_json(root / "BEST_RESPONSE.json", {
        "tag": tag, "seed": int(seed), "fit_role": "audit_fit", "selection_role": "inner_selection",
        "fit_law_sha256": sha_array(fit_law), "n_specs": len(specs), "n_cuts": len(cuts),
        "specs": [{k: v for k, v in s.items() if k != "model_directory"} for s in specs.values()]})
    return specs, cuts


# ---------------------------------------------------------------------------
# RD_TASK
# ---------------------------------------------------------------------------

def nm_task_only_policy(bank: Round0Bank):
    """d_1 ('task_only', a SwitchedCostPolicy) from the frozen NM policy bank, if present."""
    pb = bank.policy_bank
    if pb is None or "task_only" not in pb.names:
        return None
    d1 = pb.policies[pb.names.index("task_only")]
    if not isinstance(getattr(d1, "oracle", None), policies.PairedOracle):
        return None  # pre-M2 bank: refit the same recipe instead of mixing oracle versions
    return d1, {"sha256": (bank.manifest or {}).get("policy_bank_sha256")}


def receiver_score(policy, ntr, sel, out_dir, seed):
    """Inner_selection task CE of a law under a receiver slate fitted on that PURE law.

    The receiver mirrors the independent audit's utility probe (fit on the
    released law itself, standard slate, selected on inner_selection). The
    coverage-rule decoders are used only to price tokens for the cost oracle;
    scoring a re-assigned deterministic policy with a decoder trained on a D17
    mixture would understate its utility.
    """
    law_tr, law_sel = policy(legal_inputs(ntr)), policy(legal_inputs(sel))
    decoder, record = fit_decoder_on_law(ntr, law_tr, sel, law_sel, out_dir, seed)
    return task_score(decoder, sel, law_sel), record


def run_rd_task(anchor, role_dict, d17, bank: Round0Bank, bank_dir, out_dir, *,
                rounds=LLOYD_ROUNDS, tau_grid=TAU_GRID, log=print):
    """RD_TASK: d_1 then Lloyd rounds; inner_selection picks (round, switch tau).

    Each round's paired task oracle (M2) is thresholded at every tau in
    ``tau_grid`` (deviate only where the predicted paired gain exceeds tau);
    every variant is scored by a pure-law receiver on inner_selection. The best
    tau of a round seeds the next Lloyd round. tau_grid=(policies.TAU,) is the
    literal DESIGN_SPEC recipe.
    """
    out = Path(out_dir)
    seed0 = SEED_BASE["RD_TASK"] + 1000 * anchor
    ntr, sel, coef = role_dict["nuisance_train"], role_dict["inner_selection"], role_dict["coefficient_split"]
    scaler, scaler_source = feature_builder(ntr)
    z_tr = scaler(raw_features(legal_inputs(ntr)))
    base_tr = d17_tokens(d17, ntr["token_codes"])
    hh, pw = np.asarray(ntr["households"]), np.asarray(ntr["weights"], dtype=np.float64)
    crossfit_provided = bank.crossfit if len(bank.crossfit) == 2 else None
    nm_d1 = nm_task_only_policy(bank)
    records, policy = [], None
    receiver_seed = seed0 + 3  # common random numbers: identical laws get identical receiver scores
    d17_score, _ = receiver_score(_D17Law(d17), ntr, sel, out / "d17_receiver", receiver_seed)
    for r in range(rounds + 1):
        t0 = time.perf_counter()
        seed = seed0 + 100 * r
        if r == 0 and nm_d1 is not None:
            d1, pin = nm_d1
            oracle, standardizer = d1.oracle, d1.standardizer
            origin, cost_record = f"NM policy bank d_1 (task_only) oracle, sha {pin['sha256']}", None
        else:
            if policy is None:  # round 0 without an NM d_1: same recipe as NM d_1
                law_tr, law_sel = role_dict["_round0_law_ntr"], role_dict["_round0_law_sel"]
                provided = crossfit_provided
            else:  # Lloyd: cross-fit decoders on the previous round's deterministic law
                law_tr = coverage(policy(legal_inputs(ntr)), d17, ntr["token_codes"])
                law_sel = coverage(policy(legal_inputs(sel)), d17, sel["token_codes"])
                provided = None
            valid, losses, _ = crossfit_task_losses(
                ntr, law_tr, sel, law_sel, out / f"round_r{r:02d}", seed, provided=provided)
            oracle, cost_record = paired_oracle(z_tr[valid], losses, base_tr[valid], hh[valid], seed, pw[valid])
            standardizer = scaler.std
            origin = "cross-fitted paired task oracle (policies.fit_paired_oracle, M2)" + (
                " using bank cross-fit decoders" if provided else "")
        round_best = None
        for k, tau in enumerate(tau_grid):
            cand = make_policy(f"RD_TASK_r{r}_tau{tau:g}", d17, standardizer, oracle, tau,
                               {"variant": "RD_TASK", "round": r})
            tag = f"round_r{r:02d}/tau_{k}"
            score, dec_record = receiver_score(cand, ntr, sel, out / tag / "receiver", receiver_seed)
            pinfo = save_policy(cand, out / tag / "policy.joblib")
            rec = {"round": r, "tau": tau, "origin": origin, "policy": {**pinfo, "path": f"{tag}/policy.joblib"},
                   "receiver_decoder": dec_record, "inner_selection_task": score,
                   "inner_selection_task_minus_D17_receiver": score["balanced"] - d17_score["balanced"],
                   "census_coefficient_split": policy_census(cand.tokens(legal_inputs(coef)), coef, d17),
                   "census_nuisance_train": policy_census(cand.tokens(legal_inputs(ntr)), ntr, d17),
                   "cost_regression": cost_record if k == 0 else "same oracle as tau_0"}
            write_json(out / tag / "CANDIDATE.json", rec)
            records.append(rec)
            if round_best is None or score["balanced"] < round_best[0]:
                round_best = (score["balanced"], cand)
            log(f"[RD_TASK a{anchor}] round {r} tau {tau:g}: inner task {score['balanced']:.5f} "
                f"(D17 receiver {d17_score['balanced']:.5f}); changed "
                f"{rec['census_coefficient_split']['fraction_changed_vs_D17']:.3f}")
        policy = round_best[1]
        log(f"[RD_TASK a{anchor}] round {r} done in {time.perf_counter() - t0:.1f}s")
    best = min(records, key=lambda rec: (rec["inner_selection_task"]["balanced"], rec["round"], rec["tau"]))
    pb = PersonBank(coef, d17)  # diagnostic only (not a selection input)
    pb.add(bank.specs, bank.cut_meta, "round0")
    diag = []
    for rec in records:
        pol = load_policy(out / rec["policy"]["path"], rec["policy"]["sha256"])
        chk = pb.check(pol(legal_inputs(coef)))
        diag.append({"round": rec["round"], "tau": rec["tau"], "feasible": chk["feasible"],
                     "max_violation": chk["max_violation"]})
    selected = {"schema": "pcrl-sc-rd-v1", "variant": "RD_TASK", "anchor": anchor,
                "selected_round": best["round"], "selected_tau": best["tau"], "policy": best["policy"],
                "selection_rule": "lowest inner_selection balanced task CE under a receiver slate fit on the candidate's pure deterministic law, over (round, tau in tau_grid); ties: earlier round, smaller tau",
                "oracle": "policies.fit_paired_oracle (M2 paired Delta over D17); policies.switched_policy_from_oracle",
                "tau_grid": list(tau_grid), "feature_builder": scaler_source,
                "frozen_round0_bank_feasibility_diagnostic": diag,
                "preflight_M1_4": {"D17_pure_receiver_inner_selection": d17_score,
                                   "selected_minus_D17_balanced": best["inner_selection_task"]["balanced"] - d17_score["balanced"],
                                   "note": "each law scored by a standard-slate receiver fit on its own pure law (nuisance_train), selected on inner_selection"},
                "candidates": [{k: v for k, v in rec.items() if k != "receiver_decoder"} for rec in records],
                "outer_labels_accessed": False}
    write_json(out / "SELECTED.json", selected)
    return selected


# ---------------------------------------------------------------------------
# RD_PRIV
# ---------------------------------------------------------------------------

def bisect_increasing(policy_at: Callable[[float], np.ndarray], feasible: Callable[[np.ndarray], dict],
                      *, first, start, steps=BISECTION_STEPS, factor=4.0, max_expand=12):
    """Smallest feasible value of a parameter whose feasibility is ~monotone increasing.

    Tests ``first``; if infeasible, brackets geometrically from ``start`` (x factor,
    <= max_expand), then <= steps geometric bisections. Returns (value or None,
    trace, status); the trace records every evaluation (non-monotonicity visible).
    """
    trace = []

    def test(v):
        chk = feasible(policy_at(v))
        trace.append({"value": float(v), "feasible": chk["feasible"], "max_violation": chk["max_violation"]})
        return chk["feasible"]

    if test(first):
        return float(first), trace, "first_feasible"
    lo, hi = float(first), float(start)
    expanded = 0
    while not test(hi):
        lo, hi = hi, hi * factor
        expanded += 1
        if expanded >= max_expand:
            return None, trace, "no_feasible_value_in_bracket"
    for _ in range(steps):
        mid = float(np.sqrt(lo * hi)) if lo > 0 else hi / factor
        if test(mid):
            hi = mid
        else:
            lo = mid
    return hi, trace, "bisection"


def bisect_mu(policy_at, feasible, *, steps=BISECTION_STEPS, start=1.0, max_expand=12):
    """DESIGN_SPEC mu search: mu=0, geometric bracket (x4) from ``start``, <=12 bisections."""
    mu, trace, status = bisect_increasing(policy_at, feasible, first=0.0, start=start,
                                          steps=steps, max_expand=max_expand)
    return mu, [{"mu": t["value"], **{k: v for k, v in t.items() if k != "value"}} for t in trace], \
        {"first_feasible": "mu0_feasible", "no_feasible_value_in_bracket": "no_feasible_mu_in_bracket"}.get(status, status)


def run_rd_priv(anchor, role_dict, d17, bank: Round0Bank, out_dir, *, rounds=PRIV_ROUNDS,
                tau=None, direction="round0", log=print):
    """RD_PRIV (see module doc). Policy: switched argmin of Delta_u - mu * Delta_p (paired, M2).

    Per round: spec mu-bisection on the current bank; if no mu is feasible
    (frozen-attacker prices can point away from D17), the registered fallback
    keeps the least-violating traced mu and bisects the D17-switch threshold
    tau upward (nested deviation sets; tau -> inf is D17, always feasible).
    """
    tau0 = policies.TAU if tau is None else float(tau)
    out = Path(out_dir)
    seed0 = SEED_BASE["RD_PRIV"] + 1000 * anchor
    ntr, sel, coef = role_dict["nuisance_train"], role_dict["inner_selection"], role_dict["coefficient_split"]
    scaler, scaler_source = feature_builder(ntr)
    z_tr = scaler(raw_features(legal_inputs(ntr)))
    z_coef = scaler(raw_features(legal_inputs(coef)))
    base_tr = d17_tokens(d17, ntr["token_codes"])
    base_coef = d17_tokens(d17, coef["token_codes"])
    hh, pw = np.asarray(ntr["households"]), np.asarray(ntr["weights"], dtype=np.float64)
    t0 = time.perf_counter()
    if bank.multipliers:
        raw_mult, dual_source = dict(bank.multipliers), "bank directory (round0_t32/DUALS.json)"
    else:
        dual = round0_dual_prices(bank, coef, d17)
        raw_mult, dual_source = dual["multipliers"], f"recomputed fit_b.fixed_bank_dual_prices ({dual['status']})"
    lam, lam_record = price_direction(bank.cut_meta, raw_mult)
    valid, losses, _ = crossfit_task_losses(
        ntr, role_dict["_round0_law_ntr"], sel, role_dict["_round0_law_sel"], out / "crossfit", seed0,
        provided=bank.crossfit if len(bank.crossfit) == 2 else None)
    u_oracle, u_record = paired_oracle(z_tr[valid], losses, base_tr[valid], hh[valid], seed0, pw[valid])
    attacks_ntr = {}
    for cut in bank.cut_meta:
        if lam.get(cut["id"], 0) > 0 and cut["attack_id"] not in attacks_ntr:
            attacks_ntr[cut["attack_id"]] = attack_losses(bank.specs[cut["attack_id"]], ntr)
    # price costs on the task-valid rows so both oracles see the same people
    p_costs = person_price_targets(len(z_tr), attacks_ntr, lam, bank.cut_meta)
    p_oracle, p_record = paired_oracle(z_tr[valid], p_costs[valid], base_tr[valid], hh[valid], seed0 + 50, pw[valid])
    setup_seconds = time.perf_counter() - t0
    du_coef = u_oracle.predict_delta(z_coef, base_coef)
    dp_coef = p_oracle.predict_delta(z_coef, base_coef)
    pb = PersonBank(coef, d17)
    pb.add(bank.specs, bank.cut_meta, "round0")
    all_specs = dict(bank.specs)
    records, policies_by_round = [], []
    cur_p_oracle, cur_dp_coef, cur_lam = p_oracle, dp_coef, dict(lam)
    for r in range(rounds + 1):
        t1 = time.perf_counter()
        pb.calibrate()

        def policy_mu(mu):
            return onehot(switch_tokens(du_coef - mu * cur_dp_coef, base_coef, tau0))
        mu, trace, status = bisect_mu(policy_mu, pb.check)
        tau_r, tau_trace = tau0, []
        if mu is None:
            mu = min(trace, key=lambda t: (t["max_violation"], t["mu"]))["mu"]
            g_coef = du_coef - mu * cur_dp_coef

            def policy_tau(t):
                return onehot(switch_tokens(g_coef, base_coef, t))
            tau_r, tau_trace, tau_status = bisect_increasing(policy_tau, pb.check, first=tau0, start=4 * tau0,
                                                             max_expand=16)
            status += f"; fallback: least-violating mu={mu:.4g}, tau bisection -> {tau_status}"
            if tau_r is None:  # D17 witness is feasible, so this is numerically unreachable
                tau_r = float("inf")
                status += " (tau=inf: D17)"
        oracle = MuPairedOracle(u_oracle.models, cur_p_oracle.models, float(mu))
        policy = make_policy(f"RD_PRIV_r{r}", d17, scaler.std, oracle, tau_r,
                             {"variant": "RD_PRIV", "round": r, "mu": float(mu), "direction": direction})
        pinfo = save_policy(policy, out / f"round_r{r:02d}" / "policy.joblib")
        policies_by_round.append(policy)
        law_coef = policy(legal_inputs(coef))
        if not np.array_equal(np.argmax(law_coef, 1), switch_tokens(du_coef - mu * cur_dp_coef, base_coef, tau_r)):
            raise AssertionError("frozen RD_PRIV policy differs from the bisected policy")
        current_check = pb.check(law_coef)
        specs, cuts = fit_best_response(role_dict, policy, out / f"round_r{r:02d}" / "best_response",
                                        SEED_BASE["BR"] + 1000 * anchor + 100 * r, f"RD_PRIV_r{r:02d}")
        all_specs.update(specs)
        pb.add(all_specs, cuts, f"BR_r{r:02d}")
        if direction == "augmented" and r < rounds:
            # NOT the registered default: price new BR cuts at the mean round-0 price of their role
            by_role = {}
            for cut in bank.cut_meta:
                by_role.setdefault(cut["role"], []).append(lam.get(cut["id"], 0.0))
            for cut in cuts:
                cur_lam[cut["id"]] = float(np.mean(by_role[cut["role"]]))
                attacks_ntr.setdefault(cut["attack_id"], attack_losses(all_specs[cut["attack_id"]], ntr))
            meta = list(bank.cut_meta) + [c for c in pb.cuts if c["origin"] != "round0"]
            costs = person_price_targets(len(z_tr), attacks_ntr, cur_lam, meta)
            cur_p_oracle, _ = paired_oracle(z_tr[valid], costs[valid], base_tr[valid], hh[valid],
                                            seed0 + 51 + r, pw[valid])
            cur_dp_coef = cur_p_oracle.predict_delta(z_coef, base_coef)
        receiver_task, _ = receiver_score(policy, ntr, sel, out / f"round_r{r:02d}" / "receiver",
                                          seed0 + 3)  # common receiver seed across rounds
        rec = {"round": r, "mu": mu, "tau": tau_r, "bisection_status": status, "bisection_trace": trace,
               "tau_trace": tau_trace, "policy": {**pinfo, "path": f"round_r{r:02d}/policy.joblib"},
               "check_on_bank_used_for_bisection": {k: v for k, v in current_check.items() if k != "rho"},
               "inner_selection_task_receiver": receiver_task,
               "census_coefficient_split": policy_census(np.argmax(law_coef, 1), coef, d17),
               "seconds": time.perf_counter() - t1}
        write_json(out / f"round_r{r:02d}" / "ROUND.json", rec)
        records.append(rec)
        log(f"[RD_PRIV a{anchor}] round {r}: mu={mu:.4g} tau={tau_r:.4g} ({status}); inner task "
            f"{receiver_task['balanced']:.5f}; changed "
            f"{rec['census_coefficient_split']['fraction_changed_vs_D17']:.3f}; bank cuts {len(pb.cuts)}; {rec['seconds']:.1f}s")
    pb.calibrate()
    for rec, pol in zip(records, policies_by_round):
        chk = pb.check(pol(legal_inputs(coef)))
        rec["final_bank_check"] = {k: v for k, v in chk.items() if k != "rho"}
    # M4: selection set = rounds ∪ {exact D17 witness}; the witness is scored by the same
    # pure-law receiver recipe and common seed, and is feasible on any bank by construction.
    witness_task, _ = receiver_score(_D17Law(d17), ntr, sel, out / "d17_receiver", seed0 + 3)
    witness_check = pb.check(pb.d17_law)
    if not witness_check["feasible"]:
        raise AssertionError("D17 witness infeasible on its own calibrated bank")
    feasible = [rec for rec in records if rec["final_bank_check"]["feasible"]]
    witness = {"round": None, "witness": True, "inner_selection_task_receiver": witness_task,
               "final_bank_check": {k: v for k, v in witness_check.items() if k != "rho"}}
    best = min(feasible + [witness], key=lambda rec: (rec["inner_selection_task_receiver"]["balanced"],
                                                      rec["round"] is None, rec["round"] or 0))
    if best is witness:
        # exact D17 (one-hot D17 law; the audit collapses it as an alias of D17)
        policy_pin = {**save_policy(_D17Law(d17), out / "d17_witness" / "policy.joblib"),
                      "path": "d17_witness/policy.joblib"}
        flag = "WITNESS_FALLBACK" if not feasible else "WITNESS_SELECTED_BY_RULE"
        selected_round = None
    else:
        flag, policy_pin, selected_round = None, best["policy"], best["round"]
    selected = {"schema": "pcrl-sc-rd-v1", "variant": "RD_PRIV", "anchor": anchor,
                "selected_round": selected_round, "policy": policy_pin, "flag": flag,
                "selection_rule": "M4: selection set = rounds ∪ {exact D17 witness}; lowest inner_selection balanced task CE (receiver slate fit on the member's pure law, common seed) among members feasible on the final bank (round-0 + best-response refits on every round policy, incl. itself); ties: round before witness; witness only -> D17 exactly (WITNESS_FALLBACK)",
                "witness": witness, "witness_selected": best is witness,
                "oracle": "two policies.fit_paired_oracle fits (paired task u and paired price lambda.a, M2) combined as Delta_u - mu*Delta_p",
                "oracle_records": {"task": u_record, "price": p_record},
                "direction": direction, "dual_source": dual_source, "price_direction": lam_record,
                "nonzero_price_cuts": int(sum(v > 0 for v in lam.values())),
                "feature_builder": scaler_source, "base_switch_tau": tau0,
                "final_bank_cut_count": len(pb.cuts), "final_rho": pb._rho,
                "setup_seconds": setup_seconds, "rounds": records,
                "outer_labels_accessed": False}
    write_json(out / "SELECTED.json", selected)
    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def attach_round0_laws(role_dict, d17, q_hist):
    """Round-0 coverage law (Q_hist + D17 + U)/3 for nuisance/selection cross-fits."""
    for name, key in (("nuisance_train", "_round0_law_ntr"), ("inner_selection", "_round0_law_sel")):
        codes = np.asarray(role_dict[name]["token_codes"])
        role_dict[key] = (np.asarray(q_hist)[codes] + np.asarray(d17)[codes] + 1 / N_TOKENS) / 3
    return role_dict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--anchor", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--variant", choices=("RD_TASK", "RD_PRIV", "SMOKE_BANK"), required=True)
    parser.add_argument("--bank", required=True, help="frozen round-0 bank dir (fit_nm build-bank)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--direction", choices=("round0", "augmented"), default="round0")
    parser.add_argument("--switch-tau", type=float, default=None, help="RD_TASK: single tau instead of TAU_GRID; RD_PRIV: base tau (default policies.TAU)")
    args = parser.parse_args(argv)
    _torch_threads()
    t0 = time.perf_counter()
    role_dict, d17, q_hist, _ = load_roles(args.anchor, smoke=args.smoke)
    attach_round0_laws(role_dict, d17, q_hist)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.variant == "SMOKE_BANK":
        if not args.smoke:
            raise SystemExit("SMOKE_BANK is for --smoke only; science banks come from fit_nm build-bank")
        build_smoke_bank(role_dict, d17, q_hist, args.bank, SEED_BASE["RD_TASK"] + 999)
        print(f"smoke bank written to {args.bank} in {time.perf_counter()-t0:.1f}s")
        return
    bank = load_bank(args.bank)
    roles_check = check_bank_roles(bank, role_dict)
    rounds = args.rounds if args.rounds is not None else (1 if args.smoke else None)
    if args.variant == "RD_TASK":
        grid = TAU_GRID if args.switch_tau is None else (args.switch_tau,)
        result = run_rd_task(args.anchor, role_dict, d17, bank, args.bank, out,
                             rounds=rounds if rounds is not None else LLOYD_ROUNDS, tau_grid=grid)
    else:
        result = run_rd_priv(args.anchor, role_dict, d17, bank, out,
                             rounds=rounds if rounds is not None else PRIV_ROUNDS,
                             tau=args.switch_tau, direction=args.direction)
    write_json(out / "RUN.json", {"variant": args.variant, "anchor": args.anchor, "smoke": args.smoke,
                                  "wall_seconds": time.perf_counter() - t0,
                                  "bank": bank.source, "bank_role_fingerprints": roles_check,
                                  "selected_round": result["selected_round"],
                                  "argv": sys.argv[1:] if argv is None else list(argv)})
    print(f"{args.variant} a{args.anchor}: selected round {result['selected_round']} "
          f"in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    # Dispatch through the package module so pickled policies reference
    # ``experiments.pcrl_shared_context_release_v1.rd`` classes, not ``__main__``.
    from experiments.pcrl_shared_context_release_v1 import rd as _rd
    _rd.main()
