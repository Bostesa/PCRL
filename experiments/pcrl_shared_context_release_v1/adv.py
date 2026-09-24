"""ADV_B1/ADV_B2: PPAN-style adversarial categorical 17-token policy (competitor).

Tripathy, Wang & Ishwar (arXiv 1712.07008) style privatizer: an encoder MLP maps
the legal features [X_A, H_A, logit(p), r, risk(11)] plus onehot(D17 token of the
stored T0 code) to a softmax over the SAME
17-token codebook; a receiver decoder MLP (H_A, onehot z) -> same_residence and
four adversary MLPs A/SEX, A/RAC1P on (H_A, z) and AB/SEX, AB/RAC1P on
(H_A, H_B, z) are trained jointly. All losses use the EXACT expectation over
the 17 tokens (no sampling):

    encoder/decoder:  E_z~q[CE_Y]  - beta * mean_k E_z~q[CE_{S_k}]      (nuisance_train ∪ coefficient_split)
    adversaries:      E_z~q[CE_{S_k}]                                    (audit_fit)

with 5 adversary Adam steps per encoder step, weight decay, adversary warm
starts, and an encoder warm start at a smoothed D17 imitation (so training
starts at the reference release). Betas {0.5, 2.0}. CPU torch, one thread,
fixed seeds.

Checkpoint selection (inner_selection, DESIGN_SPEC §4 + amendment M1(d)):
every epoch's law is checked on coefficient_split against the frozen round-0
bank; early stopping uses that key. The K best epochs by
(frozen-bank infeasible, max violation or inner task CE) are shortlisted, a
best-response attacker slate is refit on each shortlisted law (audit_fit,
validated on inner_selection), and all of them join the final bank. Amendment
M4: the selection set is the shortlist plus the exact D17 witness (feasible by
construction; scored with the warm-start decoder/adversaries); the unit's rule
picks among final-bank-feasible members. If no checkpoint is feasible the unit
is D17 exactly (flag WITNESS_FALLBACK) and ``load_law`` returns the one-hot D17
law, so the audit collapses it as an alias. The released object is the per-person softmax law, with entries below
PRUNE zeroed and rows renormalized (registered; bounds the per-person support
that the exact-expansion hist_gb auditors are scaled by).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from . import rd

WIDTH = 64
LR = 3e-4          # set after the 30% a0 smoke diagnostic (1e-3/1e-4 overfit within 5 epochs)
WEIGHT_DECAY = 1e-3
DECODER_WARM_STEPS = 300
ADV_WARM_STEPS = 300
BATCH = 256
ADV_STEPS = 5
MAX_EPOCHS = 300
MIN_EPOCHS = 30
PATIENCE = 40
WARM_D17_MASS = 0.97  # encoder warm start target: 0.97 * onehot(D17) + 0.03 * uniform
SHORTLIST = 4
PRUNE = 1e-3
BETAS = (0.5, 2.0)
ADV_ROLES = (("A", "SEX"), ("A", "RAC1P"), ("AB", "SEX"), ("AB", "RAC1P"))


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def build_modules(n_features, seed):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(int(seed))

    def mlp(d_in, d_out):
        return nn.Sequential(nn.Linear(d_in, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(),
                             nn.Linear(WIDTH, d_out))
    encoder = mlp(n_features, rd.N_TOKENS)
    decoder = mlp(4 + rd.N_TOKENS, 2)
    adversaries = {f"{v}/{t}": mlp((4 if v == "A" else 6) + rd.N_TOKENS, rd.CLASS_COUNT[t])
                   for v, t in ADV_ROLES}
    return encoder, decoder, adversaries


def expected_ce(head, h, q, y, weight, mask):
    """sum_i w_i sum_z q_iz CE(y_i, head(h_i, z)) / sum_i w_i over rows with mask."""
    torch = _torch()
    idx = torch.nonzero(mask, as_tuple=True)[0]
    if len(idx) == 0:
        return torch.zeros(())
    h, q, y, w = h[idx], q[idx], y[idx], weight[idx]
    n = len(idx)
    eye = torch.eye(rd.N_TOKENS, dtype=h.dtype)
    inp = torch.cat([h[:, None, :].expand(n, rd.N_TOKENS, h.shape[1]),
                     eye[None].expand(n, rd.N_TOKENS, rd.N_TOKENS)], dim=2)
    logp = torch.log_softmax(head(inp), dim=2)                     # (n,17,k)
    nll = -logp.gather(2, y[:, None, None].expand(n, rd.N_TOKENS, 1)).squeeze(2)  # (n,17)
    per_person = (q * nll).sum(1)
    return (w * per_person).sum() / w.sum()


def pruned_law(probabilities: np.ndarray, prune=PRUNE) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.where(p >= prune, p, 0.0)
    empty = p.sum(1) <= 0
    if empty.any():
        raise FloatingPointError("pruning removed an entire row")
    return p / p.sum(1, keepdims=True)


def encoder_features(inputs, scaler, d17_token_map) -> np.ndarray:
    """[standardized policy features (49), onehot(D17 token of stored T0) (17)] = 66 legal columns.

    The stored T0 code is a legal input; giving the encoder its D17 token makes
    the reference release exactly representable (warm start) without having to
    relearn the historical codebook from X_A, r, p and risk.
    """
    from experiments.pcrl_shared_context_release_v1 import policies
    legal = policies.check_legal(inputs)
    z = scaler(rd.raw_features(legal))
    ref = np.asarray(d17_token_map, dtype=np.int64)[np.asarray(legal["token_codes"], dtype=np.int64)]
    return np.column_stack((z, rd.onehot(ref)))


N_ENCODER_FEATURES = len(rd.FEATURE_NAMES) + rd.N_TOKENS


class AdvLaw:
    """Frozen encoder + feature scaler + D17 token map -> pruned softmax law (n,17)."""

    def __init__(self, encoder, scaler, d17_token_map, prune=PRUNE):
        self.encoder = encoder
        self.scaler = scaler
        self.d17_token_map = np.asarray(d17_token_map, dtype=np.int64)
        self.prune = prune

    def __call__(self, inputs: Mapping) -> np.ndarray:
        torch = _torch()
        z = torch.as_tensor(encoder_features(inputs, self.scaler, self.d17_token_map), dtype=torch.float32)
        self.encoder.eval()
        with torch.no_grad():
            p = torch.softmax(self.encoder(z).double(), dim=1).numpy()
        return pruned_law(p, self.prune)


def _tensors(rows, scaler, h_mean, h_scale, d17_token_map):
    torch = _torch()
    f = torch.as_tensor(encoder_features(rd.legal_inputs(rows), scaler, d17_token_map), dtype=torch.float32)
    ha = torch.as_tensor((np.asarray(rows["ha"]) - h_mean[:4]) / h_scale[:4], dtype=torch.float32)
    hab = torch.as_tensor((np.column_stack((rows["ha"], rows["hb"])) - h_mean) / h_scale, dtype=torch.float32)
    w = np.asarray(rows["weights"], dtype=np.float64)
    bw = torch.as_tensor(0.5 + 0.5 * w / w.mean(), dtype=torch.float32)
    labels = {}
    for target in ("same_residence", "SEX", "RAC1P"):
        y = np.asarray(rows["labels"][target])
        m = rd.valid_mask(rows, target)
        labels[target] = (torch.as_tensor(np.where(m, y, 0), dtype=torch.long), torch.as_tensor(m))
    return {"f": f, "ha": ha, "hab": hab, "w": bw, "labels": labels,
            "codes": np.asarray(rows["token_codes"])}


def _concat_rows(a, b):
    out = {k: np.concatenate([np.asarray(a[k]), np.asarray(b[k])]) for k in a if k != "labels"}
    out["labels"] = {t: np.concatenate([np.asarray(a["labels"][t]), np.asarray(b["labels"][t])])
                     for t in a["labels"]}
    return out


def run_adv(anchor, beta, role_dict, d17, bank: rd.Round0Bank, out_dir, *, max_epochs=MAX_EPOCHS,
            min_epochs=MIN_EPOCHS, patience=PATIENCE, shortlist=SHORTLIST, init="d17", select="task",
            lr=None, weight_decay=None, log=print):
    if select not in ("task", "privacy"):
        raise ValueError("select must be 'task' (spec rule) or 'privacy' (optional P-style rule)")
    torch = _torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # M5: privacy-select units use seeds distinct from task-select units (+500)
    seed = rd.SEED_BASE["ADV"] + 1000 * anchor + int(round(beta * 10)) + (500 if select == "privacy" else 0)
    rng = np.random.default_rng(seed)
    ntr, coef, aud, sel = (role_dict[k] for k in ("nuisance_train", "coefficient_split", "audit_fit", "inner_selection"))
    scaler, scaler_source = rd.feature_builder(ntr)
    train_rows = _concat_rows(ntr, coef)
    hab0 = np.column_stack((ntr["ha"], ntr["hb"]))
    h_mean, h_scale = hab0.mean(0), np.where(hab0.std(0) > 1e-12, hab0.std(0), 1.0)
    d17_map_tokens = np.argmax(np.asarray(d17), axis=1)
    T = _tensors(train_rows, scaler, h_mean, h_scale, d17_map_tokens)
    A = _tensors(aud, scaler, h_mean, h_scale, d17_map_tokens)
    S = _tensors(sel, scaler, h_mean, h_scale, d17_map_tokens)
    encoder, decoder, adversaries = build_modules(T["f"].shape[1], seed)
    lr = LR if lr is None else float(lr)
    weight_decay = WEIGHT_DECAY if weight_decay is None else float(weight_decay)
    enc_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr,
                               weight_decay=weight_decay)
    adv_opt = torch.optim.Adam([p for m in adversaries.values() for p in m.parameters()], lr=lr,
                               weight_decay=weight_decay)
    t_start = time.perf_counter()

    def law_t(F):
        return torch.softmax(encoder(F), dim=1)

    def adv_input(D, view):
        return D["ha"] if view == "A" else D["hab"]

    def adv_loss_all(D, q, idx=None):
        losses = []
        for view, target in ADV_ROLES:
            y, m = D["labels"][target]
            h, w = adv_input(D, view), D["w"]
            if idx is not None:  # q is already the batch law
                h, y, m, w = h[idx], y[idx], m[idx], w[idx]
            losses.append(expected_ce(adversaries[f"{view}/{target}"], h, q, y, w, m))
        return losses

    # --- warm starts -------------------------------------------------------
    d17_train = rd.d17_tokens(d17, T["codes"])
    if init == "d17":
        target = torch.as_tensor(WARM_D17_MASS * rd.onehot(d17_train) + (1 - WARM_D17_MASS) / rd.N_TOKENS,
                                 dtype=torch.float32)
        opt = torch.optim.Adam(encoder.parameters(), lr=1e-2)
        for _ in range(300):
            opt.zero_grad()
            loss = -(target * torch.log_softmax(encoder(T["f"]), 1)).sum(1).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        q_T0, q_A0 = law_t(T["f"]), law_t(A["f"])
    y_task, m_task = T["labels"]["same_residence"]
    # decoder warm start, early-stopped on inner_selection (every 10 full-batch steps)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=3e-3, weight_decay=weight_decay)
    ys0, ms0 = S["labels"]["same_residence"]
    with torch.no_grad():
        q_S0 = law_t(S["f"])
    best_dec = (np.inf, copy.deepcopy(decoder.state_dict()))
    for step in range(1, DECODER_WARM_STEPS + 1):
        dec_opt.zero_grad()
        expected_ce(decoder, T["ha"], q_T0, y_task, T["w"], m_task).backward()
        dec_opt.step()
        if step % 10 == 0:
            with torch.no_grad():
                v = float(expected_ce(decoder, S["ha"], q_S0, ys0, S["w"], ms0))
            if v < best_dec[0]:
                best_dec = (v, copy.deepcopy(decoder.state_dict()))
    decoder.load_state_dict(best_dec[1])
    # adversary warm start on audit_fit, each adversary early-stopped on inner_selection
    warm_opt = torch.optim.Adam([p for m in adversaries.values() for p in m.parameters()], lr=3e-3,
                                weight_decay=weight_decay)
    names = [f"{v}/{t}" for v, t in ADV_ROLES]
    best_adv = {n: (np.inf, copy.deepcopy(adversaries[n].state_dict())) for n in names}
    for step in range(1, ADV_WARM_STEPS + 1):
        warm_opt.zero_grad()
        sum(adv_loss_all(A, q_A0)).backward()
        warm_opt.step()
        if step % 10 == 0:
            with torch.no_grad():
                vals = [float(v) for v in adv_loss_all(S, q_S0)]
            for n, v in zip(names, vals):
                if v < best_adv[n][0]:
                    best_adv[n] = (v, copy.deepcopy(adversaries[n].state_dict()))
    for n in names:
        adversaries[n].load_state_dict(best_adv[n][1])
    warm_seconds = time.perf_counter() - t_start

    # --- frozen round-0 bank on coefficient_split (screening key only) ----
    coef_inputs = rd.legal_inputs(coef)
    sel_inputs = rd.legal_inputs(sel)
    pb = rd.PersonBank(coef, d17)
    pb.add(bank.specs, bank.cut_meta, "round0")
    pb.calibrate()

    def evaluate():
        law = AdvLaw(encoder, scaler, d17_map_tokens)
        with torch.no_grad():
            sel_q = torch.as_tensor(law(sel_inputs), dtype=torch.float32)
            ys, ms = S["labels"]["same_residence"]
            # balanced U/W task CE on inner_selection with the jointly trained decoder
            ones = torch.ones_like(S["w"])
            raw_w = torch.as_tensor(np.asarray(sel["weights"], dtype=np.float64), dtype=torch.float32)
            u = float(expected_ce(decoder, S["ha"], sel_q, ys, ones, ms))
            wt = float(expected_ce(decoder, S["ha"], sel_q, ys, raw_w, ms))
            adv_sel = [float(x) for x in adv_loss_all(S, sel_q)]
        chk = pb.check(law(coef_inputs))
        support = (law(sel_inputs) > 0).sum(1)
        return {"task_U": u, "task_W": wt, "task_balanced": 0.5 * (u + wt),
                "adv_ce_inner_selection": dict(zip([f"{v}/{t}" for v, t in ADV_ROLES], adv_sel)),
                "frozen_bank_feasible": chk["feasible"], "frozen_bank_max_violation": chk["max_violation"],
                "support_mean": float(support.mean()), "support_max": int(support.max())}

    def key(rec):
        """Early-stopping / shortlist key: task CE (task units) or the M5 privacy key
        task CE - beta * mean adversary CE (privacy units), both on inner_selection;
        frozen-bank-infeasible epochs rank after all feasible ones."""
        if not rec["frozen_bank_feasible"]:
            return (1, rec["frozen_bank_max_violation"])
        if select == "privacy":
            adv_ce = rec["adv_ce_inner_selection"]
            return (0, rec["task_balanced"] - beta * sum(adv_ce.values()) / len(adv_ce))
        return (0, rec["task_balanced"])

    history, states = [], []
    ev = evaluate()
    history.append({"epoch": 0, **ev, "seconds": 0.0})
    states.append((copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict())))
    best_key, best_epoch = key(history[-1]), 0
    n_train, n_aud = len(T["f"]), len(A["f"])
    epoch_seconds = []
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        encoder.train()
        order = rng.permutation(n_train)
        for start in range(0, n_train, BATCH):
            idx = torch.as_tensor(order[start:start + BATCH])
            for _ in range(ADV_STEPS):
                aidx = torch.as_tensor(rng.choice(n_aud, size=min(BATCH, n_aud), replace=False))
                with torch.no_grad():
                    qa = law_t(A["f"][aidx])
                adv_opt.zero_grad()
                sum(adv_loss_all(A, qa, aidx)).backward()
                adv_opt.step()
            q = law_t(T["f"][idx])
            task = expected_ce(decoder, T["ha"][idx], q, y_task[idx], T["w"][idx], m_task[idx])
            adv = adv_loss_all(T, q, idx)
            loss = task - beta * sum(adv) / len(adv)
            enc_opt.zero_grad()
            adv_opt.zero_grad()
            loss.backward()
            enc_opt.step()   # adversary gradients from this loss are discarded
        epoch_seconds.append(time.perf_counter() - t0)
        ev = evaluate()
        with torch.no_grad():
            ev["train_task_balanced_rows"] = float(expected_ce(decoder, T["ha"], law_t(T["f"]), y_task, T["w"], m_task))
        history.append({"epoch": epoch, **ev, "seconds": epoch_seconds[-1]})
        states.append((copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict())))
        k = key(history[-1])
        if k < best_key:
            best_key, best_epoch = k, epoch
        log(f"[ADV b{beta} a{anchor}] epoch {epoch}: task {ev['task_balanced']:.5f} feasible0 "
            f"{ev['frozen_bank_feasible']} maxviol {ev['frozen_bank_max_violation']:.5f} "
            f"support {ev['support_mean']:.2f} ({epoch_seconds[-1]:.1f}s)")
        if epoch >= min_epochs and epoch - best_epoch >= patience:
            break
    train_seconds = time.perf_counter() - t_start

    # --- shortlist, best-response refits, final-bank selection (M1(d), M4, M5) ---
    # exact D17 witness (M4), scored with the warm-start decoder and adversaries on inner_selection
    witness_law = rd.onehot(d17_map_tokens[np.asarray(sel["token_codes"], dtype=np.int64)])
    dec0 = copy.deepcopy(decoder)
    dec0.load_state_dict(states[0][1])
    with torch.no_grad():
        wq = torch.as_tensor(witness_law, dtype=torch.float32)
        ys, ms = S["labels"]["same_residence"]
        raw_w = torch.as_tensor(np.asarray(sel["weights"], dtype=np.float64), dtype=torch.float32)
        wu = float(expected_ce(dec0, S["ha"], wq, ys, torch.ones_like(S["w"]), ms))
        ww = float(expected_ce(dec0, S["ha"], wq, ys, raw_w, ms))
    witness_task = 0.5 * (wu + ww)
    privacy_note = None
    if select == "privacy":
        # P-route task cap against the exact witness under the same warm-start decoder
        pool = [h for h in history if h["frozen_bank_feasible"] and h["task_balanced"] <= witness_task + 0.001]
        if not pool:
            privacy_note = ("NO_ELIGIBLE_EPOCH: no frozen-feasible epoch with inner task <= D17 witness task + 0.001; "
                            "selection set is the witness only (explicit, not a task-rule fallback)")
    else:
        pool = list(history)
    ranked = sorted(pool, key=lambda rec: (key(rec), rec["epoch"]))[:shortlist]
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    all_specs = dict(bank.specs)
    laws = {}
    t_br = time.perf_counter()
    for rank, rec in enumerate(ranked):
        e = rec["epoch"]
        enc = copy.deepcopy(encoder)
        enc.load_state_dict(states[e][0])
        laws[e] = AdvLaw(enc, scaler, d17_map_tokens)
        torch.save(states[e][0], ckpt_dir / f"encoder_e{e:03d}.pt")  # every shortlisted law is kept
        specs, cuts = rd.fit_best_response(
            role_dict, laws[e], out / f"best_response_e{e:03d}",
            rd.SEED_BASE["BR"] + 1000 * anchor + 500 + 10 * rank + int(round(beta * 10))
            + (500 if select == "privacy" else 0),
            f"ADV_b{beta:g}{'_P' if select == 'privacy' else ''}_e{e:03d}")
        all_specs.update(specs)
        pb.add(all_specs, cuts, f"BR_e{e:03d}")
    br_seconds = time.perf_counter() - t_br
    pb.calibrate()

    def ab_sex_slack(chk):
        """min over AB/SEX cuts and weightings of (L_a - rho) on coefficient_split (final bank)."""
        g = chk["group_max_violation"]
        return -max(g["AB/SEX|U"], g["AB/SEX|W"]) - pb.delta

    final = []
    for rec in ranked:
        chk = pb.check(laws[rec["epoch"]](coef_inputs))
        final.append({"epoch": rec["epoch"], "inner_selection_task": rec["task_balanced"],
                      "selection_key": list(key(rec)),
                      "final_bank_ab_sex_slack": ab_sex_slack(chk),
                      "frozen_bank_max_violation": rec["frozen_bank_max_violation"],
                      "final_bank": {k: v for k, v in chk.items() if k != "rho"}})
    witness_check = pb.check(pb.d17_law)
    if not witness_check["feasible"]:
        raise AssertionError("D17 witness infeasible on its own calibrated bank")
    witness = {"epoch": None, "witness": True, "inner_selection_task": witness_task,
               "final_bank_ab_sex_slack": ab_sex_slack(witness_check),
               "final_bank": {k: v for k, v in witness_check.items() if k != "rho"}}
    feasible = [f for f in final if f["final_bank"]["feasible"]] + [witness]
    if select == "privacy":  # M5: largest AB/SEX slack vs attackers refit on each checkpoint
        chosen = min(feasible, key=lambda f: (-f["final_bank_ab_sex_slack"], f["inner_selection_task"],
                                              f["epoch"] is None, f["epoch"] or 0))
    else:
        chosen = min(feasible, key=lambda f: (f["inner_selection_task"], f["epoch"] is None, f["epoch"] or 0))
    scaler_path = out / "feature_scaler.npz"
    np.savez(scaler_path, mean=scaler.mean, scale=scaler.scale, d17_token_map=d17_map_tokens)
    if chosen is witness:
        e, enc_pin, dec_pin = None, None, None
        flag = ("WITNESS_FALLBACK" if len(feasible) == 1 else "WITNESS_SELECTED_BY_RULE")
    else:
        e, flag = chosen["epoch"], None
        enc_path = ckpt_dir / f"encoder_e{e:03d}.pt"
        torch.save(states[e][0], enc_path)
        dec_path = ckpt_dir / f"decoder_e{e:03d}.pt"
        torch.save(states[e][1], dec_path)
        enc_pin = {"path": str(enc_path.relative_to(out)), "sha256": rd.sha_file(enc_path)}
        dec_pin = {"path": str(dec_path.relative_to(out)), "sha256": rd.sha_file(dec_path)}
    selected = {"schema": "pcrl-sc-adv-v1", "variant": f"ADV_beta{beta:g}", "anchor": anchor, "beta": beta,
                "selected_epoch": e, "flag": flag, "witness_selected": chosen is witness,
                "witness": witness,
                "encoder": enc_pin, "decoder": dec_pin,
                "scaler": {"path": scaler_path.name, "sha256": rd.sha_file(scaler_path)},
                "architecture": {"encoder": f"{N_ENCODER_FEATURES}-{WIDTH}-{WIDTH}-17 ReLU softmax (49 policy features + onehot D17(T0))",
                                 "decoder": f"(4+17)-{WIDTH}-{WIDTH}-2", "adversaries": f"(4|6+17)-{WIDTH}-{WIDTH}-(2|9)"},
                "training": {"lr": lr, "weight_decay": weight_decay, "batch": BATCH, "adv_steps_per_encoder_step": ADV_STEPS,
                             "max_epochs": max_epochs, "min_epochs": min_epochs, "patience": patience, "warm_d17_mass": WARM_D17_MASS,
                             "epochs_run": len(history) - 1, "init": init, "seed": seed,
                             "encoder_rows": "nuisance_train ∪ coefficient_split", "adversary_rows": "audit_fit",
                             "loss": "E_q[CE_Y] - beta * mean_k E_q[CE_S_k] (exact 17-token expectation)"},
                "prune": PRUNE, "feature_builder": scaler_source,
                "select": select, "privacy_note": privacy_note, "shortlist_pool_size": len(pool),
                "selection_rule": ("M4: selection set = shortlist (K best by early-stopping key: frozen-bank feasibility, then inner_selection task CE) ∪ {exact D17 witness}; best-response attackers refit on each shortlisted law; lowest inner_selection task CE among final-bank-feasible members (ties: checkpoint before witness, earlier epoch); witness only -> D17 exactly (WITNESS_FALLBACK)"
                                   if select == "task" else
                                   "M4 + M5 (ADV_B*_P): early stopping and shortlist by the privacy key task CE - beta*mean adversary CE on inner_selection, among frozen-feasible epochs with inner task <= D17-witness task + 0.001; best-response attackers refit on each; set = shortlist ∪ {exact D17 witness}; largest final-bank AB/SEX slack min_(AB/SEX cuts, U/W)(L_a - rho) on coefficient_split among feasible members, ties lower inner task; witness only -> D17 exactly (WITNESS_FALLBACK)"),
                "shortlist": final, "final_bank_cut_count": len(pb.cuts),
                "timing_seconds": {"warm_start": warm_seconds, "train_total": train_seconds,
                                   "per_epoch_mean": float(np.mean(epoch_seconds)) if epoch_seconds else None,
                                   "best_response_total": br_seconds,
                                   "best_response_per_checkpoint": br_seconds / max(1, len(ranked))},
                "history": history, "outer_labels_accessed": False}
    rd.write_json(out / "SELECTED.json", selected)
    return selected


def load_law(unit_dir):
    """Pinned ADV law: callable(legal inputs dict) -> pruned softmax (n,17) float64."""
    torch = _torch()
    root = Path(unit_dir)
    import json
    selected = json.loads((root / "SELECTED.json").read_text())
    items = ("scaler",) if selected.get("witness_selected") else ("encoder", "scaler")
    for item in items:
        if rd.sha_file(root / selected[item]["path"]) != selected[item]["sha256"]:
            raise ValueError(f"frozen ADV {item} hash mismatch")
    with np.load(root / selected["scaler"]["path"]) as s:
        scaler = rd.Scaler(s["mean"], s["scale"])
        d17_map_tokens = s["d17_token_map"].copy()
    if selected.get("witness_selected"):
        from . import policies

        def witness(inputs: Mapping) -> np.ndarray:  # exact one-hot D17 (M4 WITNESS_FALLBACK)
            extra = set(inputs) - set(rd.LEGAL_INPUTS)
            if extra:
                raise ValueError(f"illegal release inputs {sorted(extra)}")
            legal = policies.check_legal(inputs)
            return rd.onehot(d17_map_tokens[np.asarray(legal["token_codes"], dtype=np.int64)])
        witness.selected = selected
        return witness
    encoder, _, _ = build_modules(N_ENCODER_FEATURES, 0)
    encoder.load_state_dict(torch.load(root / selected["encoder"]["path"], weights_only=True))
    law = AdvLaw(encoder, scaler, d17_map_tokens, selected.get("prune", PRUNE))

    def call(inputs: Mapping) -> np.ndarray:
        extra = set(inputs) - set(rd.LEGAL_INPUTS)
        if extra:
            raise ValueError(f"illegal release inputs {sorted(extra)}")
        return law(inputs)
    call.selected = selected
    return call


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--anchor", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--beta", type=float, required=True, choices=BETAS)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--select", choices=("task", "privacy"), default="task",
                        help="checkpoint rule: 'task' = DESIGN_SPEC rule (default); 'privacy' = optional P-style rule")
    args = parser.parse_args(argv)
    _torch()
    t0 = time.perf_counter()
    role_dict, d17, q_hist, _ = rd.load_roles(args.anchor, smoke=args.smoke)
    bank = rd.load_bank(args.bank)
    rd.check_bank_roles(bank, role_dict)
    max_epochs = args.max_epochs or (3 if args.smoke else MAX_EPOCHS)
    result = run_adv(args.anchor, args.beta, role_dict, d17, bank, args.out,
                     max_epochs=max_epochs, min_epochs=min(MIN_EPOCHS, max_epochs),
                     shortlist=2 if args.smoke else SHORTLIST, select=args.select)
    rd.write_json(Path(args.out) / "RUN.json", {"anchor": args.anchor, "beta": args.beta, "smoke": args.smoke,
                                                "wall_seconds": time.perf_counter() - t0, "bank": bank.source,
                                                "argv": sys.argv[1:] if argv is None else list(argv)})
    print(f"ADV beta={args.beta} a{args.anchor}: epoch {result['selected_epoch']} flag={result['flag']} "
          f"in {time.perf_counter()-t0:.1f}s; per-epoch {result['timing_seconds']['per_epoch_mean']}")


if __name__ == "__main__":
    main()
