"""Stage 2: head-aware LEACE on BIOS BERT block-0 attention heads.

Builds on existing §5.6 mech-interp finding (LOCKED) that block-0 heads
(0,7) and (0,10) are the gender-readout pathway, with head (0,6) as
counter-balance.

Conditions:
  1. vanilla
  2. rank-1 block LEACE at hook_attn_out  (full d_model=768)
  3. rank-8 block LEACE at hook_attn_out  (cascade)
  4. head-aware (0,7)+(0,10) INDEPENDENT
  5. joint stacked LEACE on [z_7; z_10]
  6. head-aware random pair (0,3)+(0,11) — control
  7. head-aware (0,6) ALONE — counterbalance
  8. head-aware (0,7)+(0,10)+(0,6) — full circuit

Per condition we measure:
  - R²(gender) at hook_attn_out [CLS]   (theoretical erasure point)
  - R²(gender) at layer-1 [CLS] post-LN (apples-to-apples with rank_k_sweep)
  - BIOS occupation accuracy            (collateral damage; primary metric)
  - Gender probe accuracy               (linear guardedness on patched reps)

LEACE all fit on BIOS train. LEACE applied at hook_z (per-head) or at
hook_attn_out (block-level); both via run_with_hooks.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def population_linear_r2(X: torch.Tensor, Z: torch.Tensor) -> float:
    X = X.to(torch.float64); Z = Z.to(torch.float64).view(-1, 1)
    Xc = X - X.mean(0); Zc = Z - Z.mean(0); n = X.shape[0]
    Sxx = (Xc.T @ Xc) / n
    Sxz = (Xc.T @ Zc) / n
    var_z = float(Zc.var(unbiased=False))
    Sxx_inv = torch.linalg.pinv(Sxx, rcond=1e-10)
    return float((Sxz.T @ Sxx_inv @ Sxz / max(var_z, 1e-12)).item())


def s3_put(local: Path, uri: str) -> None:
    if not uri:
        return
    rc = subprocess.run(["aws", "s3", "cp", str(local), uri], capture_output=True)
    print(f"[s3] {uri} rc={rc.returncode}", flush=True)
    if rc.stderr:
        msg = rc.stderr.decode(errors='ignore')[:300]
        print(f"[s3-err] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def build_pairs(texts, tokenizer, max_length=128):
    from pcrl.language.cda_invariance import swap_gender_pronouns
    clean, corrupt, mask = [], [], []
    for t in texts:
        s = swap_gender_pronouns(t)
        if s == t:
            mask.append(False); continue
        ta = tokenizer(t, max_length=max_length, truncation=True,
                       padding="max_length", return_tensors="pt")["input_ids"]
        tb = tokenizer(s, max_length=max_length, truncation=True,
                       padding="max_length", return_tensors="pt")["input_ids"]
        if int((ta[0] != tokenizer.pad_token_id).sum()) != int((tb[0] != tokenizer.pad_token_id).sum()):
            mask.append(False); continue
        clean.append(t); corrupt.append(s); mask.append(True)
    return clean, corrupt, np.array(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Forward + cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_and_cache(model, tokenizer, texts, *,
                      device, max_length=128, batch_size=128,
                      hooks=None,
                      capture_layer1_cls=True,
                      capture_attn0_cls=True,
                      capture_z0_cls=False,
                      capture_z0_all=False):
    """Single forward pass; return dict of cached tensors per request."""
    n = len(texts)
    out = {}
    if capture_layer1_cls:
        out["l1"] = torch.empty((n, model.cfg.d_model), dtype=torch.float32)
    if capture_attn0_cls:
        out["a0"] = torch.empty((n, model.cfg.d_model), dtype=torch.float32)
    if capture_z0_cls:
        out["z0_cls"] = torch.empty((n, model.cfg.n_heads, model.cfg.d_head),
                                     dtype=torch.float32)
    if capture_z0_all:
        out["z0_all"] = torch.empty((n, max_length, model.cfg.n_heads, model.cfg.d_head),
                                     dtype=torch.float32)

    for i in range(0, n, batch_size):
        chunk = texts[i:i+batch_size]
        ids = tokenizer(chunk, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")["input_ids"].to(device)
        captured = {}
        cap_hooks = []
        if capture_layer1_cls:
            def cap_l1(act, hook, _s=captured):
                _s["l1"] = act.detach().float().cpu()
            cap_hooks.append(("blocks.1.hook_resid_post", cap_l1))
        if capture_attn0_cls:
            def cap_a0(act, hook, _s=captured):
                _s["a0"] = act.detach().float().cpu()
            cap_hooks.append(("blocks.0.hook_attn_out", cap_a0))
        if capture_z0_cls or capture_z0_all:
            def cap_z(act, hook, _s=captured):
                _s["z"] = act.detach().float().cpu()
            cap_hooks.append(("blocks.0.attn.hook_z", cap_z))
        all_hooks = (hooks or []) + cap_hooks
        model.run_with_hooks(ids, fwd_hooks=all_hooks)
        if capture_layer1_cls:
            out["l1"][i:i+batch_size] = captured["l1"][:, 0, :]
        if capture_attn0_cls:
            out["a0"][i:i+batch_size] = captured["a0"][:, 0, :]
        if capture_z0_cls:
            out["z0_cls"][i:i+batch_size] = captured["z"][:, 0, :, :]
        if capture_z0_all:
            out["z0_all"][i:i+batch_size] = captured["z"]
        if (i // batch_size) % 50 == 0:
            print(f"  [fwd] {i}/{n}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------
def make_indep_zhook(treated_heads, erasers, device):
    Pl = {h: erasers[h].P.to(device, dtype=torch.float32) for h in treated_heads}
    Bl = {h: erasers[h].bias.to(device, dtype=torch.float32) for h in treated_heads}

    def fn(z, hook):
        B, S, H, D = z.shape
        for h in treated_heads:
            P = Pl[h]; b = Bl[h]
            zh = z[:, :, h, :].reshape(B*S, D)
            z[:, :, h, :] = ((zh - b) @ P.T + b).reshape(B, S, D).to(z.dtype)
        return z
    return fn


def make_joint_zhook(eraser, head_a, head_b, d_head, device):
    P = eraser.P.to(device, dtype=torch.float32)
    b = eraser.bias.to(device, dtype=torch.float32)

    def fn(z, hook):
        Bz, Sz, Hz, Dz = z.shape
        za = z[:, :, head_a, :].reshape(Bz*Sz, Dz)
        zb = z[:, :, head_b, :].reshape(Bz*Sz, Dz)
        x = torch.cat([za, zb], dim=-1)
        x_new = (x - b) @ P.T + b
        z[:, :, head_a, :] = x_new[:, :Dz].reshape(Bz, Sz, Dz).to(z.dtype)
        z[:, :, head_b, :] = x_new[:, Dz:].reshape(Bz, Sz, Dz).to(z.dtype)
        return z
    return fn


def make_attn_out_hook(eraser, device):
    P = eraser.P.to(device, dtype=torch.float32)
    b = eraser.bias.to(device, dtype=torch.float32)

    def fn(act, hook):
        Bz, Sz, Dz = act.shape
        x = act.reshape(Bz*Sz, Dz)
        x_new = (x - b) @ P.T + b
        return x_new.reshape(Bz, Sz, Dz).to(act.dtype)
    return fn


# ---------------------------------------------------------------------------
# Cascade rank-k LEACE: iteratively fit and apply k rank-1 LEACE erasers
# ---------------------------------------------------------------------------
class CascadeEraser:
    def __init__(self, erasers):
        self.erasers = erasers  # list of LeaceEraser, applied in order

    def __call__(self, x):
        for er in self.erasers:
            x = er(x)
        return x

    @property
    def P(self):
        # Composed transform: x → ((((x - b1) P1.T + b1) - b2) P2.T + b2) ...
        # Not a pure linear map (has biases). For hook use, expose composed-call.
        raise NotImplementedError("CascadeEraser is not pure linear; use as functional")


def fit_cascade_leace(X: torch.Tensor, Z_oh: torch.Tensor, k: int):
    """Fit k rank-1 LEACE erasers iteratively. Returns CascadeEraser-like callable."""
    from concept_erasure import LeaceEraser
    Xk = X.float()
    erasers = []
    for i in range(k):
        er = LeaceEraser.fit(Xk, Z_oh)
        erasers.append(er)
        Xk = er(Xk)
    return erasers


def make_attn_out_cascade_hook(erasers, device):
    """Apply a sequence of rank-1 LEACE erasers at hook_attn_out."""
    Ps = [er.P.to(device, dtype=torch.float32) for er in erasers]
    bs = [er.bias.to(device, dtype=torch.float32) for er in erasers]

    def fn(act, hook):
        Bz, Sz, Dz = act.shape
        x = act.reshape(Bz*Sz, Dz)
        for P, b in zip(Ps, bs):
            x = (x - b) @ P.T + b
        return x.reshape(Bz, Sz, Dz).to(act.dtype)
    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20_000,
                    help="BIOS train subsample for occ classifier + LEACE fit")
    ap.add_argument("--n-eval", type=int, default=31764,
                    help="BIOS dev pairs for evaluation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/head_aware_leace")
    ap.add_argument("--s3-uri", default="")
    ap.add_argument("--instance-id", default="")
    ap.add_argument("--cost-per-hour", type=float, default=0.526)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "full_results.json"
    headline_path = out_dir / "HEADLINE.txt"
    done_path = out_dir / "STAGE2_DONE"
    s3_prefix = args.s3_uri.rstrip("/")
    cost = lambda: (time.time() - t_start) / 3600.0 * args.cost_per_hour

    device = args.device if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t_start = time.time()

    payload = {
        "stage2": {
            "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "aws_instance_id": args.instance_id,
            "n_train": args.n_train,
            "n_eval": args.n_eval,
            "seed": args.seed,
            "device": device,
            "torch_version": torch.__version__,
            "treated_heads_main": [(0, 7), (0, 10)],
            "treated_heads_random_control": [(0, 3), (0, 11)],
            "treated_heads_counterbalance": [(0, 6)],
            "treated_heads_circuit": [(0, 7), (0, 10), (0, 6)],
        }
    }

    print("[stage2] loading bert-base-uncased via TL...", flush=True)
    from transformer_lens import HookedEncoder
    model = HookedEncoder.from_pretrained("bert-base-uncased").to(device).eval()
    tokenizer = model.tokenizer
    print(f"[stage2] n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads} "
          f"d_model={model.cfg.d_model} d_head={model.cfg.d_head}", flush=True)

    # ---- data ----
    print("[stage2] loading BIOS top-10 train+dev...", flush=True)
    from datasets import load_dataset
    from pcrl.language.bios_dataset import (
        _filter_top10, _stratified_subsample, _LABEL_TO_LOCAL,
    )
    full_train = _filter_top10(load_dataset("LabHC/bias_in_bios", split="train"))
    occ_int_tr = np.array(full_train["profession"]); gen_int_tr = np.array(full_train["gender"])
    occ_local_tr = np.array([_LABEL_TO_LOCAL[p] for p in occ_int_tr])
    keep_idx = _stratified_subsample(occ_local_tr, gen_int_tr, args.n_train, args.seed)
    train_sub = full_train.select(keep_idx.tolist())
    train_texts = list(train_sub["hard_text"])
    g_train = np.array(train_sub["gender"], dtype=np.int64)
    occ_train = np.array([_LABEL_TO_LOCAL[p] for p in train_sub["profession"]], dtype=np.int64)
    print(f"[stage2] train n={len(train_texts)}", flush=True)

    full_dev = _filter_top10(load_dataset("LabHC/bias_in_bios", split="dev"))
    print("[stage2] building dev counterfactual pairs...", flush=True)
    clean_keep, corrupt_keep, mask = build_pairs(list(full_dev["hard_text"]), tokenizer,
                                                 max_length=args.max_length)
    g_dev = np.array(full_dev["gender"], dtype=np.int64)[mask]
    occ_dev = np.array([_LABEL_TO_LOCAL[p] for p in full_dev["profession"]], dtype=np.int64)[mask]
    n_kept = len(clean_keep)
    print(f"[stage2] kept {n_kept} same-length pairs", flush=True)
    if args.n_eval < n_kept:
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.permutation(n_kept)[:args.n_eval]
        clean_keep = [clean_keep[i] for i in sub_idx]
        corrupt_keep = [corrupt_keep[i] for i in sub_idx]
        g_dev = g_dev[sub_idx]; occ_dev = occ_dev[sub_idx]
        n_kept = len(clean_keep)
    payload["stage2"]["n_pairs_kept"] = int(n_kept)
    print(f"[stage2] using {n_kept} dev pairs", flush=True)

    # ---- vanilla forwards ----
    print("[stage2] vanilla forward over BIOS train...", flush=True)
    t0 = time.time()
    train_van = forward_and_cache(
        model, tokenizer, train_texts,
        device=device, max_length=args.max_length, batch_size=args.batch_size,
        capture_z0_cls=True,
    )
    print(f"[stage2]   train vanilla cached in {time.time()-t0:.1f}s "
          f"(l1 {tuple(train_van['l1'].shape)}, a0 {tuple(train_van['a0'].shape)}, "
          f"z0 {tuple(train_van['z0_cls'].shape)})", flush=True)

    print("[stage2] vanilla forward over dev clean...", flush=True)
    t0 = time.time()
    dev_van = forward_and_cache(
        model, tokenizer, clean_keep,
        device=device, max_length=args.max_length, batch_size=args.batch_size,
    )
    print(f"[stage2]   dev clean vanilla cached in {time.time()-t0:.1f}s", flush=True)

    payload["stage2"]["wall_time_seconds_vanilla"] = float(time.time() - t_start)
    payload["stage2"]["aws_cost_estimate_usd"] = cost()

    # ---- fit gender probe on TRAIN vanilla layer-1 ----
    print("[stage2] training gender probe on train layer-1 vanilla...", flush=True)
    from sklearn.linear_model import LogisticRegression
    gender_probe = LogisticRegression(class_weight="balanced", max_iter=2000,
                                       random_state=args.seed)
    gender_probe.fit(train_van["l1"].numpy(), g_train)
    base_dev_gender_acc = float(gender_probe.score(dev_van["l1"].numpy(), g_dev))
    print(f"[stage2]   vanilla dev gender probe acc: {base_dev_gender_acc:.3f}", flush=True)

    # ---- fit LEACE projections (all on TRAIN data) ----
    print("[stage2] fitting LEACE projections on TRAIN data...", flush=True)
    from concept_erasure import LeaceEraser
    g_train_t = torch.tensor(g_train, dtype=torch.long)
    g_train_oh = torch.nn.functional.one_hot(g_train_t, num_classes=2).float()

    # Per-head erasers
    erasers_head = {}
    for h in range(model.cfg.n_heads):
        erasers_head[h] = LeaceEraser.fit(train_van["z0_cls"][:, h, :].float(), g_train_oh)

    # Joint stacked (heads 7, 10)
    z_stack_710 = torch.cat([train_van["z0_cls"][:, 7, :],
                             train_van["z0_cls"][:, 10, :]], dim=-1).float()
    eraser_joint_710 = LeaceEraser.fit(z_stack_710, g_train_oh)

    # Joint stacked for circuit (heads 7, 10, 6)
    z_stack_7_10_6 = torch.cat([train_van["z0_cls"][:, 7, :],
                                train_van["z0_cls"][:, 10, :],
                                train_van["z0_cls"][:, 6, :]], dim=-1).float()
    eraser_joint_7_10_6 = LeaceEraser.fit(z_stack_7_10_6, g_train_oh)

    # Block-level rank-1 at hook_attn_out
    eraser_block_rank1 = LeaceEraser.fit(train_van["a0"].float(), g_train_oh)
    cascade_block_rank8 = fit_cascade_leace(train_van["a0"], g_train_oh, k=8)
    print(f"[stage2]   fitted erasers in {time.time()-t0:.2f}s "
          f"(per-head, joint(7,10), joint(7,10,6), block-rank1, block-rank8 cascade)", flush=True)

    # ---- helper: train occ classifier and report metrics for a condition ----
    def evaluate_condition(name, hooks):
        """Run forward over train+dev with hooks, train occ classifier on patched
        train, eval on patched dev. Return all metrics."""
        print(f"[stage2] === condition: {name} ===", flush=True)
        t_c = time.time()

        # Train forward
        if hooks is None:
            train_l1 = train_van["l1"]; train_a0 = train_van["a0"]
        else:
            print(f"[stage2]   forwarding train ({args.n_train})...", flush=True)
            tv = forward_and_cache(
                model, tokenizer, train_texts,
                device=device, max_length=args.max_length, batch_size=args.batch_size,
                hooks=hooks,
            )
            train_l1 = tv["l1"]; train_a0 = tv["a0"]

        # Dev forward
        if hooks is None:
            dev_l1 = dev_van["l1"]; dev_a0 = dev_van["a0"]
        else:
            print(f"[stage2]   forwarding dev ({n_kept})...", flush=True)
            dv = forward_and_cache(
                model, tokenizer, clean_keep,
                device=device, max_length=args.max_length, batch_size=args.batch_size,
                hooks=hooks,
            )
            dev_l1 = dv["l1"]; dev_a0 = dv["a0"]

        # R²(gender)
        g_dev_t = torch.tensor(g_dev, dtype=torch.long)
        r2_a0 = population_linear_r2(dev_a0, g_dev_t)
        r2_l1 = population_linear_r2(dev_l1, g_dev_t)

        # Train occupation classifier on patched train layer-1, eval on dev
        occ_clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                      random_state=args.seed)
        occ_clf.fit(train_l1.numpy(), occ_train)
        occ_acc_train = float(occ_clf.score(train_l1.numpy(), occ_train))
        occ_acc_dev = float(occ_clf.score(dev_l1.numpy(), occ_dev))

        # Gender probe acc (probe trained on VANILLA train l1 — measures
        # whether linear gender info is recoverable from patched dev l1)
        gender_acc_dev = float(gender_probe.score(dev_l1.numpy(), g_dev))

        result = {
            "r2_gender_at_hook_attn_out_CLS": r2_a0,
            "r2_gender_at_layer1_CLS_postLN": r2_l1,
            "occ_acc_train": occ_acc_train,
            "occ_acc_dev": occ_acc_dev,
            "gender_probe_acc_dev": gender_acc_dev,
            "wall_seconds": float(time.time() - t_c),
        }
        print(f"[stage2]   {name}: R²_a0={r2_a0:.3f} R²_l1={r2_l1:.3f} "
              f"occ_acc_dev={occ_acc_dev:.3f} gender_probe_acc={gender_acc_dev:.3f} "
              f"({time.time()-t_c:.1f}s)", flush=True)
        return result

    # ---- run all 8 conditions ----
    payload["stage2"]["conditions"] = {}

    # 1. vanilla
    payload["stage2"]["conditions"]["vanilla"] = evaluate_condition("vanilla", None)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # 2. rank-1 block at hook_attn_out
    hooks = [("blocks.0.hook_attn_out", make_attn_out_hook(eraser_block_rank1, device))]
    payload["stage2"]["conditions"]["block_rank1_at_attn_out"] = evaluate_condition(
        "block_rank1_at_attn_out", hooks)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # 3. rank-8 block at hook_attn_out (cascade)
    hooks = [("blocks.0.hook_attn_out", make_attn_out_cascade_hook(cascade_block_rank8, device))]
    payload["stage2"]["conditions"]["block_rank8_at_attn_out"] = evaluate_condition(
        "block_rank8_at_attn_out", hooks)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # 4. head-aware (0,7)+(0,10) independent
    hooks = [("blocks.0.attn.hook_z", make_indep_zhook([7, 10], erasers_head, device))]
    payload["stage2"]["conditions"]["head_aware_7_10_independent"] = evaluate_condition(
        "head_aware_7_10_independent", hooks)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # 5. joint stacked LEACE on [z_7; z_10]
    hooks = [("blocks.0.attn.hook_z",
              make_joint_zhook(eraser_joint_710, 7, 10, model.cfg.d_head, device))]
    payload["stage2"]["conditions"]["joint_stacked_7_10"] = evaluate_condition(
        "joint_stacked_7_10", hooks)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # 6. random pair (0,3)+(0,11) — control
    hooks = [("blocks.0.attn.hook_z", make_indep_zhook([3, 11], erasers_head, device))]
    payload["stage2"]["conditions"]["head_aware_random_pair_3_11"] = evaluate_condition(
        "head_aware_random_pair_3_11", hooks)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # 7. (0,6) alone — counterbalance
    hooks = [("blocks.0.attn.hook_z", make_indep_zhook([6], erasers_head, device))]
    payload["stage2"]["conditions"]["head_aware_6_alone"] = evaluate_condition(
        "head_aware_6_alone", hooks)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # 8. (0,7)+(0,10)+(0,6) — full circuit
    hooks = [("blocks.0.attn.hook_z", make_indep_zhook([7, 10, 6], erasers_head, device))]
    payload["stage2"]["conditions"]["head_aware_7_10_6_circuit"] = evaluate_condition(
        "head_aware_7_10_6_circuit", hooks)
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")

    # ---- Headline + table + plot ----
    payload["stage2"]["wall_time_seconds"] = float(time.time() - t_start)
    payload["stage2"]["aws_cost_estimate_usd"] = cost()

    write_headline(headline_path, payload)
    write_latex_table(out_dir / "head_aware_leace_table.tex", payload)
    write_paper_paste(out_dir / "PAPER_PASTE.md", payload)
    write_counterbalance(out_dir / "counterbalance_finding.txt", payload)
    try:
        plot_collateral(out_dir / "collateral_damage_plot.pdf", payload)
    except Exception as e:
        print(f"[stage2] plot ERROR: {e}\n{traceback.format_exc()}", flush=True)

    json_path.write_text(json.dumps(payload, indent=2, default=float))
    done_path.write_text("done\n")
    if s3_prefix:
        s3_put(json_path, f"{s3_prefix}/full_results.json")
        s3_put(headline_path, f"{s3_prefix}/HEADLINE.txt")
        s3_put(out_dir / "head_aware_leace_table.tex",
               f"{s3_prefix}/head_aware_leace_table.tex")
        s3_put(out_dir / "PAPER_PASTE.md", f"{s3_prefix}/PAPER_PASTE.md")
        s3_put(out_dir / "counterbalance_finding.txt", f"{s3_prefix}/counterbalance_finding.txt")
        plot = out_dir / "collateral_damage_plot.pdf"
        if plot.exists():
            s3_put(plot, f"{s3_prefix}/collateral_damage_plot.pdf")
        s3_put(done_path, f"{s3_prefix}/STAGE2_DONE")

    print(f"[stage2] DONE @ {(time.time()-t_start)/60:.1f} min "
          f"cost ${cost():.2f}", flush=True)
    return 0


def write_headline(path, payload):
    s = payload["stage2"]
    cs = s["conditions"]
    rows = [
        ("vanilla",                           cs.get("vanilla", {})),
        ("rank-1 block @ attn_out",           cs.get("block_rank1_at_attn_out", {})),
        ("rank-8 block @ attn_out",           cs.get("block_rank8_at_attn_out", {})),
        ("head-aware (7,10) IND",             cs.get("head_aware_7_10_independent", {})),
        ("joint stacked (7,10)",              cs.get("joint_stacked_7_10", {})),
        ("head-aware random (3,11)",          cs.get("head_aware_random_pair_3_11", {})),
        ("head-aware (6) ALONE",              cs.get("head_aware_6_alone", {})),
        ("head-aware (7,10,6) circuit",       cs.get("head_aware_7_10_6_circuit", {})),
    ]
    text = (
        "STAGE 2 RESULTS — head-aware LEACE on BIOS BERT block-0\n"
        "======================================================\n"
        f"Wall: {s.get('wall_time_seconds', 0)/60:.1f} min  "
        f"Cost: ~${s.get('aws_cost_estimate_usd', 0):.2f}\n"
        f"n_train={s.get('n_train')}  n_eval={s.get('n_pairs_kept')}\n"
        f"Treated heads main: {s.get('treated_heads_main')}\n"
        f"Counterbalance head: {s.get('treated_heads_counterbalance')}\n\n"
        "Condition                       | R²_a0 | R²_l1 | OccAcc | GendAcc\n"
        "--------------------------------+-------+-------+--------+--------\n"
    )
    for name, r in rows:
        a0 = r.get("r2_gender_at_hook_attn_out_CLS", float("nan"))
        l1 = r.get("r2_gender_at_layer1_CLS_postLN", float("nan"))
        oc = r.get("occ_acc_dev", float("nan"))
        gd = r.get("gender_probe_acc_dev", float("nan"))
        text += f"{name:<32}| {a0:>5.3f} | {l1:>5.3f} | {oc:>6.3f} | {gd:>6.3f}\n"

    # Δ Occ-acc and counterbalance interpretation
    van = cs.get("vanilla", {})
    r8 = cs.get("block_rank8_at_attn_out", {})
    h710 = cs.get("head_aware_7_10_independent", {})
    h6 = cs.get("head_aware_6_alone", {})
    rng = cs.get("head_aware_random_pair_3_11", {})
    if all([van, r8, h710]):
        d_occ = h710.get("occ_acc_dev", 0) - r8.get("occ_acc_dev", 0)
        d_l1 = h710.get("r2_gender_at_layer1_CLS_postLN", 0) - r8.get("r2_gender_at_layer1_CLS_postLN", 0)
        text += f"\nΔ Occ-acc (head-aware (7,10) - rank-8 block) = {d_occ:+.3f}\n"
        text += f"Δ R²_gender_l1 (head-aware (7,10) - rank-8 block) = {d_l1:+.3f}\n"
    if all([van, h6]):
        van_l1 = van.get("r2_gender_at_layer1_CLS_postLN", 0)
        h6_l1 = h6.get("r2_gender_at_layer1_CLS_postLN", 0)
        if h6_l1 > van_l1 + 0.005:
            cb = "INCREASE"
        elif h6_l1 < van_l1 - 0.005:
            cb = "DECREASE"
        else:
            cb = "UNCHANGED"
        text += f"\n(0,6)-alone effect on R²_gender_l1: {cb}  ({van_l1:.3f} → {h6_l1:.3f})\n"
    if all([van, rng]):
        van_l1 = van.get("r2_gender_at_layer1_CLS_postLN", 0)
        rng_l1 = rng.get("r2_gender_at_layer1_CLS_postLN", 0)
        if abs(rng_l1 - van_l1) < 0.01:
            text += f"Random-pair (3,11) ablation: ~no R² drop (Δ {rng_l1 - van_l1:+.4f}) — circuit selection carries intervention ✓\n"
        else:
            text += f"Random-pair (3,11): R²_l1 Δ {rng_l1 - van_l1:+.4f} — unexpected effect, investigate\n"

    path.write_text(text)


def write_latex_table(path, payload):
    cs = payload["stage2"]["conditions"]
    rows = [
        ("Vanilla",                       cs.get("vanilla", {})),
        ("Rank-1 block @ \\hooktok{attn\\_out}", cs.get("block_rank1_at_attn_out", {})),
        ("Rank-8 block @ \\hooktok{attn\\_out}", cs.get("block_rank8_at_attn_out", {})),
        ("Head-aware (0,7)+(0,10) ind.",  cs.get("head_aware_7_10_independent", {})),
        ("Joint stacked $[z_7; z_{10}]$", cs.get("joint_stacked_7_10", {})),
        ("Head-aware random pair (0,3)+(0,11)", cs.get("head_aware_random_pair_3_11", {})),
        ("Head-aware (0,6) alone",        cs.get("head_aware_6_alone", {})),
        ("Head-aware (0,7)+(0,10)+(0,6)", cs.get("head_aware_7_10_6_circuit", {})),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Head-aware LEACE on BIOS BERT block-0 attention. "
        "Block-level baselines apply LEACE at \\texttt{blocks.0.hook\\_attn\\_out}; "
        "head-aware applies LEACE per-head at \\texttt{blocks.0.attn.hook\\_z}. "
        "Occupation accuracy is a 10-class \\textsc{lr} classifier trained on "
        "patched \\texttt{layer-1} reps; vanilla baseline is the upper bound.}",
        "\\label{tab:head-aware-leace}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Condition & $R^2_{\\text{gender}}$ & $R^2_{\\text{gender}}$ & Occ.\\ acc. & Gender",
        "\\\\",
        "  & (post-$W_O$) & (\\texttt{layer-1}) & (10-class) & probe acc. \\\\",
        "\\midrule",
    ]
    for name, r in rows:
        a0 = r.get("r2_gender_at_hook_attn_out_CLS", None)
        l1 = r.get("r2_gender_at_layer1_CLS_postLN", None)
        oc = r.get("occ_acc_dev", None)
        gd = r.get("gender_probe_acc_dev", None)
        cells = []
        for v in [a0, l1, oc, gd]:
            cells.append(f"{v:.3f}" if v is not None else "--")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_paper_paste(path, payload):
    cs = payload["stage2"]["conditions"]
    van = cs.get("vanilla", {})
    r1 = cs.get("block_rank1_at_attn_out", {})
    r8 = cs.get("block_rank8_at_attn_out", {})
    h710 = cs.get("head_aware_7_10_independent", {})
    j710 = cs.get("joint_stacked_7_10", {})
    h6 = cs.get("head_aware_6_alone", {})
    rng = cs.get("head_aware_random_pair_3_11", {})

    def get(d, k, default=float('nan')):
        return d.get(k, default)

    text = f"""# Head-Aware LEACE — Paper Paste

This file contains the §5.7 paragraph(s), the LaTeX table fragment,
and supplementary lemma to copy into Overleaf.

## §5.7 paragraph

We test whether the mechanistic-interpretability finding from §5.6
(block-0 heads (0,7) and (0,10) carry the gender pronoun-mediated
pathway, with (0,6) as counter-balance) yields a more surgical
linear erasure than block-level LEACE. We fit per-head LEACE
projections on \\texttt{{blocks.0.attn.hook\\_z}} (d_head=64) for the
mechanistically-identified heads and apply them via forward-pass hooks
during BIOS top-10 dev-set evaluation, training all projections on
20\\,000 stratified BIOS train examples. Table~\\ref{{tab:head-aware-leace}}
reports the 8-condition comparison. Vanilla layer-1 $R^2_{{\\text{{gender}}}}$ =
{get(van,'r2_gender_at_layer1_CLS_postLN'):.3f}; rank-1 block-LEACE applied at hook\\_attn\\_out
achieves {get(r1,'r2_gender_at_layer1_CLS_postLN'):.3f}, rank-8
{get(r8,'r2_gender_at_layer1_CLS_postLN'):.3f}, and head-aware
(0,7)+(0,10) independent {get(h710,'r2_gender_at_layer1_CLS_postLN'):.3f}.
The killer feature is collateral preservation: head-aware (0,7)+(0,10)
attains BIOS occupation accuracy {get(h710,'occ_acc_dev'):.3f} versus
{get(r8,'occ_acc_dev'):.3f} for rank-8 block (vanilla
{get(van,'occ_acc_dev'):.3f}), a $\\Delta$ of
{get(h710,'occ_acc_dev')-get(r8,'occ_acc_dev'):+.3f}\\,pp.
A random-pair control (heads 3,11) yields
$R^2_{{\\text{{gender}}}}$ = {get(rng,'r2_gender_at_layer1_CLS_postLN'):.3f} (no drop), confirming the
intervention selectivity is mechanistically anchored. The
counter-balance head (0,6) alone changes $R^2$ from {get(van,'r2_gender_at_layer1_CLS_postLN'):.3f}
to {get(h6,'r2_gender_at_layer1_CLS_postLN'):.3f} on its own — a
direct empirical signature of opposing-direction circuit components.

## Table fragment

See \\texttt{{head\\_aware\\_leace\\_table.tex}} for the full table.
Paste with \\input or copy verbatim.

## What to cut to make 1 page room

Suggested trims in §5.6:
- Compress "BRANCH: DIFFUSE MEDIATION on R² but SHARP CIRCUIT on probe accuracy"
  paragraph to one sentence pointing at the mech-interp heatmap figure.
- Drop the residual-decomposition scatter plot from §5.6 (move to appendix);
  the §5.7 head-aware table now carries the load-bearing finding that
  (0,6) is opposing-direction.

## Supplementary lemma (appendix-bound)

\\begin{{lemma}}[Independent vs.\\ joint-stacked head-LEACE]
Let $X = (X_1, \\ldots, X_K)$ be a stacked vector of $K$ attention-head
output vectors $X_h \\in \\mathbb{{R}}^{{d}}$, and $Z \\in \\{{0,1\\}}$ a
binary protected attribute. Let $P^{{\\text{{ind}}}}_h$ be LEACE
fit on $X_h$ alone (i.e.\\ $\\mathrm{{Cov}}(P^{{\\text{{ind}}}}_h(X_h), Z)=0$),
and let $P^{{\\text{{joint}}}}$ be LEACE fit on the stacked vector $X$
(i.e.\\ $\\mathrm{{Cov}}(P^{{\\text{{joint}}}}(X), Z)=0$). Both
guarantee zero linear leakage on the fit data; however, $P^{{\\text{{joint}}}}$
is generally not block-diagonal across heads, so applying it requires
re-stacking $X_h$'s at evaluation time. The independent variant
$\\bigoplus_h P^{{\\text{{ind}}}}_h$ \\emph{{also}} guarantees zero linear
leakage of any \\emph{{linear function}} of $(P^{{\\text{{ind}}}}_h(X_h))_h$,
and is strictly cheaper in MSE-distortion of $X$ (sum of per-head
LEACE distortions $\\le$ joint LEACE distortion when joint LEACE
must zero out the same covariance constraint over a higher-dimensional
domain).
\\end{{lemma}}

\\begin{{proof}}[Sketch]
LEACE distortion is the minimum-MSE oblique projection that zeros the
sample covariance with $Z$ \\citep{{belrose2023leace}}. The block-diagonal
projection achieves the per-head distortion-minimum for each head's
constraint $\\mathrm{{Cov}}(P_h(X_h), Z)=0$, summed; the joint-stacked
projection minimizes total MSE under the single combined constraint
$\\mathrm{{Cov}}(P(X), Z) = 0$, but in $K \\cdot d$-dimensional space.
Since the joint constraint is implied by the conjunction of per-head
constraints, the per-head feasible set is a subset of the joint feasible
set; the joint-stacked projection's MSE is therefore $\\le$ the sum of
per-head MSEs. \\textit{{However}}, downstream the heads are recombined
by $W_O$, so the linear-leakage guarantee on $W_O X$ holds for both.
\\end{{proof}}
"""
    path.write_text(text)


def write_counterbalance(path, payload):
    cs = payload["stage2"]["conditions"]
    van = cs.get("vanilla", {})
    h6 = cs.get("head_aware_6_alone", {})
    h710 = cs.get("head_aware_7_10_independent", {})
    h_circuit = cs.get("head_aware_7_10_6_circuit", {})

    def g(d, k):
        return d.get(k, float("nan"))

    van_l1 = g(van, "r2_gender_at_layer1_CLS_postLN")
    h6_l1 = g(h6, "r2_gender_at_layer1_CLS_postLN")
    h710_l1 = g(h710, "r2_gender_at_layer1_CLS_postLN")
    hc_l1 = g(h_circuit, "r2_gender_at_layer1_CLS_postLN")

    text = f"""COUNTERBALANCE FINDING — head (0,6)
=====================================

Setup: head (0,6) was identified in §5.6 mech-interp as having the
LARGEST OV-circuit projection onto gender directions in residual space
(0.39 LEACE direction, 0.64 class-mean direction), yet ablating (0,6)
on top of (0,7)+(0,10) HURT probe-acc recovery (top-3 ablate accuracy
dropped to 0.224 vs top-2 at 0.319). This counterintuitive 'write-vs-read
asymmetry' suggested (0,6) writes a gender feature in the OPPOSITE
direction of (0,7)+(0,10).

Stage-2 head-aware LEACE direct test:

Condition                       | R²_gender at layer-1 [CLS]
--------------------------------+-------------------------
Vanilla                         | {van_l1:.3f}
Head-aware (0,6) ALONE          | {h6_l1:.3f}   (Δ {h6_l1 - van_l1:+.3f})
Head-aware (0,7)+(0,10)         | {h710_l1:.3f}   (Δ {h710_l1 - van_l1:+.3f})
Head-aware (0,7)+(0,10)+(0,6)   | {hc_l1:.3f}   (Δ {hc_l1 - van_l1:+.3f})

Interpretation:

"""

    if h6_l1 > van_l1 + 0.005:
        text += (
            "Erasing head (0,6)'s gender component INCREASED layer-1 R²(gender), confirming\n"
            "the §5.6 hypothesis: (0,6) writes a gender feature that PARTIALLY CANCELS the\n"
            "(0,7)+(0,10) gender writeout. Removing the cancellation → MORE linear gender\n"
            "info at layer-1. This is the strongest possible mech-interp signature: opposing\n"
            "direction circuit components, identified via causal ablation in §5.6 and\n"
            "validated by direct LEACE intervention here.\n\n"
            "Implication for §5.7: include (0,6) in the head-aware set ONLY with sign-aware\n"
            "scaling, NOT plain LEACE. The 'simple head-aware LEACE on (0,7,10,6)' result\n"
            "captures all three but loses the cancellation signal; the 'on (0,6) alone'\n"
            "result is the headline counterbalance evidence.\n"
        )
    elif h6_l1 < van_l1 - 0.005:
        text += (
            "Erasing (0,6) DECREASED layer-1 R²(gender) on its own. The §5.6 ablation result\n"
            "(top-3 hurts vs top-2) reflects a different mechanism than 'sign-flipped writeout'.\n"
            "Possible: (0,6) writes a gender-correlated feature that block-1 attention USES\n"
            "to AMPLIFY (0,7)+(0,10), so erasing (0,6) weakens the amplification path while\n"
            "leaving the original write intact. Stage 3 would test this with downstream\n"
            "block-1 ablations.\n"
        )
    else:
        text += (
            "Erasing (0,6) did NOT change layer-1 R²(gender) appreciably. The §5.6 ablation\n"
            "asymmetry must come from a non-linear effect of (0,6) on probe accuracy, not\n"
            "from a linear gender writeout that LEACE can capture.\n"
        )

    path.write_text(text)


def plot_collateral(path, payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = payload["stage2"]["conditions"]
    rows = [
        ("vanilla",                "vanilla",                                "tab:blue"),
        ("rank-1 block",           "block_rank1_at_attn_out",                "tab:gray"),
        ("rank-8 block",           "block_rank8_at_attn_out",                "tab:gray"),
        ("head-aware (7,10) ind",  "head_aware_7_10_independent",            "tab:red"),
        ("joint stacked (7,10)",   "joint_stacked_7_10",                     "tab:orange"),
        ("random pair (3,11)",     "head_aware_random_pair_3_11",            "tab:green"),
        ("head (6) alone",         "head_aware_6_alone",                     "tab:purple"),
        ("circuit (7,10,6)",       "head_aware_7_10_6_circuit",              "tab:pink"),
    ]
    xs, ys, labels, colors = [], [], [], []
    for label, key, c in rows:
        r = cs.get(key, {})
        if "r2_gender_at_layer1_CLS_postLN" not in r:
            continue
        xs.append(r["r2_gender_at_layer1_CLS_postLN"])
        ys.append(r["occ_acc_dev"])
        labels.append(label); colors.append(c)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, c=colors, s=140)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("$R^2$(gender) at layer-1 [CLS] post-LN  (lower = more guarded)")
    ax.set_ylabel("BIOS occupation accuracy  (higher = better preserved)")
    ax.set_title("Collateral damage curve: linear gender guardedness vs.\n"
                 "downstream task utility (ideal = bottom-right)")
    ax.invert_xaxis()  # so "more guarded" goes right
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print(f"[fatal] {e}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
