"""experiments/run_rank_mechanism.py

Two experiments for the §5.5 rank-mechanism upgrade. Both share a single
forward pass over a BIOS dev sample on frozen bert-base-uncased.

E1: Per-token-type cosine analysis.
    Classify dev sentences into 4 disjoint token-type subsets (only-P,
    only-H, only-N, only-FN). For each subset, compute the layer-0
    [CLS] female-male centroid difference v_X. Report pairwise
    cosines |v_X · v_Y| / ||v_X|| ||v_Y||. Repeat at layer 6.
    Verdict: low mean off-diagonal cosine at layer 0 confirms the
    multi-rank gender encoding hypothesis; layer 6 is expected to
    consolidate.

E2: Rank-k LEACE sweep at layer 0.
    Stack the 4 token-type centroid differences with 10 occupation-
    conditional centroid differences (gives up to 14 candidate
    directions). SVD to get an orthonormal concept basis U. For
    k ∈ {1, 2, 4, 8}, build the rank-k erasure projection
    P_k = I - U_{:k} U_{:k}^T applied to ALL token positions at
    layer 0, then re-forward and measure R²([CLS]_layer1, gender).

NO BACKBONE TRAINING. Pure forward + linear algebra.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Token-type detectors (regexes are compiled once)
# ---------------------------------------------------------------------------

PRONOUNS = ["he", "she", "his", "her", "him", "hers", "himself", "herself"]
HONORIFICS = ["mr", "mrs", "ms", "dr", "miss", "madam", "sir"]
GENDERED_NOUNS = [
    "actress", "actor", "waitress", "waiter", "waitstaff",
    "businesswoman", "businessman", "policewoman", "policeman",
    "saleswoman", "salesman", "stewardess", "spokeswoman", "spokesman",
    "saleswomen", "salesmen", "actresses", "waitresses",
    "chairwoman", "chairman", "congresswoman", "congressman",
    "fireman", "firewoman", "fisherman", "freshman", "freshmen",
    "gentleman", "gentlemen", "lady", "ladies",
]
FEMALE_NAMES = [
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
    "susan", "jessica", "sarah", "karen", "lisa", "nancy", "betty",
    "helen", "sandra", "donna", "carol", "ruth", "sharon", "michelle",
    "laura", "kimberly", "deborah", "dorothy", "amy", "angela",
    "ashley", "brenda", "emma", "olivia", "cynthia", "marie", "janet",
    "catherine", "frances", "christine", "samantha", "debra", "rachel",
    "carolyn", "virginia", "maria", "heather", "diane", "julie",
    "joyce", "joan", "ann", "shirley", "victoria",
]
MALE_NAMES = [
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
    "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
    "kenneth", "kevin", "brian", "george", "edward", "ronald", "timothy",
    "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
    "jonathan", "stephen", "larry", "justin", "scott", "brandon",
    "benjamin", "samuel", "gregory", "frank", "alexander", "raymond",
    "patrick", "jack", "dennis", "jerry",
]
ALL_NAMES = FEMALE_NAMES + MALE_NAMES

re_P = re.compile(r"\b(" + "|".join(PRONOUNS) + r")\b", re.IGNORECASE)
re_H = re.compile(r"\b(" + "|".join(HONORIFICS) + r")\.?\b", re.IGNORECASE)
re_N = re.compile(r"\b(" + "|".join(GENDERED_NOUNS) + r")\b", re.IGNORECASE)
re_FN = re.compile(r"\b(" + "|".join(ALL_NAMES) + r")\b", re.IGNORECASE)


def classify_sentence(text: str) -> set[str]:
    types: set[str] = set()
    if re_P.search(text):
        types.add("P")
    if re_H.search(text):
        types.add("H")
    if re_N.search(text):
        types.add("N")
    if re_FN.search(text):
        types.add("FN")
    return types


# ---------------------------------------------------------------------------
# Linear-probe R² (population formula, pinv with rcond=1e-10)
# ---------------------------------------------------------------------------

def population_linear_r2(X: torch.Tensor, Z: torch.Tensor) -> float:
    X = X.to(torch.float64); Z = Z.to(torch.float64).view(-1, 1)
    Xc = X - X.mean(0); Zc = Z - Z.mean(0)
    n = X.shape[0]
    Sxx = (Xc.T @ Xc) / n
    Sxz = (Xc.T @ Zc) / n
    var_z = Zc.var(unbiased=False)
    Sxx_inv = torch.linalg.pinv(Sxx, rcond=1e-10)
    return float((Sxz.T @ Sxx_inv @ Sxz / max(var_z.item(), 1e-12)).item())


# ---------------------------------------------------------------------------
# Per-token-type centroid differences + cosines
# ---------------------------------------------------------------------------

def centroid_diff(H: torch.Tensor, G: torch.Tensor, idx: list[int],
                  *, min_each: int = 5):
    if len(idx) < 2 * min_each:
        return None
    H_sub = H[idx]
    G_sub = G[idx]
    f = (G_sub == 1)
    m = (G_sub == 0)
    if int(f.sum()) < min_each or int(m.sum()) < min_each:
        return None
    return H_sub[f].mean(0) - H_sub[m].mean(0)


def cosine_matrix(vecs: list) -> np.ndarray:
    K = len(vecs)
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if vecs[i] is None or vecs[j] is None:
                M[i, j] = float("nan")
                continue
            ni = vecs[i] / max(float(vecs[i].norm()), 1e-12)
            nj = vecs[j] / max(float(vecs[j].norm()), 1e-12)
            M[i, j] = float(abs(float(ni @ nj)))
    return M


def mean_off_diagonal(M: np.ndarray) -> float:
    K = M.shape[0]
    mask = ~np.eye(K, dtype=bool)
    vals = M[mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


# ---------------------------------------------------------------------------
# Rank-k LEACE eraser via stacked centroid SVD
# ---------------------------------------------------------------------------

def make_rank_k_hook(P_proj: torch.Tensor, mu: torch.Tensor):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            tail = output[1:]
        else:
            h = output
            tail = ()
        new_h = mu + (h - mu) @ P_proj.T
        if tail:
            return (new_h,) + tail
        return new_h
    return hook


@torch.no_grad()
def evaluate_rank_k(bert, ids: torch.Tensor, mask: torch.Tensor,
                    P_proj: torch.Tensor, mu: torch.Tensor,
                    device: torch.device, batch_size: int) -> torch.Tensor:
    """Apply rank-k eraser at layer 0; return hidden_states[2][:, 0, :]
    (the post-erasure layer-1 [CLS])."""
    layer0 = bert.encoder.layer[0]
    h_handle = layer0.register_forward_hook(make_rank_k_hook(P_proj, mu))
    try:
        all_h1 = []
        for i in range(0, ids.shape[0], batch_size):
            b_ids = ids[i:i + batch_size].to(device)
            b_mask = mask[i:i + batch_size].to(device)
            out = bert(input_ids=b_ids, attention_mask=b_mask,
                       output_hidden_states=True, return_dict=True)
            all_h1.append(out.hidden_states[2][:, 0, :].cpu())
        return torch.cat(all_h1).float()
    finally:
        h_handle.remove()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/v2_bios_FINAL")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rank-mech] device={device} n={args.n} batch={args.batch_size}")
    t0 = time.time()

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    from pcrl.language.bios_dataset import _TOP10_SET, _LABEL_TO_LOCAL

    print(f"[rank-mech] loading BIOS dev split + filtering top-10...")
    dev = load_dataset("LabHC/bias_in_bios", split="dev")
    dev = dev.filter(lambda r: r["profession"] in _TOP10_SET, num_proc=1)
    n_total = len(dev)
    print(f"[rank-mech] dev n_total={n_total} @ {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    take = min(args.n, n_total)
    idx = rng.choice(n_total, size=take, replace=False)
    idx_sorted = sorted(idx.tolist())
    dev_sub = dev.select(idx_sorted)

    texts = list(dev_sub["hard_text"])
    professions = list(dev_sub["profession"])
    genders = list(dev_sub["gender"])
    print(f"[rank-mech] sampled n={len(texts)} @ {time.time()-t0:.1f}s")

    # ---- Token-type classification ----
    print(f"[rank-mech] classifying token-type membership...")
    types_list = [classify_sentence(t) for t in texts]
    counts = {"P": 0, "H": 0, "N": 0, "FN": 0,
              "multi": 0, "none": 0,
              "only_P": 0, "only_H": 0, "only_N": 0, "only_FN": 0}
    for ts in types_list:
        if len(ts) == 0:
            counts["none"] += 1
        elif len(ts) > 1:
            counts["multi"] += 1
            for t in ts:
                counts[t] += 1
        else:
            t = next(iter(ts))
            counts[t] += 1
            counts[f"only_{t}"] += 1
    print(f"[rank-mech] type counts: {counts}")

    # ---- Tokenize ----
    print(f"[rank-mech] tokenizing with bert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tokenizer(
        texts, max_length=args.max_length, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    input_ids = enc["input_ids"].long()
    attention_mask = enc["attention_mask"].long()

    # ---- Forward to capture hidden_states[1], [2], [7] ----
    print(f"[rank-mech] loading bert-base-uncased to {device}...")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()

    all_h1, all_h2, all_h7 = [], [], []
    n = input_ids.shape[0]
    bs = args.batch_size
    print(f"[rank-mech] forward {n} samples, batch {bs}...")
    with torch.no_grad():
        for i in range(0, n, bs):
            b_ids = input_ids[i:i + bs].to(device)
            b_mask = attention_mask[i:i + bs].to(device)
            out = bert(input_ids=b_ids, attention_mask=b_mask,
                       output_hidden_states=True, return_dict=True)
            all_h1.append(out.hidden_states[1][:, 0, :].cpu())  # layer-0 output
            all_h2.append(out.hidden_states[2][:, 0, :].cpu())  # layer-1 output
            all_h7.append(out.hidden_states[7][:, 0, :].cpu())  # layer-6 output
    H_layer0 = torch.cat(all_h1).float()  # post-BertLayer[0]
    H_layer1 = torch.cat(all_h2).float()  # post-BertLayer[1]
    H_layer6 = torch.cat(all_h7).float()  # post-BertLayer[6]
    G = torch.tensor(genders, dtype=torch.float32)
    print(f"[rank-mech] forward done @ {time.time()-t0:.1f}s; "
          f"H_layer0 shape={tuple(H_layer0.shape)}")

    # =====================================================================
    # E1: per-token-type centroid differences + cosines
    # =====================================================================
    print(f"\n[E1] per-token-type cosine analysis...")

    def get_only_indices(target: str) -> list[int]:
        return [i for i, ts in enumerate(types_list)
                if len(ts) == 1 and target in ts]

    types_order = ["P", "H", "N", "FN"]
    idx_only = {t: get_only_indices(t) for t in types_order}
    subset_sizes = {t: len(idx_only[t]) for t in types_order}
    print(f"[E1] disjoint subset sizes: {subset_sizes}")

    v0 = [centroid_diff(H_layer0, G, idx_only[t]) for t in types_order]
    v6 = [centroid_diff(H_layer6, G, idx_only[t]) for t in types_order]

    M0 = cosine_matrix(v0)
    M6 = cosine_matrix(v6)
    mean_off0 = mean_off_diagonal(M0)
    mean_off6 = mean_off_diagonal(M6)
    print(f"[E1] layer 0 cosine matrix:")
    for i, t in enumerate(types_order):
        row = "  ".join(f"{M0[i, j]:.3f}" for j in range(4))
        print(f"   {t:>3}: {row}")
    print(f"[E1] layer 0 mean off-diagonal cosine = {mean_off0:.4f}")
    print(f"[E1] layer 6 cosine matrix:")
    for i, t in enumerate(types_order):
        row = "  ".join(f"{M6[i, j]:.3f}" for j in range(4))
        print(f"   {t:>3}: {row}")
    print(f"[E1] layer 6 mean off-diagonal cosine = {mean_off6:.4f}")

    if np.isnan(mean_off0):
        e1_verdict = "INSUFFICIENT_SUBSET_SIZES"
    elif mean_off0 < 0.5:
        e1_verdict = "MULTI_RANK_CONFIRMED"
    elif mean_off0 > 0.7:
        e1_verdict = "SINGLE_RANK"
    else:
        e1_verdict = "PARTIALLY_MULTI_RANK"
    consolidation = (
        "CONSOLIDATION_OBSERVED" if (not np.isnan(mean_off6)
                                     and not np.isnan(mean_off0)
                                     and mean_off6 > mean_off0)
        else "NO_CONSOLIDATION"
    )
    print(f"[E1] verdict: {e1_verdict} / {consolidation}")

    e1_out = {
        "n_dev_sample": int(len(texts)),
        "type_counts": counts,
        "subset_sizes": subset_sizes,
        "types_order": types_order,
        "layer_0_cosine_matrix": M0.tolist(),
        "layer_0_mean_offdiag_cosine": mean_off0,
        "layer_6_cosine_matrix": M6.tolist(),
        "layer_6_mean_offdiag_cosine": mean_off6,
        "verdict_orthogonality": e1_verdict,
        "verdict_consolidation": consolidation,
    }
    (out_dir / "token_type_pca.json").write_text(json.dumps(e1_out, indent=2))
    print(f"[E1] saved → {out_dir/'token_type_pca.json'}")

    # =====================================================================
    # E2: rank-k LEACE sweep at layer 0
    # =====================================================================
    print(f"\n[E2] rank-k LEACE sweep at layer 0...")

    # Build expanded direction set: token-type centroid diffs + occupation-
    # conditional centroid diffs.
    occ_local = np.array([_LABEL_TO_LOCAL[p] for p in professions])
    occ_t = torch.tensor(occ_local, dtype=torch.long)
    occ_diffs: list[torch.Tensor] = []
    for o in range(10):
        oidx = (occ_t == o).nonzero(as_tuple=True)[0].tolist()
        v_o = centroid_diff(H_layer0, G, oidx, min_each=20)
        if v_o is not None:
            occ_diffs.append(v_o)
    print(f"[E2] occupation-conditional directions: {len(occ_diffs)}")

    direction_set = [v for v in v0 if v is not None] + occ_diffs
    if not direction_set:
        print("[E2] no usable directions — skipping rank sweep")
        e2_out = {"error": "no_directions"}
    else:
        V = torch.stack(direction_set, dim=1)  # (768, K)
        print(f"[E2] direction set shape={tuple(V.shape)}")
        U, S, _ = torch.linalg.svd(V, full_matrices=False)
        K_max = U.shape[1]
        print(f"[E2] SVD singular values: {[f'{s:.3f}' for s in S.tolist()]}")

        # Vanilla baseline R² at layer 1
        r2_vanilla = population_linear_r2(H_layer1, G)
        print(f"[E2] vanilla layer-1 R² = {r2_vanilla:.4f}")

        mu0 = H_layer0.mean(0).to(device)
        rank_results = {"vanilla_layer1_r2": r2_vanilla, "K_max": int(K_max)}
        for k in [1, 2, 4, 8]:
            if k > K_max:
                rank_results[f"rank_{k}_layer1_r2"] = None
                continue
            U_k = U[:, :k].to(device)  # (768, k)
            P_proj = (torch.eye(768, device=device) - U_k @ U_k.T)
            print(f"[E2] k={k} → rank-k forward over n={n}...")
            t_k = time.time()
            H1_k = evaluate_rank_k(
                bert, input_ids, attention_mask, P_proj, mu0,
                device, args.batch_size,
            )
            r2_k = population_linear_r2(H1_k, G)
            rank_results[f"rank_{k}_layer1_r2"] = r2_k
            print(f"   k={k}: layer-1 R² = {r2_k:.4f}  "
                  f"(elapsed {time.time()-t_k:.1f}s)")

        # Verdicts
        below_05_at = next(
            (k for k in [1, 2, 4, 8]
             if rank_results.get(f"rank_{k}_layer1_r2") is not None
             and rank_results[f"rank_{k}_layer1_r2"] < 0.5),
            None,
        )
        below_03_at = next(
            (k for k in [1, 2, 4, 8]
             if rank_results.get(f"rank_{k}_layer1_r2") is not None
             and rank_results[f"rank_{k}_layer1_r2"] < 0.3),
            None,
        )
        e2_out = {
            **rank_results,
            "first_k_below_0.5": below_05_at,
            "first_k_below_0.3": below_03_at,
            "singular_values": [float(s) for s in S.tolist()],
        }

    (out_dir / "rank_k_sweep.json").write_text(json.dumps(e2_out, indent=2))
    print(f"[E2] saved → {out_dir/'rank_k_sweep.json'}")

    # =====================================================================
    # Combined narrative
    # =====================================================================
    md_lines = [
        "# BIOS rank-mechanism findings (E1 + E2)",
        "",
        f"**Date:** auto-generated  ",
        f"**Sample:** BIOS top-10 dev, n={len(texts)} (frozen bert-base-uncased, no LoRA)  ",
        f"**Wall:** {time.time()-t0:.1f}s on {device}",
        "",
        "## E1 — Per-token-type cosine analysis",
        "",
        f"Disjoint subset sizes: {subset_sizes}.",
        "",
        "Layer-0 cosine matrix (rows/cols = P, H, N, FN):",
        "",
        "```",
    ]
    for i, t in enumerate(types_order):
        md_lines.append(f"   {t:>3}: " + "  ".join(f"{M0[i, j]:.3f}" for j in range(4)))
    md_lines += [
        "```",
        "",
        f"Layer-0 mean off-diagonal cosine: **{mean_off0:.4f}**",
        f"Layer-6 mean off-diagonal cosine: **{mean_off6:.4f}**",
        f"Verdict (orthogonality): **{e1_verdict}**  ",
        f"Verdict (consolidation):  **{consolidation}**",
        "",
        "## E2 — Rank-k LEACE sweep at layer 0",
        "",
    ]
    if e2_out.get("error"):
        md_lines.append(f"E2 skipped: {e2_out.get('error')}")
    else:
        md_lines += [
            f"Vanilla layer-1 R² (no eraser): **{e2_out['vanilla_layer1_r2']:.4f}**",
            "",
            "| k | layer-1 R² | Δ from vanilla |",
            "|---|-----------|---------------|",
        ]
        for k in [1, 2, 4, 8]:
            r = e2_out.get(f"rank_{k}_layer1_r2")
            if r is None:
                md_lines.append(f"| {k} | (skipped, k>K_max={e2_out['K_max']}) | — |")
            else:
                d = r - e2_out["vanilla_layer1_r2"]
                md_lines.append(f"| {k} | {r:.4f} | {d:+.4f} |")
        md_lines.append("")
        md_lines.append(f"First k with R² < 0.5: **{e2_out['first_k_below_0.5']}**  ")
        md_lines.append(f"First k with R² < 0.3: **{e2_out['first_k_below_0.3']}**")
    md_lines += [
        "",
        "## Synthesis for §5.5",
        "",
        "If E1 verdict = MULTI_RANK_CONFIRMED *and* E2 shows monotonic R² ",
        "decrease saturating around k=4: the layer-0 LEACE failure is a ",
        "rank-1 limitation (Belrose 2023 §3 specifies rank-1 erasure for ",
        "binary Z); gender at layer 0 is encoded multi-rank because ",
        "heterogeneous input-token sources (pronouns, honorifics, gendered ",
        "nouns, first names) project gender along distinct directions. ",
        "Mid-stack attention then consolidates these directions into a ",
        "lower-rank subspace where rank-1 erasure becomes effective ",
        "(layer-6 / layer-11 hooks succeed). This is the concept-level ",
        "analogue of attention rank-collapse phenomena described in Dong ",
        "et al. 2021 and Wang et al. 2025: attention's iterative weighted ",
        "mixing reduces the effective rank of token-position-distributed ",
        "concept signals into a low-rank residual.",
        "",
        "Citations: Belrose 2023 (LEACE), Bhardwaj 2020 (debiasing tradeoffs), ",
        "Zakizadeh 2025 (per-token erasure), Dong 2021 + Wang 2025 (attention ",
        "rank collapse).",
    ]
    md_path = out_dir / "rank_mechanism_findings.md"
    md_path.write_text("\n".join(md_lines))
    print(f"\n[combined] saved → {md_path}")
    print(f"[combined] total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
