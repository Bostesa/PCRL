"""Stage 1 (+ S3 usefulness block): frozen-head task usefulness on assessment; constant chosen on attacker_fit only.
Writes FROZEN_HEAD_UTILITY.csv (per purpose x seed x head) and returns graph ids for the endpoint inference."""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, log_loss
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from odx import run as R
from odx.surfaces import softmax
import oar.study as S

OAR_UNITS = Path.home() / "PCRL_eval_cache_private/oar_v1/run/units"


def rows_for(ds, purpose):
    p = R.purposes(ds)[purpose]
    W = R.world(ds, purpose, p["disallowed_attrs"][0])
    t = W["t"]
    f, a = W["idx"]["attacker_fit"], W["idx"]["assessment"]
    K = int(p["task_dim"])
    prior = np.bincount(t[f], minlength=K) / len(f)
    maj = int(np.argmax(prior))
    out = []
    for k in S.SEEDS:
        F = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")
        L = F[p["logits_key"]].astype(np.float64)[a]
        P = softmax(L)
        pred = L.argmax(1)
        ta = t[a]
        heads = [("frozen", pred, P)]
        if purpose == S.CELLS[ds]["purpose"]:
            u2 = np.load(OAR_UNITS / f"{ds}__s{k}__U2__A/preds.npz")
            assert np.array_equal(u2["assess_row_id"], W["row_id"][a])
            heads.append(("refit_probe_U2__A", u2["U2_P"].argmax(1), u2["U2_P"]))
            ha = np.load(OAR_UNITS / f"{ds}__s{k}__HEAD__A/preds.npz")["head_outputs_all"][a]
            heads.append(("refit_head_HEAD__A", ha.argmax(1), np.exp(ha)))
        for name, yhat, PP in heads:
            labels = list(range(K))
            if name == "frozen":                       # -log_softmax(l)[y] in float64 (no clipping)
                Z = L - L.max(1, keepdims=True)
                lsm = Z - np.log(np.exp(Z).sum(1, keepdims=True))
                LL, LLm = float(-lsm[np.arange(len(ta)), ta].mean()), "-log_softmax(logits)[y], float64"
            elif name == "refit_head_HEAD__A":
                LL, LLm = float(-ha[np.arange(len(ta)), ta].mean()), "-stored log-probability[y]"
            else:
                LL, LLm = float(log_loss(ta, np.clip(PP, 1e-15, 1), labels=labels)), "sklearn log_loss on predict_proba"
            n_disc_pos = int(((yhat == ta) & (ta != maj)).sum()); n_disc_neg = int(((yhat != ta) & (ta == maj)).sum())
            rec = np.array([((yhat == c) & (ta == c)).sum() / max((ta == c).sum(), 1) for c in labels])
            try:
                auc = roc_auc_score(ta, PP[:, 1]) if K == 2 else roc_auc_score(ta, PP, multi_class="ovr", labels=labels)
            except ValueError:
                auc = float("nan")
            out.append({"dataset": ds, "purpose": purpose, "seed": k, "head": name, "K": K, "n_assessment": int(len(a)),
                        "accuracy": float((yhat == ta).mean()), "balanced_accuracy": float(balanced_accuracy_score(ta, yhat)),
                        "log_loss": LL, "log_loss_method": LLm, "task_auc": float(auc),
                        "constant_class_from_attacker_fit": maj, "constant_accuracy": float((ta == maj).mean()),
                        "constant_log_loss": float(log_loss(ta, np.tile(np.clip(prior, 1e-12, 1), (len(ta), 1)), labels=labels)),
                        "gain_over_constant": float((yhat == ta).mean() - (ta == maj).mean()),
                        "per_class_recall": json.dumps([round(float(x), 4) for x in rec]),
                        "prediction_frequencies": json.dumps([round(float((yhat == c).mean()), 4) for c in labels]),
                        "true_frequencies": json.dumps([round(float((ta == c).mean()), 4) for c in labels]),
                        "confusion_matrix": json.dumps(confusion_matrix(ta, yhat, labels=labels).tolist()),
                        "is_constant_prediction": bool(len(np.unique(yhat)) == 1),
                        "discordant_head_right_const_wrong": n_disc_pos, "discordant_head_wrong_const_right": n_disc_neg,
                        "classes_present_in_assessment": int(len(np.unique(ta))),
                        "max_prediction_share": float(max((yhat == c).mean() for c in labels)),
                        "hard_decision_accuracy_identity": "hard = argmax of this head; identical accuracy; probability utility UNAVAILABLE for hard"})
    return out, (W, maj)


if __name__ == "__main__":
    rows = []
    for ds in ("adult", "hmda"):
        for purpose in R.purposes(ds):
            r, _ = rows_for(ds, purpose)
            rows += r
    PKG = WT / "results/combined_output_diagnosis_v1"
    with open(PKG / "FROZEN_HEAD_UTILITY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["dataset"], r["purpose"][:12], r["seed"], r["head"][:14], round(r["accuracy"], 4), round(r["constant_accuracy"], 4), round(r["gain_over_constant"], 4), round(r["balanced_accuracy"], 4), r["per_class_recall"])
