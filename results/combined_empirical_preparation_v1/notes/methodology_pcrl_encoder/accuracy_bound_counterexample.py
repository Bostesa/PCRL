"""Standalone reproduction (numpy only, no repo imports) of the 20-observation
counterexample to the retired R²->accuracy guarantee (PCRL Prop. 3 / certified_accuracy_bound)."""
import json, math
import numpy as np

h = np.array([1.0]*9 + [-9.0] + [-1.0]*9 + [9.0])
a = np.array([1]*10 + [0]*10)
n = len(a)
cov = float(np.mean((h - h.mean()) * (a - a.mean())))
X = np.column_stack([np.ones(n), h])
beta, *_ = np.linalg.lstsq(X, a, rcond=None)
pred = X @ beta
r2_affine = float(1 - np.sum((a - pred)**2) / np.sum((a - a.mean())**2))
# one-hot multi-output R² (K=2 columns, variance-weighted) — same value
Z = np.eye(2)[a]
B, *_ = np.linalg.lstsq(X, Z, rcond=None)
r2_onehot = float(1 - np.sum((Z - X @ B)**2) / np.sum((Z - Z.mean(0))**2))
acc_thresh = float(np.mean((h > 0).astype(int) == a))
pi_max = float(np.bincount(a).max() / n)
k = 2
eps = r2_onehot
bound_code = min(max(pi_max + math.sqrt(eps * k * pi_max * (1 - pi_max)), pi_max), 1.0)   # origin/main verification.py:350
bound_paper = pi_max + k * math.sqrt(eps * pi_max * (1 - pi_max))                         # PDF(23) Prop.3 line 179
# best possible linear threshold classifier on this sample (exhaustive over thresholds and both signs)
cands = np.unique(np.concatenate([h - 1e-9, h + 1e-9]))
best = max(max(np.mean((h > t).astype(int) == a), np.mean((h <= t).astype(int) == a)) for t in cands)
out = dict(n=n, class_counts=np.bincount(a).tolist(), mean_h_given_A1=float(h[a==1].mean()), mean_h_given_A0=float(h[a==0].mean()),
           cov_h_A=cov, affine_ls_beta=beta.tolist(), r2_affine=r2_affine, r2_onehot_K2=r2_onehot,
           threshold_rule="predict A=1 iff h>0", threshold_accuracy=acc_thresh, best_linear_threshold_accuracy=float(best),
           majority=pi_max, old_bound_code=bound_code, old_bound_manuscript=bound_paper,
           violation_pp=round((acc_thresh - bound_code) * 100, 2))
print(json.dumps(out, indent=1))
json.dump(out, open(__file__.replace('.py', '_out.json'), 'w'), indent=1)

# ---- Multiclass (K=3) extension proposed as an additional regression fixture ----
H = np.array([[-1, -1]]*9 + [[9, 9]] + [[1, 0]]*9 + [[-9, 0]] + [[0, 1]]*9 + [[0, -9]], float)
A = np.array([0]*10 + [1]*10 + [2]*10)
Z = np.eye(3)[A]
X = np.column_stack([np.ones(len(A)), H])
B, *_ = np.linalg.lstsq(X, Z, rcond=None)
r2_oh = float(1 - np.sum((Z - X @ B)**2) / np.sum((Z - Z.mean(0))**2))
cross_cov = (H - H.mean(0)).T @ (Z - Z.mean(0)) / len(A)
scores = np.column_stack([np.zeros(len(A)), H[:, 0], H[:, 1]])   # affine scores w_j^T h + b_j
acc3 = float(np.mean(scores.argmax(1) == A))
pmax = 1/3
mc = dict(n=30, r2_onehot=r2_oh, max_abs_cross_cov=float(np.abs(cross_cov).max()), argmax_linear_accuracy=acc3, majority=pmax,
          old_bound_code=min(max(pmax + math.sqrt(r2_oh*3*pmax*(1-pmax)), pmax), 1.0),
          old_bound_manuscript=pmax + 3*math.sqrt(max(r2_oh,0)*pmax*(1-pmax)))
print(json.dumps(mc, indent=1))
out["multiclass_fixture"] = mc
json.dump(out, open(__file__.replace('.py', '_out.json'), 'w'), indent=1)
