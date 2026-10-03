"""Synthetic-only numerical checks backing SCORE_MATH.md and STATS_REVIEW.md (math/protocol reviewer, 2026-10-03).
No study data are read. Run: python3 score_math_checks.py   (tested with numpy 2.4.4, scipy 1.17.1, sklearn 1.8.0)."""
import warnings; warnings.filterwarnings("ignore")

print('== A. identities ==')
import numpy as np
from scipy.special import expit, softmax, log_softmax
rng = np.random.default_rng(0)
# 1. invertible linear map (l0,l1) <-> (d,c)
A = np.array([[-1.0, 1.0], [0.5, 0.5]]); Ainv = np.array([[-0.5, 1.0], [0.5, 1.0]])
print("det A =", np.linalg.det(A), " A@Ainv == I:", np.array_equal(A @ Ainv, np.eye(2)))
L = rng.normal(0, 3, (10**6, 2))
d = L[:, 1] - L[:, 0]; c = (L[:, 0] + L[:, 1]) / 2
R = np.c_[c - d / 2, c + d / 2]
print("float64 inputs: max |roundtrip - l| =", np.abs(R - L).max(), " rows not bit-exact:", int((R != L).any(1).sum()))
L32 = L.astype(np.float32).astype(np.float64)   # stored logits are float64 upcasts of float32
d32 = L32[:, 1] - L32[:, 0]; c32 = (L32[:, 0] + L32[:, 1]) / 2
R32 = np.c_[c32 - d32 / 2, c32 + d32 / 2]
print("float32-upcast inputs: rows not bit-exact:", int((R32 != L32).any(1).sum()))
# 2. softmax depends on d only; argmax = 1[d>0]
P = softmax(L, axis=1)
print("max |softmax(l)_1 - sigma(d)| =", np.abs(P[:, 1] - expit(d)).max(),
      " max |softmax(l)_0 - sigma(-d)| =", np.abs(P[:, 0] - expit(-d)).max())
Lt = np.r_[L, [[1.5, 1.5], [-2.0, -2.0]]]; dt = Lt[:, 1] - Lt[:, 0]   # include exact ties
print("argmax(l) != 1[d>0] rows (incl. 2 exact ties):", int((Lt.argmax(1) != (dt > 0)).sum()))
# 3. centring invariance (binary)
C = np.c_[-d / 2, d / 2]
print("binary: max |softmax(l) - softmax(centred)| =", np.abs(P - softmax(C, axis=1)).max(),
      " centred == l - c rows differing:", int((C != L - c[:, None]).any(1).sum()))
# multiclass centring
for K in (3, 4, 5, 6):
    LK = rng.normal(0, 3, (10**5, K)).astype(np.float32).astype(np.float64)
    cK = LK.mean(1); CK = LK - cK[:, None]; RK = CK + cK[:, None]
    print(f"K={K}: max|softmax diff|={np.abs(softmax(LK,1)-softmax(CK,1)).max():.1e} "
          f"max|sum centred|={np.abs(CK.sum(1)).max():.1e} recon rows not bit-exact={int((RK!=LK).any(1).sum())} "
          f"max recon err={np.abs(RK-LK).max():.1e} argmax diff={int((LK.argmax(1)!=CK.argmax(1)).sum())}")
# 4. log-softmax head: offset is an even function of d
LS = log_softmax(L, axis=1); dL = LS[:, 1] - LS[:, 0]; cL = LS.mean(1)
g = -0.5 * (np.logaddexp(0, d) + np.logaddexp(0, -d))
print("log-prob head: max|d_logprob - d| =", np.abs(dL - d).max(), " max|c - g(d)| =", np.abs(cL - g).max(),
      " max|g(d)-g(-d)| =", np.abs(g - (-0.5 * (np.logaddexp(0, -d) + np.logaddexp(0, d)))).max())

print('== B. float saturation ==')
import numpy as np, math
from scipy.special import expit, softmax
def first_true(f, lo, hi, dtype):
    lo, hi = dtype(lo), dtype(hi)          # f(lo) False, f(hi) True, monotone; bisect over representable floats
    assert not f(lo) and f(hi)
    while True:
        mid = dtype((lo + hi) / 2)
        if mid == lo or mid == hi:
            nxt = np.nextafter(lo, hi)
            return hi if nxt == hi else (nxt if f(nxt) else hi)
        lo, hi = (lo, mid) if f(mid) else (mid, hi)
naive = lambda x: 1.0 / (1.0 + np.exp(-x))
print("largest double below 1: 1-2^-53 =", repr(1 - 2.0**-53), " nextafter(1,0) =", repr(np.nextafter(1.0, 0.0)))
print("53 ln2 =", 53 * math.log(2), " ln(2^54-1) =", math.log(2.0**54 - 1))
t = first_true(lambda x: expit(x) == 1.0, 30.0, 40.0, np.float64);  print("float64 expit(d)==1.0 for d >=", repr(t), " expit just below:", repr(expit(np.nextafter(t, 0))))
t = first_true(lambda x: naive(x) == 1.0, 30.0, 40.0, np.float64);  print("float64 naive 1/(1+e^-d)==1.0 for d >=", repr(t))
t = first_true(lambda x: softmax(np.array([0.0, x]))[1] == 1.0, 30.0, 40.0, np.float64); print("float64 scipy softmax p1==1.0 for d >=", repr(t))
t = first_true(lambda x: expit(-x) == 0.0, 700.0, 800.0, np.float64); print("float64 expit(d)==0.0 for d <= -", repr(t))
t = first_true(lambda x: naive(-x) == 0.0, 700.0, 800.0, np.float64); print("float64 naive (exp overflow) ==0.0 for d <= -", repr(t))
t = first_true(lambda x: expit(-x) < np.finfo(np.float64).tiny, 700.0, 800.0, np.float64); print("float64 expit(d) subnormal for d <= -", repr(t), " ln(2^-1022) =", math.log(2.0**-1022), " ln(2^-1075) =", -1075*math.log(2))
f32 = np.float32
t = first_true(lambda x: expit(f32(x)) == f32(1), 10.0, 30.0, f32); print("float32 expit(d)==1 for d >=", repr(t), " 24 ln2 =", 24*math.log(2))
t = first_true(lambda x: expit(-f32(x)) == f32(0), 80.0, 120.0, f32); print("float32 expit(d)==0 for d <= -", repr(t), " ln(2^-150) =", -150*math.log(2))
# margin recovery from probabilities
d = np.array([5., 15., 20., 25., 30., 33., 36., 36.7, 36.8, 40., -36.8, -40., -700., -740.])
p1 = expit(d); p0 = expit(-d)
from_p1 = np.log(p1) - np.log1p(-p1)          # p1 alone (p0 implied as 1-p1)
from_pair = np.log(p1) - np.log(p0)            # both stably computed probabilities stored
with np.errstate(all="ignore"):
    for di, a, b, q in zip(d, from_p1, from_pair, p1):
        print(f"d={di:8.1f}  p1={q!r:24}  |d - logit(p1)|={abs(di-a):.2e}  |d - (log p1 - log p0)|={abs(di-b):.2e}")
stable = lambda x: np.exp(x) / (1.0 + np.exp(x))
t = first_true(lambda x: stable(-x) == 0.0, 700.0, 800.0, np.float64); print("float64 stable e^d/(1+e^d)==0 for d <= -", repr(t))
t = first_true(lambda x: softmax(np.array([0.0, -x]))[1] == 0.0, 700.0, 800.0, np.float64); print("float64 scipy softmax p1==0 for d <= -", repr(t))
f32=np.float32
t = first_true(lambda x: softmax(np.array([0, x], dtype=f32))[1] == f32(1), 10.0, 30.0, f32); print("float32 softmax p1==1 for d >=", repr(t))
t = first_true(lambda x: softmax(np.array([0, -x], dtype=f32))[1] == f32(0), 80.0, 120.0, f32); print("float32 softmax p1==0 for d <= -", repr(t))

print('== C. information vs fitted AUC ==')
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def mi(pairs):                      # exact plug-in MI (bits) of a finite joint distribution given as {(s,r): prob}
    ps, pr = Counter(), Counter()
    for (s, r), p in pairs.items(): ps[s] += p; pr[r] += p
    return sum(p * np.log2(p / (ps[s] * pr[r])) for (s, r), p in pairs.items() if p > 0)
# (a) data processing: offset carries S, margin carries task bit Y (fixture 1)
J = {}
for s in (0, 1):
    for y in (0, 1):
        for e in (0, 1):          # offset noise
            d = 2 * y - 1; c = s + e            # c encodes S (noisily), d encodes task bit only
            J[(s, (d, c))] = J.get((s, (d, c)), 0) + 0.125
f = lambda r: (-r[0] / 2, r[0] / 2)            # centring: keep d only
Jf = Counter()
for (s, r), p in J.items(): Jf[(s, f(r))] += p
print(f"fixture offset-carries-S: I(S;full)={mi(J):.4f} bits  I(S;centred)={mi(Jf):.4f} bits")
# (b) fixed public head on released Z adds nothing: I(S; Z, h(Z)) = I(S; Z)
rng = np.random.default_rng(1)
Jz = Counter(); Jzh = Counter(); h = lambda z: int(z >= 2)
for s in (0, 1):
    for z in range(4):
        p = 0.5 * (0.4 if z == s + 1 else 0.2)
        Jz[(s, z)] += p; Jzh[(s, (z, h(z)))] += p
print(f"public fixed head: I(S;Z)={mi(Jz):.6f}  I(S;Z,h(Z))={mi(Jzh):.6f}")
# (c) MI ordering does not imply AUC ordering (non-nested releases, Bayes-optimal scores)
#  A: reveals S exactly w.p. 0.1, else uninformative.  B: binary symmetric channel, flip prob 0.4.
JA = Counter({(0, 'S0'): .05, (1, 'S1'): .05, (0, '?'): .45, (1, '?'): .45})
JB = Counter({(0, 0): .3, (0, 1): .2, (1, 1): .3, (1, 0): .2})
aucA = 0.1 * 1 + 0.9 * (0.1 + 0.9 * 0.5); aucB = 1 - 0.4
print(f"A: I={mi(JA):.4f} bits, Bayes AUC={aucA:.3f};  B: I={mi(JB):.4f} bits, Bayes AUC={aucB:.3f}")
# (d) fitted attackers: removing a deterministic function of the release can move AUC either way
n = 4000
l = rng.normal(0, 1, (3 * n, 2)); dd = l[:, 1] - l[:, 0]
s = (dd + 0.5 * rng.normal(size=3 * n) > 0).astype(int)
tr, te = slice(0, n), slice(2 * n, 3 * n)
def auc(model, X): model.fit(X[tr], s[tr]); return roc_auc_score(s[te], model.predict_proba(X[te])[:, 1])
stump = lambda: HistGradientBoostingClassifier(max_depth=1, max_iter=10, random_state=0)
print(f"stump GBT on (l0,l1,d): {auc(stump(), np.c_[l, dd]):.3f}  -> remove d (a function of (l0,l1)): {auc(stump(), l):.3f}")
W = rng.normal(0, 40, (2, 400)); ph = rng.uniform(0, 2 * np.pi, 400)
junk = np.sin(l @ W + ph)                           # deterministic, high-frequency functions of (l0,l1)
n2 = 300; tr = slice(0, n2)
lr = lambda: LogisticRegression(C=10.0, max_iter=5000)
print(f"LR (n_fit={n2}) on (l0,l1,junk(l)): {auc(lr(), np.c_[l, junk]):.3f}  -> remove junk: {auc(lr(), l):.3f}")

print('== D. bootstrap / normal approximation ==')
import numpy as np
from scipy.stats import norm, skew, kurtosis
rng = np.random.default_rng(20261021)
def wauc(y, s, W):                    # weighted Mann-Whitney AUC, ties 1/2; W: (B, n) weights
    u, inv = np.unique(s, return_inverse=True)
    out = []
    for w in np.atleast_2d(W):
        gp = np.bincount(inv, w * y, len(u)); gn = np.bincount(inv, w * (1 - y), len(u))
        out.append((gp * (np.cumsum(gn) - gn + 0.5 * gn)).sum() / (gp.sum() * gn.sum()))
    return np.array(out)
def draw(n):                          # fixed "fitted" scorers on fresh rows: offset-using vs centred-only
    y = (rng.random(n) < 1 / 3).astype(float)
    d = rng.normal(0, 1, n) + 0.75 * y; c = rng.normal(0, 1, n) + 0.55 * y
    return y, d + c, d                # full-attacker score, centred-attacker score
n, B = 5000, 1999
truth = []
for _ in range(400):
    y, a, b = draw(n); one = np.ones((1, n)); truth.append(wauc(y, a, one)[0] - wauc(y, b, one)[0])
print(f"sampling SD of AUC difference over 400 fresh n={n} samples: {np.std(truth, ddof=1):.5f}  (mean diff {np.mean(truth):.4f})")
for rep in range(3):
    y, a, b = draw(n); W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    diff = wauc(y, a, W) - wauc(y, b, W)
    se = diff.std(ddof=1); z = norm.isf(0.05 / 60)
    q = np.quantile(diff, [0.005, 0.995]); nq = diff.mean() + np.array([-1, 1]) * norm.isf(0.005) * se
    print(f"bootstrap rep {rep}: SE={se:.5f} skew={skew(diff):+.3f} exkurt={kurtosis(diff):+.3f} "
          f"0.5/99.5% quantiles {q.round(4)} vs normal {nq.round(4)};  z*SE={z*se:.4f}")
print("MC relative SD of SE at B=1999 (normal theory): %.4f" % (1 / np.sqrt(2 * (B - 1))))
# degenerate case: identical predictions in both views -> every replicate difference is exactly 0
y, a, b = draw(n); W = rng.multinomial(n, np.full(n, 1 / n), size=200).astype(float)
dd = wauc(y, a, W) - wauc(y, a, W); print("identical views: SE =", dd.std(ddof=1), " -> interval [0,0]; lower 0 < 0.02 -> NOT_ESTABLISHED")
# accuracy difference with few discordant rows
corr_h = np.ones(n); corr_c = np.ones(n); corr_h[:6] = 1; corr_c[:6] = 0   # head right / constant wrong on 6 rows only
W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
g = (W * (corr_h - corr_c)).sum(1) / n
print(f"6 discordant rows: est={6/n:.5f} boot SE={g.std(ddof=1):.5f} skew={skew(g):+.2f}  lower={6/n - norm.isf(0.05/60)*g.std(ddof=1):+.5f}")
n_i, n_j, B = 3083, 302, 1999                  # HMDA-like smallest supported pair (group 1 has 302 assessment rows)
y = np.r_[np.zeros(n_i), np.ones(n_j)]; n = len(y)
a = rng.normal(0, 1, n) + 0.6 * y; b = a + rng.normal(0, 0.7, n)   # two correlated pair scores
W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
diff = wauc(y, a, W) - wauc(y, b, W)
print(f"pair {n_i} vs {n_j}: point={wauc(y,a,np.ones((1,n)))[0]-wauc(y,b,np.ones((1,n)))[0]:.4f} SE={diff.std(ddof=1):.4f} skew={skew(diff):+.3f} exkurt={kurtosis(diff):+.3f}")
# zero-denominator pair score: rule den==0 -> 0.5 (tie), as stored_model_eval.metrics.pair_score
P = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.2, 0.7, 0.1]])
den = P[:, 0] + P[:, 1]
print("pair score P1/(P0+P1):", np.where(den > 0, P[:, 1] / np.maximum(den, 1e-300), 0.5))
# log-domain equivalent: sigma(log P_j - log P_i) has the same ranks wherever both logs are finite
lp = np.log(np.array([[1e-320, 3e-321], [0.3, 0.6]]))
print("log-domain score:", 1 / (1 + np.exp(lp[:, 0] - lp[:, 1])), " ratio score:", np.exp(lp[:, 1]) / (np.exp(lp[:, 0]) + np.exp(lp[:, 1])))
