"""Equal-strength attacker banks for the decision-preserving compression study (audit/baseline owner).

Not a training critic. Every attacker is FITTED on AUDIT_FIT (6,065 rows) and SELECTED on INNER_SELECTION (2,235 rows);
only dpc.assess passes assessment rows to `final_audit`, after its pushed-EVALUATION_LOCK gate. Every inner entry point
runs osf.audit.guard_inner (AUDIT_FIT / INNER_SELECTION positions must carry exactly those roles, no overlap with the
assessment, no sealed labels) and never indexes the assessment role. Labels read here: SEX on AUDIT_FIT and
INNER_SELECTION (inner), plus SEX on the caller's `score_rows` (final; dpc.assess only).

ONE SLATE FOR EVERY PRIMARY CONTRACT (prompt section 11). The pinned source FINAL slate smf.audit.final_slate(False)
(itself jcv.audit factories, unchanged): LR (StandardScaler + LogisticRegression, C in {0.01, 0.1, 1, 10, 100}), MLP
(StandardScaler + MLP hidden in {(64,), (128,), (64,64), (128,128)}, adam, alpha 1e-4, early stopping on 10% of
AUDIT_FIT, max_iter 300, random_state = attacker seed), HGB (lr {0.05, 0.1} x leaves {15, 31}, 200 iterations, no early
stopping), DA_LR / DA_MLP (smf.audit.Canon: float64 centre, SVD on AUDIT_FIT, drop s_j <= max(n, d) eps64 max(s_1,
||X||_2), whiten the rest). It is used for policy code views, source complete / scores / probabilities / decisions
views, reference score views and the m = 1 class-only views alike, in inner audits and in the final audit.

FINITE VIEWS ADD CELL-CONDITIONAL SEX READERS (new for this study; smf used alpha {0.1, 1, 10}). A view is finite when
the release declares an exact identity key per person (`tokens`): policy tokens tok_i, decisions d_i, FARE cells, or
the exact float64 value of a finitely-valued view (`value_tokens`). Readers, fitted on AUDIT_FIT only:
  local  CC_alpha{a}, a in {0.5, 1, 5}:  P(S=1 | t) = (n_t1 + a pi_1) / (n_t + a)       (prior-weighted Dirichlet, the
         predecessor jcv.audit.CellConditional convention); pi = AUDIT_FIT SEX prior; a token unseen in AUDIT_FIT -> pi.
  pair   CCpair_alpha{a}: keyed on the ACTUAL token tuple (t1, t2), same formula; a tuple unseen in AUDIT_FIT uses one
         fixed fallback rule R_a in {local_1 (local reader of recipient 1, same a), local_2, prior}, chosen BEFORE any
         scoring as the lowest INNER_SELECTION cross-entropy over the INNER_SELECTION rows whose tuple is unseen (ties ->
         local_1, local_2, prior); no unseen INNER_SELECTION tuple -> prior. R_a is frozen into the reader and applied
         unchanged to scored rows. Coverage and fallback counts are published per view (`coverage`).
  The readers key on the integer identity only through equality, so any invertible renumbering leaves every prediction
  unchanged, and two IDs that decode to identical probabilities stay distinct cells.

CATEGORICAL REPRESENTATION FOR THE SLATE. Policy recipient view i = [one-hot(token over the FULL alphabet alpha_i),
decoded probabilities q_i, one-hot(decision_i)]; pair = [v1, v2] (aligned rows). The one-hot COLUMN ORDER is a function
of where tokens occur, never of their numeric IDs: tokens ordered by first occurrence in AUDIT_FIT (D row order), then
in INNER_SELECTION, then in all release rows (label-free), then never-occurring alphabet entries (all-zero columns,
trailing). Hence a renumbered release yields a bit-identical design matrix and bit-identical predictions for EVERY slate
member, not only the cell readers (test_audit). Integer token IDs are never an attack feature.

BANKS AND DUAL SELECTION. Per target view the candidates are, in this order:
  v1 / v2  own:<v>:<attacker>                              (slate + local CC readers)
  pair     coalition:pair:<attacker> (slate on [v1, v2] + CCpair), ignore_recipient_2:v1:<attacker> (every v1
           candidate), ignore_recipient_1:v2:<attacker> (every v2 candidate)
  composed (source releases only; below) the same lists with prefix composed[<policy unit>]:
AUC-selected attacker (primary) = highest INNER_SELECTION AUC; CE-selected attacker (proper loss) = lowest
INNER_SELECTION cross-entropy; kept separately; ties (1e-12) -> earlier bank order. Orientation fixed: score =
P(S = 1) from the attacker's class-1 column; an AUC below 0.5 stays below 0.5; nothing is flipped or clamped, on any
row. The inner coalition AUC is >= the better inner local AUC as a property of the bank; on scored rows the selected
coalition attacker is scored on its own. Cross-entropy uses CLIP = 1e-12 on P(S = s_true), with P(S=0) = 1 - P(S=1).

COMPOSED SOURCE READERS (section 11). The policy maps are public and recipient i's map reads only recipient i's teacher
probability vector p_i, which the source complete [r_i, c_i] and score c_i views determine exactly through the public
deployed head (softmax of the centred logits / the head on r_i; the probability view is p_i itself). So an attacker of
a source score view may apply any fitted policy of the SAME teacher and seed, and every policy code-view candidate is
also a source-view candidate: its predictions on INNER_SELECTION are, by construction, the policy's own stored inner
predictions (same AUDIT_FIT fit on the same derived views). `composed_inner` therefore REUSES the stored predictions of
the policy inner units instead of refitting (documented in every composed record; the row sets are hash-checked), and
returns per view the maximum inner AUC (and minimum CE) over the source's own bank and every policy bank, with the
winning candidate id. The decisions-only source family composes only with class-only (m = 1) policies, whose tokens are
functions of d_i; any other policy is refused there. Recipient i's composed candidates are composed only into recipient
i's view, and a policy's pair candidates only into the pair view (no cross-policy mixtures). The protected-code
attackers themselves never receive source scores. In `final_audit` the full-slate code readers of the locked selected
policies join the source banks (`composed=`), are selected on INNER_SELECTION with everything else, and the selected
one is refit at the attacker seeds; identical (view, labels, rows, attacker, seed) fits are memoised within a process
(`memo`), which is mathematically identical reuse, not an approximation.

INNER AUDIT RECORD (`inner_audit`): auc / ce / selected / ce_selected per view; worse_local, mean_local,
coalition_minus_best_local; tables (every candidate, inner_auc, inner_ce); selection with the full banks; coverage
(finite views); n_fit, n_select, slate members, cc alphas, meta; and PRIVATE: inner_predictions = {"keys": [...],
"P": (n_keys, n_select) float64 P(S = 1)} plus sel_row_id. `public_record` strips the arrays (keeps their sha256).

RELEASE FAMILIES. policy: code (primary) = [one-hot token, q_i, one-hot decision]. source (tea__s{k}__<teacher>,
teacher.npz): interface [c_i, p_i] (primary; the continuous score interface), complete [r_i, c_i] (the old primary
view), scores c_i, probs p_i, decisions one-hot(d_i). reference (ref__s{k}__<label>): dpc.baselines.reference_view_sets
(interface primary; scores, probs, decisions; F / F0 cells; E complete, descriptive). Probability and centred-logit
views are alternative interfaces to the same output, not independent releases: p = softmax(c) exactly in real
arithmetic; in float64 a probability underflows to 0 below ~4.9e-324 (exp(-746) = 0) or rounds to 1 when
1 - p < 2^-54, where the centred logit stays finite (the predecessor's amendment A1), so the two views can differ in
information only beyond those limits (and log loss uses the 1e-12 clip on either).

UNIT CONTRACT (lead's dpc.run.stage_inner): `inner_unit(kind, k, cid, D)` returns the JSON-safe record {recovery
(primary family), families {secondary}, utility {"0", "1"} (dpc.utility on INNER_SELECTION), class_preservation_ok,
token_states (policies), ...} and "_files" = {"inner_preds.npz": writer} with keys_<family>, P_<family> (n_candidates,
n_INNER) and sel_row_id; the lead saves it as inner__<release unit>. `load_inner(unit)` re-attaches the predictions
(hash-checked); `composed_from_units(k, teacher, policy_cids, D, family)` composes them without refitting.

REAL-DATA CONTROLS (`run_prelock_controls`; the LEAD runs it after SELECTION_AND_AUDIT_LOCK; inner roles only, sealed
D). INNER_SELECTION is split into exact-record-group halves A / B (smf.audit.null_split); SEX is permuted within
AUDIT_FIT, A and B (smf.audit.frozen_permutation, drawn and hashed before any fit) to S*; each check fits on AUDIT_FIT,
SELECTS on A over the whole bank and EVALUATES that attacker on held-out B (slate selection accounted for).
  null       the real release vs S*: pass iff AUC_B <= 0.5 + 3.5 sd0 (Mann-Whitney null sd on B).
  plants     declared-interface transformations of noisy S* (20% of rows replaced by a random class; a perfect reader
             has AUC 0.9), sent through np.savez_compressed / np.load and `policy_views`; pass iff AUC_B > 0.75:
    CONF_r{i}   within-predicted-class confidence token: token t of recipient i becomes (t, b), b = noisy S*, the b = 1
                copy decoding to (1 - 0.05) q_t + 0.05 e_d (more confident, SAME decision d); the decisions-only audit
                of the planted release is reported alongside (it must stay near null: an audit that only checks
                predicted labels is caught by this control, not passed by it).
    COLL_r{i}   decoder collision: the same split with IDENTICAL decoded probabilities for both copies; a decoded-
                probability-only audit is reported alongside (it cannot see the plant).
    XOR         coalition plant: recipient 1 split by b1 ~ Bernoulli(1/2) (independent), recipient 2 by b2 = b1 XOR
                noisy S*, both collision-style; each local view is independent of S* (reported against the null
                threshold), the pair must detect it.
  null_calibration   repeated permutations (seed CONTROL_SEED + 100 + rep) on one release; every rep kept.
Failures are recorded, never dropped. Thresholds are smf.audit's (NULL_Z 3.5, PLANT_MIN 0.75), unchanged.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from osf.audit import guard_inner
from smf import audit as AU
from smf.audit import final_slate, inner_slate, null_sd  # noqa: F401  (pinned slate and null sd, unchanged)

HOME = Path.home()
UNITS = HOME / "PCRL_eval_cache_private" / "dpc_v1" / "run" / "units"
FIT_ROLE, SEL_ROLE = "AUDIT_FIT", "INNER_SELECTION"
ASSESS_NAMES = ("OSF_DEVELOPMENT_ASSESSMENT", "DEVELOPMENT_ASSESSMENT")
ATT_SEEDS = (0, 1, 2)
CC_ALPHAS = (0.5, 1.0, 5.0)
FALLBACK_RULES = ("local_1", "local_2", "prior")
CLIP = 1e-12
TIE = 1e-12
KS = (2, 6)
PRIMARY_VIEWS = ("v1", "v2", "pair")
LOCAL_IGNORE = (("v1", "ignore_recipient_2"), ("v2", "ignore_recipient_1"))
SOURCE_FAMILIES = ("interface", "complete", "scores", "probs", "decisions")   # interface = primary
SLATES = {"final": lambda: final_slate(False), "inner": lambda: inner_slate(False)}
CONTROL_SEED, NULL_Z, PLANT_MIN = AU.CONTROL_SEED, AU.NULL_Z, AU.PLANT_MIN
CONF_ETA = 0.05


# ------------------------------------------------------------------ small helpers
def sha_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def safe(label):
    return str(label).replace("*", "star").replace("/", "_").replace("|", "_").replace(" ", "_")


def onehot(idx, K):
    idx = np.asarray(idx, dtype=np.int64)
    out = np.zeros((len(idx), int(K)))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def auc1(y, p1):
    """Fixed-orientation binary AUC of the score P(S = 1) for y == 1 (never flipped)."""
    return float(roc_auc_score(np.asarray(y) == 1, np.asarray(p1, dtype=np.float64)))


def ce1(y, p1):
    y = np.asarray(y)
    p1 = np.asarray(p1, dtype=np.float64)
    pt = np.where(y == 1, p1, 1.0 - p1)
    return float(-np.mean(np.log(np.clip(pt, CLIP, 1.0))))


def _p1(model, X):
    P = model.predict_proba(X)
    cl = list(getattr(model, "classes_", range(P.shape[1])))
    if 1 not in [int(c) for c in cl]:
        return np.zeros(len(X))
    return np.asarray(P[:, [int(c) for c in cl].index(1)], dtype=np.float64)


def _pick(rows, key, maximize):
    best = None
    for j, r in enumerate(rows):
        v = r[key]
        if best is None or (v > rows[best][key] + TIE if maximize else v < rows[best][key] - TIE):
            best = j
    return best


def roles(D):
    return np.asarray(D["idx"][FIT_ROLE]), np.asarray(D["idx"][SEL_ROLE])


def _check_score_rows(D, score_rows, y):
    f, s = roles(D)
    score_rows = np.asarray(score_rows)
    if np.intersect1d(score_rows, f).size or np.intersect1d(score_rows, s).size:
        raise ValueError("REFUSED: scored rows overlap the attacker fitting or selection rows")
    AU.check_labels(np.asarray(y), score_rows)
    return score_rows


# ------------------------------------------------------------------ views
def canonical_columns(tok, D, alphabet):
    """Column position of every token value in [0, alphabet): first occurrence in AUDIT_FIT (D order), then in
    INNER_SELECTION, then in all rows; never-occurring values last (all-zero columns). A function of the token
    PARTITION of the rows, not of the numeric IDs."""
    tok = np.asarray(tok, dtype=np.int64)
    if tok.size and (tok.min() < 0 or tok.max() >= alphabet):
        raise ValueError(f"REFUSED: token ids outside [0, {alphabet})")
    f, s = roles(D)
    order, seen = [], set()
    for ix in (np.sort(f), np.sort(s), np.arange(len(tok))):
        t = tok[ix]
        _, first = np.unique(t, return_index=True)
        for v in t[np.sort(first)]:
            if int(v) not in seen:
                seen.add(int(v))
                order.append(int(v))
    order += [v for v in range(int(alphabet)) if v not in seen]
    col = np.empty(int(alphabet), dtype=np.int64)
    col[np.asarray(order, dtype=np.int64)] = np.arange(int(alphabet))
    return col


def value_tokens(X):
    """Exact identity of each row's released value (float64 bytes): finite-view key for a finitely-valued view."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    _, inv = np.unique(X, axis=0, return_inverse=True)
    return np.asarray(inv, dtype=np.int64).ravel()


def align(z, D):
    """Release arrays in D row order (refuses a release that does not cover exactly D's rows in D order)."""
    zz = {k: np.asarray(z[k]) for k in (z.files if hasattr(z, "files") else z)}
    if "row_id" in zz and not np.array_equal(zz["row_id"], D["row_id"]):
        raise ValueError("REFUSED: release rows are not aligned with D (row_id order)")
    n = len(D["row_id"])
    for k, v in zz.items():
        if v.ndim >= 1 and v.shape[0] not in (n,) and k not in ("alpha1", "alpha2", "row_id") and v.size > 1:
            raise ValueError(f"REFUSED: release array {k} has {v.shape[0]} rows, D has {n}")
    return zz


def token_decoder_check(tok, q, hard):
    """Every token must decode to ONE probability vector and ONE decision (no per-person state beyond the token),
    probabilities finite, nonnegative, summing to one, and argmax = the decision (class preservation of the output)."""
    tok = np.asarray(tok, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.int64)
    bad = []
    if not np.all(np.isfinite(q)) or q.min() < 0:
        bad.append("decoded probabilities not finite and nonnegative")
    if np.max(np.abs(q.sum(1) - 1.0)) > 1e-9:
        bad.append("decoded probabilities do not sum to one")
    if not np.array_equal(q.argmax(1), hard):
        bad.append("argmax of decoded probabilities differs from the released decision")
    _, first, inv = np.unique(tok, return_index=True, return_inverse=True)
    inv = inv.ravel()
    if not (np.array_equal(q, q[first][inv]) and np.array_equal(hard, hard[first][inv])):
        bad.append("a token decodes to more than one probability vector / decision")
    return {"ok": not bad, "problems": bad, "states_used": int(len(first))}


def code_view(tok, q, hard, alphabet, K, D):
    col = canonical_columns(tok, D, alphabet)
    return np.hstack([onehot(col[np.asarray(tok, dtype=np.int64)], alphabet), np.asarray(q, dtype=np.float64),
                      onehot(hard, K)])


def policy_views(z, D, meta=None):
    """Code views of a policy release (tok1, tok2, q1, q2, hard1, hard2, alpha1, alpha2, row_id)."""
    zz = align(z, D)
    X, T, chk = {}, {}, {}
    for i, K in ((1, KS[0]), (2, KS[1])):
        tok = zz[f"tok{i}"].astype(np.int64)
        a = int(np.asarray(zz[f"alpha{i}"]).ravel()[0])
        q, h = zz[f"q{i}"], zz[f"hard{i}"]
        c = token_decoder_check(tok, q, h)
        if not c["ok"]:
            raise ValueError(f"REFUSED: recipient {i} interface invalid: {c['problems']}")
        chk[f"v{i}"] = {**c, "alphabet": a}
        X[f"v{i}"] = code_view(tok, q, h, a, K, D)
        T[f"v{i}"] = tok
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return {"family": "code", "X": X, "tokens": T, "meta": {**(meta or {}), "interface": chk,
            "view": "[one-hot token (full alphabet, occurrence-ordered columns), decoded q_i, one-hot decision_i]"}}


def source_views(t, D, family, meta=None):
    """Source continuous-release views of a frozen teacher (tea__s{k}__<teacher>/teacher.npz or dpc.admit.teacher:
    p1 p2 d1 d2 c1 c2 r1 r2). Families: interface [c_i, p_i] (the continuous score interface; PRIMARY), complete
    [r_i, c_i] (features + centred scores; the old primary view), scores c_i, probs p_i, decisions one-hot(d_i)
    (finite, tokens d_i). Also used for reference score releases (dpc.baselines)."""
    n = len(D["row_id"])
    if "row_id" in t and not np.array_equal(np.asarray(t["row_id"]), D["row_id"]):
        raise ValueError("REFUSED: teacher outputs are not aligned with D")
    need = {"interface": ("c", "p"), "complete": ("r", "c"), "scores": ("c",), "probs": ("p",), "decisions": ("d",)}
    if family not in need:
        raise ValueError(f"unknown source family {family!r}")
    X, T = {}, None
    for i, K in ((1, KS[0]), (2, KS[1])):
        for key in need[family]:
            if len(np.asarray(t[f"{key}{i}"])) != n:
                raise ValueError(f"REFUSED: teacher array {key}{i} is not over D's rows")
        if family == "decisions":
            T = T or {}
            d = np.asarray(t[f"d{i}"], dtype=np.int64)
            X[f"v{i}"], T[f"v{i}"] = onehot(d, K), d
        else:
            X[f"v{i}"] = np.hstack([np.asarray(t[f"{key}{i}"], dtype=np.float64).reshape(n, -1)
                                    for key in need[family]])
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return {"family": family, "X": X, "tokens": T, "meta": {**(meta or {}), "family": family}}


def finite_by_value(views):
    """Mark a finitely-valued view set finite: tokens = exact float64 value identity of each recipient view."""
    views = dict(views)
    views["tokens"] = {w: value_tokens(views["X"][w]) for w in ("v1", "v2")}
    views["meta"] = {**views.get("meta", {}), "tokens": "exact value identity (finite-valued view)"}
    return views


def decision_views(d1, d2, D, meta=None):
    return source_views({"d1": d1, "d2": d2}, D, "decisions", meta)


# ------------------------------------------------------------------ cell-conditional readers (exact identity)
def _pair_keys(t1, t2):
    t1, t2 = np.asarray(t1, dtype=np.int64), np.asarray(t2, dtype=np.int64)
    base = int(max(t2.max(initial=0), 0)) + 1
    return t1 * base + t2


def _cc_table(keys_fit, y_fit):
    u, inv = np.unique(keys_fit, return_inverse=True)
    n = np.bincount(inv.ravel(), minlength=len(u)).astype(np.float64)
    n1 = np.bincount(inv.ravel(), weights=(np.asarray(y_fit) == 1).astype(np.float64), minlength=len(u))
    return u, n, n1


def _cc_lookup(table, keys, alpha, prior1):
    """(P(S=1), seen mask) of the prior-weighted Dirichlet cell reader; unseen -> prior1."""
    u, n, n1 = table
    pos = np.searchsorted(u, keys)
    pos_c = np.clip(pos, 0, max(len(u) - 1, 0))
    seen = (len(u) > 0) & (u[pos_c] == keys) if len(u) else np.zeros(len(keys), bool)
    p = np.full(len(keys), prior1, dtype=np.float64)
    p[seen] = (n1[pos_c[seen]] + alpha * prior1) / (n[pos_c[seen]] + alpha)
    return p, seen


def cc_local(tok, y, fit_idx, pred_idx, alphas=CC_ALPHAS):
    """Local cell readers on exact token identity: {f"CC_alpha{a}": P(S=1) on pred_idx}, coverage."""
    tok = np.asarray(tok, dtype=np.int64)
    prior1 = float(np.mean(np.asarray(y)[fit_idx] == 1))
    tab = _cc_table(tok[fit_idx], np.asarray(y)[fit_idx])
    out, seen = {}, None
    for a in alphas:
        out[f"CC_alpha{a}"], seen = _cc_lookup(tab, tok[pred_idx], a, prior1)
    return out, {"prior1": prior1, "tokens_in_fit": int(len(tab[0])), "seen": seen}


def cc_pair(t1, t2, y, fit_idx, pred_idx, sel_pos, alphas=CC_ALPHAS, rules=None):
    """Pair cell readers on the actual tuple (t1, t2); unseen tuples use a fallback rule chosen per alpha on the
    INNER_SELECTION positions `sel_pos` of pred_idx (lowest CE over unseen-tuple rows) unless `rules` fixes them."""
    y = np.asarray(y)
    keys = _pair_keys(t1, t2)
    prior1 = float(np.mean(y[fit_idx] == 1))
    tab = _cc_table(keys[fit_idx], y[fit_idx])
    loc1, _ = cc_local(t1, y, fit_idx, pred_idx, alphas)
    loc2, _ = cc_local(t2, y, fit_idx, pred_idx, alphas)
    ys = y[pred_idx][sel_pos]
    out, chosen, ce_table = {}, {}, {}
    seen = None
    for a in alphas:
        p, seen = _cc_lookup(tab, keys[pred_idx], a, prior1)
        cand = {"local_1": loc1[f"CC_alpha{a}"], "local_2": loc2[f"CC_alpha{a}"],
                "prior": np.full(len(pred_idx), prior1)}
        un_sel = ~seen[sel_pos]
        if rules is not None:
            r = rules[f"CCpair_alpha{a}"]
        elif un_sel.sum() == 0:
            r = "prior"
            ce_table[f"CCpair_alpha{a}"] = None
        else:
            ces = {k: ce1(ys[un_sel], cand[k][sel_pos][un_sel]) for k in FALLBACK_RULES}
            ce_table[f"CCpair_alpha{a}"] = ces
            r = min(FALLBACK_RULES, key=lambda k: (ces[k], FALLBACK_RULES.index(k)))
        chosen[f"CCpair_alpha{a}"] = r
        p = p.copy()
        p[~seen] = cand[r][~seen]
        out[f"CCpair_alpha{a}"] = p
    return out, {"prior1": prior1, "pairs_in_fit": int(len(tab[0])), "seen": seen, "fallback_rule": chosen,
                 "fallback_ce_on_unseen_inner_rows": ce_table}


# ------------------------------------------------------------------ bank fitting
def _fit_slate_view(X, y, fit_idx, pred_idx, slate, seed, memo=None, xhash=None):
    """{attacker: P(S=1) on pred_idx} for every slate member at one seed; memoised on content when memo is given."""
    out, secs = {}, {}
    for name, fac in slate:
        key = None
        if memo is not None:
            key = (xhash, name, seed)
            if key in memo:
                out[name], secs[name] = memo[key], 0.0
                continue
        t0 = time.time()
        m = fac(seed).fit(X[fit_idx], y[fit_idx])
        out[name] = _p1(m, X[pred_idx])
        secs[name] = round(time.time() - t0, 3)
        if memo is not None:
            memo[key] = out[name]
    return out, secs


def _content_hash(X, y, fit_idx, pred_idx):
    return sha_arrays(np.asarray(X)[fit_idx], np.asarray(y)[fit_idx], np.asarray(X)[pred_idx],
                      np.asarray(fit_idx), np.asarray(pred_idx))


def _source_bank(views, y, fit_idx, pred_idx, sel_pos, slate_name, seed, finite, memo, rules=None, prefix=""):
    """Fit one view set: {pred_key: P}, {pred_key: (kind, view, attacker)}, coverage, timings."""
    slate = SLATES[slate_name]()
    preds, origin, cov, secs = {}, {}, {}, {}
    toks = views.get("tokens") if finite else None
    for w in PRIMARY_VIEWS:
        X = np.asarray(views["X"][w], dtype=np.float64)
        xh = _content_hash(X, y, fit_idx, pred_idx) if memo is not None else None
        P, s = _fit_slate_view(X, y, fit_idx, pred_idx, slate, seed, memo, xh)
        for name in P:
            preds[f"{prefix}{w}:{name}"] = P[name]
            origin[f"{prefix}{w}:{name}"] = ("slate", w, name)
        secs[w] = s
        if toks is None:
            continue
        t0 = time.time()
        if w == "pair":
            Pc, c = cc_pair(toks["v1"], toks["v2"], y, fit_idx, pred_idx, sel_pos,
                            rules=(rules or {}).get(f"{prefix}pair"))
            cov[w] = {"pairs_in_fit": c["pairs_in_fit"], "fallback_rule": c["fallback_rule"],
                      "fallback_ce_on_unseen_inner_rows": c["fallback_ce_on_unseen_inner_rows"], "_seen": c["seen"]}
        else:
            Pc, c = cc_local(toks[w], y, fit_idx, pred_idx)
            cov[w] = {"tokens_in_fit": c["tokens_in_fit"], "_seen": c["seen"]}
        for name in Pc:
            preds[f"{prefix}{w}:{name}"] = Pc[name]
            origin[f"{prefix}{w}:{name}"] = ("cc", w, name)
        secs[w]["CC"] = round(time.time() - t0, 3)
    return preds, origin, cov, secs


def _bank_lists(prefixes):
    """Ordered candidate lists per target view: [(candidate label, view, pred-key prefix)]."""
    out = {w: [] for w in PRIMARY_VIEWS}
    for px in prefixes:
        tag = px[:-1] + ":" if px else ""
        for w in ("v1", "v2"):
            out[w].append((f"{tag}own", w, px))
        out["pair"].append((f"{tag}coalition", "pair", px))
        for lw, nm in LOCAL_IGNORE:
            out["pair"].append((f"{tag}{nm}", lw, px))
    return out


def _attackers_of(preds, px, view):
    pre = f"{px}{view}:"
    return [k[len(pre):] for k in preds if k.startswith(pre) and ":" not in k[len(pre):]]


def _select(preds, prefixes, ys, sel_pos):
    """Dual selection per target view over the ordered banks. Returns selection dict and per-view bank rows."""
    lists = _bank_lists(prefixes)
    sel, banks = {}, {}
    for w in PRIMARY_VIEWS:
        rows = []
        for cand, view, px in lists[w]:
            for att in _attackers_of(preds, px, view):
                p = preds[f"{px}{view}:{att}"][sel_pos]
                rows.append({"candidate": cand, "view": view, "attacker": att, "pred_key": f"{px}{view}:{att}",
                             "inner_auc": auc1(ys, p), "inner_ce": ce1(ys, p)})
        ia, ic = _pick(rows, "inner_auc", True), _pick(rows, "inner_ce", False)
        sel[w] = {"auc": {**rows[ia], "bank_index": ia, "label": _label(rows[ia])},
                  "ce": {**rows[ic], "bank_index": ic, "label": _label(rows[ic])}}
        banks[w] = rows
    return sel, banks


def _label(r):
    return f"{r['candidate']}:{r['view']}:{r['attacker']}"


def _coverage_summary(cov, sel_pos, score_pos=None):
    out = {}
    for w, c in cov.items():
        seen = c["_seen"]
        d = {k: v for k, v in c.items() if k != "_seen"}
        d["inner_rows"] = int(len(sel_pos))
        d["inner_rows_seen_in_fit"] = int(seen[sel_pos].sum())
        d["inner_rows_fallback"] = int((~seen[sel_pos]).sum())
        if score_pos is not None:
            d["scored_rows"] = int(len(score_pos))
            d["scored_rows_seen_in_fit"] = int(seen[score_pos].sum())
            d["scored_rows_fallback"] = int((~seen[score_pos]).sum())
        if w == "pair":
            d["fallback_counts_by_alpha"] = {
                a: {"rule": r, "inner_rows": d["inner_rows_fallback"],
                    **({"scored_rows": d["scored_rows_fallback"]} if score_pos is not None else {})}
                for a, r in c["fallback_rule"].items()}
        else:
            d["fallback_rule"] = "AUDIT_FIT SEX prior"
        out[w] = d
    return out


def _is_finite(views, finite):
    has = bool(views.get("tokens"))
    if finite is None:
        return has
    if finite and not has:
        raise ValueError("finite=True but the views declare no token identity")
    return bool(finite)


# ------------------------------------------------------------------ INNER audit
def inner_audit(views, D, finite=None, slate="final", y=None, composed=(), memo=None):
    """Inner audit of one frozen release: fit on AUDIT_FIT, select and score on INNER_SELECTION only.

    views: dict from policy_views / source_views / dpc.baselines.reference_view_sets ({"X": {v1, v2, pair}, "tokens":
    {v1, v2} | None, "family", "meta"}). finite: None -> finite iff the views declare tokens. composed: optional list
    of (unit name, policy views) whose code readers join this (source) bank by refitting (normally use
    composed_inner, which reuses the policy inner predictions). Returns the record described in the module docstring;
    its "inner_predictions" and "sel_row_id" are private arrays (see public_record)."""
    t0 = time.time()
    yy = np.asarray(D["sex"] if y is None else y)
    guard_inner(D, yy)
    fit_idx, sel_idx = roles(D)
    sel_pos = np.arange(len(sel_idx))
    fin = _is_finite(views, finite)
    preds, origin, cov, secs = _source_bank(views, yy, fit_idx, sel_idx, sel_pos, slate, 0, fin, memo)
    prefixes = [""]
    comp_cov = {}
    for uname, cv in composed:
        px = f"composed[{uname}]:"
        p2, o2, c2, s2 = _source_bank(cv, yy, fit_idx, sel_idx, sel_pos, slate, 0, bool(cv.get("tokens")), memo,
                                      prefix=px)
        preds.update(p2)
        origin.update(o2)
        comp_cov[uname] = _coverage_summary(c2, sel_pos)
        prefixes.append(px)
    ys = yy[sel_idx]
    sel, banks = _select(preds, prefixes, ys, sel_pos)
    rec = _summarise(sel, banks)
    keys = list(preds)
    rec.update({
        "kind": "inner", "family": views.get("family"), "meta": views.get("meta", {}), "finite": fin,
        "slate": slate, "slate_members": [nm for nm, _ in SLATES[slate]()],
        "cc_alphas": list(CC_ALPHAS) if fin else [], "fallback_rules": list(FALLBACK_RULES) if fin else [],
        "roles": {"fit": FIT_ROLE, "select_and_score": SEL_ROLE},
        "statistic": "selected-attacker (bank maximum) INNER_SELECTION AUC; selection-optimistic, identical for every "
                     "release; not a null test",
        "orientation": "P(S=1), fixed; never flipped or clamped",
        "coverage": _coverage_summary(cov, sel_pos), "composed_coverage": comp_cov,
        "composed_units": [u for u, _ in composed],
        "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)),
        "sel_row_id_sha256": sha_arrays(np.asarray(D["row_id"])[sel_idx]),
        "fit_row_id_sha256": sha_arrays(np.asarray(D["row_id"])[fit_idx]),
        "fit_seconds": secs,
        "inner_predictions": {"keys": keys, "P": np.stack([preds[k] for k in keys]) if keys else np.zeros((0, 0))},
        "sel_row_id": np.asarray(D["row_id"])[sel_idx],
        "wall_s": round(time.time() - t0, 2)})
    return rec


def _summarise(sel, banks):
    out = {"auc": {}, "ce": {}, "selected": {}, "ce_selected": {}, "selection": {}, "tables": {}}
    for w in PRIMARY_VIEWS:
        out["auc"][w], out["ce"][w] = sel[w]["auc"]["inner_auc"], sel[w]["ce"]["inner_ce"]
        out["selected"][w], out["ce_selected"][w] = sel[w]["auc"]["label"], sel[w]["ce"]["label"]
        out["selection"][w] = {"auc": sel[w]["auc"], "ce": sel[w]["ce"]}
        out["tables"][w] = banks[w]
    out["worse_local"] = max(out["auc"]["v1"], out["auc"]["v2"])
    out["mean_local"] = (out["auc"]["v1"] + out["auc"]["v2"]) / 2
    out["coalition_minus_best_local"] = out["auc"]["pair"] - out["worse_local"]
    return out


def public_record(rec):
    """JSON-safe copy: private arrays replaced by their sha256 (predictions stay in the private unit)."""
    out = {k: v for k, v in rec.items() if k not in ("inner_predictions", "sel_row_id")}
    ip = rec.get("inner_predictions")
    if ip is not None:
        out["inner_predictions_sha256"] = sha_arrays(ip["P"])
        out["inner_prediction_keys"] = list(ip["keys"])
    return _jsonable(out)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


# ------------------------------------------------------------------ composed source readers (reuse, no refit)
def _class_only(meta):
    m = meta.get("m")
    if m is None and meta.get("config"):
        parts = str(meta["config"]).split("|")
        m = next((int(p[1:]) for p in parts if p.startswith("m") and p[1:].isdigit()), None)
    return m == 1


def composed_inner(source_inner, policy_inners):
    """Composed source reader: per view the max inner AUC (min CE) over the source's own bank and every policy's code
    bank, REUSING the stored INNER_SELECTION predictions (no refit). policy_inners: list of policy inner records
    (with predictions) of the SAME teacher and seed; for the decisions family only class-only (m = 1) policies."""
    sm = source_inner.get("meta", {})
    fam = source_inner.get("family")
    ys = None
    for r in [source_inner] + list(policy_inners):
        if r["sel_row_id_sha256"] != source_inner["sel_row_id_sha256"]:
            raise ValueError("REFUSED: inner records were scored on different INNER_SELECTION rows")
        if r.get("fit_row_id_sha256") != source_inner.get("fit_row_id_sha256"):
            raise ValueError("REFUSED: inner records were fitted on different AUDIT_FIT rows")
    for p in policy_inners:
        pm = p.get("meta", {})
        for k in ("teacher", "seed"):
            if sm.get(k) is None or pm.get(k) is None or str(sm[k]) != str(pm[k]):
                raise ValueError(f"REFUSED: composed policy {pm.get('unit')} has {k}={pm.get(k)!r}, source has "
                                 f"{sm.get(k)!r} (only the same teacher and seed compose)")
        if p.get("family") != "code":
            raise ValueError("REFUSED: only policy code-view records compose into a source bank")
        if fam == "decisions" and not _class_only(pm):
            raise ValueError(f"REFUSED: {pm.get('unit')} is not a function of the decision; it cannot compose into "
                             "the decisions-only source family")
    if "labels_sha256" in source_inner:
        ys = source_inner["labels_sha256"]
    out = {"auc": {}, "ce": {}, "selected": {}, "ce_selected": {}, "winner_is_composed": {},
           "ce_winner_is_composed": {},
           "own_auc": dict(source_inner["auc"]), "own_ce": dict(source_inner["ce"]), "n_candidates": {},
           "best_per_policy_auc": {}, "policies": [p["meta"].get("unit") for p in policy_inners],
           "source": {"family": fam, **{k: sm.get(k) for k in ("teacher", "seed", "unit", "label")}},
           "reuse": "policy candidates enter with their stored INNER_SELECTION predictions (the composed attacker "
                    "A(g(p)) equals the policy attacker A on the same rows, fitted on the same AUDIT_FIT views); "
                    "no refit; selection values recomputed from the stored predictions",
           "labels_sha256": ys}
    y_sel = source_inner.get("_y_sel")
    for w in PRIMARY_VIEWS:
        rows = [dict(r, origin="source") for r in source_inner["tables"][w]]
        for p in policy_inners:
            u = p["meta"].get("unit")
            prow = [dict(r, candidate=f"composed[{u}]:{r['candidate']}", origin=u) for r in p["tables"][w]]
            if y_sel is not None and p.get("inner_predictions") is not None:   # recompute from stored predictions
                P = dict(zip(p["inner_predictions"]["keys"], p["inner_predictions"]["P"]))
                for r in prow:
                    r["inner_auc"], r["inner_ce"] = auc1(y_sel, P[r["pred_key"]]), ce1(y_sel, P[r["pred_key"]])
            if prow:
                b = _pick(prow, "inner_auc", True)
                out["best_per_policy_auc"].setdefault(w, {})[u] = {"auc": prow[b]["inner_auc"],
                                                                   "label": _label(prow[b])}
            rows += prow
        ia, ic = _pick(rows, "inner_auc", True), _pick(rows, "inner_ce", False)
        out["auc"][w], out["ce"][w] = rows[ia]["inner_auc"], rows[ic]["inner_ce"]
        out["selected"][w], out["ce_selected"][w] = _label(rows[ia]), _label(rows[ic])
        out["winner_is_composed"][w] = rows[ia]["origin"] != "source"
        out["ce_winner_is_composed"][w] = rows[ic]["origin"] != "source"
        out["n_candidates"][w] = len(rows)
    out["worse_local"] = max(out["auc"]["v1"], out["auc"]["v2"])
    out["mean_local"] = (out["auc"]["v1"] + out["auc"]["v2"]) / 2
    return out


def attach_labels(rec, D):
    """Attach INNER_SELECTION SEX (private, in memory) so composed_inner recomputes from stored predictions."""
    guard_inner(D)
    _, s = roles(D)
    if sha_arrays(np.asarray(D["row_id"])[s]) != rec["sel_row_id_sha256"]:
        raise ValueError("REFUSED: D's INNER_SELECTION rows differ from the record's")
    rec["_y_sel"] = np.asarray(D["sex"])[s]
    return rec


# ------------------------------------------------------------------ FINAL audit (dpc.assess passes assessment rows)
def final_audit(views, D, score_rows, finite=None, slate="final", composed=(), seeds=ATT_SEEDS, y=None, memo=None):
    """Fit on AUDIT_FIT, dual-select on INNER_SELECTION (whole bank incl. composed policy code readers), refit each
    selected attacker at attacker seeds, score on score_rows (disjoint from both, unsealed). composed: list of
    (policy unit, policy views) of the locked selected policies of the same teacher/seed (source releases only).
    Returns (record, probs) with probs[f"{crit}_{w}"] = (len(seeds), len(score_rows), 2) [P(S=0), P(S=1)]."""
    t0 = time.time()
    yy = np.asarray(D["sex"] if y is None else y)
    guard_inner(D, yy)
    fit_idx, sel_idx = roles(D)
    score_rows = _check_score_rows(D, score_rows, yy)
    pred_idx = np.concatenate([sel_idx, score_rows])
    sel_pos, score_pos = np.arange(len(sel_idx)), np.arange(len(sel_idx), len(pred_idx))
    memo = {} if memo is None else memo
    fin = _is_finite(views, finite)
    sets = [("", views, fin)] + [(f"composed[{u}]:", cv, bool(cv.get("tokens"))) for u, cv in composed]
    preds, origin, covs, rules, vmap = {}, {}, {}, {}, {}
    for px, v, f in sets:
        p, o, c, _ = _source_bank(v, yy, fit_idx, pred_idx, sel_pos, slate, seeds[0], f, memo, prefix=px)
        preds.update(p)
        origin.update(o)
        covs[px or "own"] = _coverage_summary(c, sel_pos, score_pos)
        rules[px] = (c.get("pair") or {}).get("fallback_rule")
        vmap[px] = (v, f)
    ys = yy[sel_idx]
    sel, banks = _select(preds, [px for px, _, _ in sets], ys, sel_pos)
    probs, scored = {}, {}
    y_sc = yy[score_rows]
    for w in PRIMARY_VIEWS:
        scored[w] = {}
        for crit in ("auc", "ce"):
            s = sel[w][crit]
            key = s["pred_key"]
            px = key[:key.index("]:") + 2] if key.startswith("composed[") else ""
            kind, view, att = origin[key]
            Ps = []
            for sd in seeds:
                if sd == seeds[0]:
                    p1 = preds[key][score_pos]
                elif kind == "cc":       # cell readers are deterministic counts: the seed refit is identical
                    p1 = preds[key][score_pos]
                else:
                    v, _ = vmap[px]
                    X = np.asarray(v["X"][view], dtype=np.float64)
                    fac = dict(SLATES[slate]())[att]
                    mk = (_content_hash(X, yy, fit_idx, pred_idx), att, sd)
                    if mk not in memo:
                        m = fac(sd).fit(X[fit_idx], yy[fit_idx])
                        memo[mk] = _p1(m, X[pred_idx])
                    p1 = memo[mk][score_pos]
                Ps.append(np.stack([1.0 - p1, p1], 1))
            P = np.stack(Ps)
            probs[f"{crit}_{w}"] = P
            scored[w][crit] = {"label": s["label"], "candidate": s["candidate"], "source_view": view, "attacker": att,
                               "inner_auc": s["inner_auc"], "inner_ce": s["inner_ce"], "seeds": list(seeds),
                               "auc_per_seed": [auc1(y_sc, p[:, 1]) for p in P],
                               "ce_per_seed": [ce1(y_sc, p[:, 1]) for p in P]}
            scored[w][crit]["auc_mean"] = float(np.mean(scored[w][crit]["auc_per_seed"]))
            scored[w][crit]["ce_mean"] = float(np.mean(scored[w][crit]["ce_per_seed"]))
    rec = _summarise(sel, banks)
    rec.update({"kind": "final", "family": views.get("family"), "meta": views.get("meta", {}), "finite": fin,
                "slate": slate, "slate_members": [nm for nm, _ in SLATES[slate]()], "cc_alphas": list(CC_ALPHAS),
                "composed_units": [u for u, _ in composed], "coverage": covs, "scored": scored,
                "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)), "n_score": int(len(score_rows)),
                "orientation": "P(S=1) column, fixed; never flipped or clamped on scored rows",
                "reuse": "identical (view content, labels, rows, attacker, seed) fits memoised within the process",
                "wall_s": round(time.time() - t0, 2)})
    return rec, probs


# ------------------------------------------------------------------ inner units (lead's dpc.run contract)
def udir(name, units_dir=None):
    return Path(units_dir or UNITS) / name


def _fn():
    from rgj import finalize as FN
    return FN


def _load_npz(unit, fname, units_dir=None):
    FN = _fn()
    d = udir(unit, units_dir)
    if not FN.unit_complete(d):
        raise SystemExit(f"REFUSED: {unit} is missing or not hash-complete")
    z = np.load(d / fname, allow_pickle=False)
    return {k: z[k] for k in z.files}, hashlib.sha256((d / "COMPLETE.json").read_bytes()).hexdigest()


def parse_cid(cid):
    """Lead's config ids: "<teacher>|<FAMILY>|m<m>[|l<lam>]", "SRC|<teacher>", "REF|<label>"."""
    p = str(cid).split("|")
    if p[0] == "SRC":
        return {"kind": "source", "teacher": p[1]}
    if p[0] == "REF":
        return {"kind": "reference", "label": p[1]}
    return {"kind": "policy", "teacher": p[0], "family": p[1], "m": int(p[2][1:]),
            "lam": float(p[3][1:]) if len(p) > 3 else None}


def unit_of(k, cid):
    c = parse_cid(cid)
    if c["kind"] == "source":
        return f"tea__s{k}__{c['teacher']}"
    if c["kind"] == "reference":
        return f"ref__s{k}__{c['label']}"
    return f"pol__s{k}__{str(cid).replace('|', '_')}"


def load_teacher(teacher, k, units_dir=None):
    t, sha = _load_npz(f"tea__s{k}__{teacher}", "teacher.npz", units_dir)
    return t, sha


def view_sets(kind, k, cid, D, units_dir=None):
    """(primary family, {family: views}, release outputs {p1, p2, hard1, hard2}, provenance) of one release.
    policy: code (primary) | source: interface [c_i, p_i] (primary), complete, scores, probs, decisions |
    reference: dpc.baselines.reference_view_sets (interface primary)."""
    c = parse_cid(cid)
    unit = unit_of(k, cid)
    if kind != c["kind"]:
        raise ValueError(f"kind {kind!r} does not match config id {cid!r}")
    if kind == "policy":
        z, sha = _load_npz(unit, "release.npz", units_dir)
        meta = {"kind": "policy", "teacher": c["teacher"], "seed": k, "unit": unit, "config": cid, "m": c["m"],
                "family_method": c["family"]}
        V = policy_views(z, D, meta=meta)
        out = {"p1": z["q1"], "p2": z["q2"], "hard1": z["hard1"], "hard2": z["hard2"]}
        return "code", {"code": V}, out, {"unit": unit, "complete_sha256": sha, "z": z}
    if kind == "source":
        t, sha = load_teacher(c["teacher"], k, units_dir)
        sets = {}
        for fam in SOURCE_FAMILIES:
            sets[fam] = source_views(t, D, fam, meta={"kind": "source", "teacher": c["teacher"], "seed": k,
                                                      "unit": unit, "label": cid})
        out = {"p1": t["p1"], "p2": t["p2"], "hard1": t["d1"], "hard2": t["d2"]}
        return "interface", sets, out, {"unit": unit, "complete_sha256": sha}
    from dpc import baselines as BL
    sets, out, prov = BL.reference_view_sets(c["label"], k, D, units_dir=units_dir, with_outputs=True)
    return "interface", sets, out, {"unit": unit, **prov}


def inner_unit(kind, k, cid, D, units_dir=None, slate="final"):
    """ONE inner audit unit for dpc.run.stage_inner (the lead saves it as inner__<unit> with save_unit).

    Returns a JSON-safe dict: recovery (primary family: auc/ce/selected/ce_selected per view, tables, coverage, ...),
    families {name: same structure} (secondary families), utility {"0", "1"} on INNER_SELECTION (dpc.utility),
    class_preservation_ok + token_states (policies), provenance; and "_files" = {"inner_preds.npz": writer} holding
    every candidate's INNER_SELECTION predictions per family (keys_<fam>, P_<fam>, sel_row_id) for composed readers
    and the verifier. Fits AUDIT_FIT, selects and scores INNER_SELECTION only."""
    from dpc import utility as UTL
    t0 = time.time()
    guard_inner(D)
    primary, sets, out, prov = view_sets(kind, k, cid, D, units_dir)
    recs = {fam: inner_audit(V, D, slate=slate) for fam, V in sets.items()}
    arrays = {"sel_row_id": recs[primary]["sel_row_id"]}
    for fam, r in recs.items():
        arrays[f"keys_{fam}"] = np.asarray(r["inner_predictions"]["keys"])
        arrays[f"P_{fam}"] = r["inner_predictions"]["P"]
    res = {"kind": kind, "cid": cid, "seed": k, "unit_of": prov["unit"],
           "of_complete_sha256": prov.get("complete_sha256"),
           "primary_family": primary, "recovery": public_record(recs[primary]),
           "families": {fam: public_record(r) for fam, r in recs.items() if fam != primary},
           "utility": UTL.inner_release_utility(out, D), "slate": slate,
           "private_file": "inner_preds.npz (keys_<family>, P_<family> (n_candidates, n_INNER) P(S=1), sel_row_id)"}
    if kind == "policy":
        c = parse_cid(cid)
        t, tsha = load_teacher(c["teacher"], k, units_dir)
        z = prov["z"]
        cp = {f"recipient_{i}": bool(np.array_equal(np.asarray(z[f"hard{i}"]), np.asarray(t[f"d{i}"])))
              for i in (1, 2)}
        fit = np.asarray(D["idx"]["OSF_DEFENSE_FIT"] if "OSF_DEFENSE_FIT" in D["idx"] else D["idx"]["DEFENSE_FIT"])
        st = {f"recipient_{i}": {"fit_rows": int(len(np.unique(z[f"tok{i}"][fit]))),
                                 "all_rows": int(len(np.unique(z[f"tok{i}"]))),
                                 "alphabet": int(np.asarray(z[f"alpha{i}"]).ravel()[0])} for i in (1, 2)}
        res.update({"class_preservation_ok": all(cp.values()), "class_preservation": cp,
                    "class_preservation_rule": "hard_i == teacher d_i on ALL D rows (both recipients)",
                    "teacher_unit_complete_sha256": tsha,
                    "token_states": int(sum(v["fit_rows"] for v in st.values())),
                    "token_states_rule": "distinct tokens on OSF_DEFENSE_FIT rows, summed over both recipients",
                    "token_states_detail": st, "view_fingerprint": view_fingerprint(sets["code"])})
    res["wall_s"] = round(time.time() - t0, 2)
    res = _jsonable(res)
    res["_files"] = {"inner_preds.npz": lambda p, a=arrays: np.savez_compressed(p, **a)}
    return res


def load_inner(unit, units_dir=None):
    """{family: inner record with its private predictions re-attached} of inner__<release unit> (hash-checked)."""
    FN = _fn()
    name = unit if unit.startswith("inner__") else f"inner__{unit}"
    d = udir(name, units_dir)
    if not FN.unit_complete(d):
        raise SystemExit(f"REFUSED: {name} is missing or not hash-complete")
    rec = json.loads((d / "record.json").read_text())
    z = np.load(d / "inner_preds.npz", allow_pickle=False)
    out = {}
    fams = {rec["primary_family"]: rec["recovery"], **rec.get("families", {})}
    for fam, r in fams.items():
        r = dict(r)
        r["inner_predictions"] = {"keys": [str(x) for x in z[f"keys_{fam}"]], "P": z[f"P_{fam}"]}
        r["sel_row_id"] = z["sel_row_id"]
        if sha_arrays(r["inner_predictions"]["P"]) != r["inner_predictions_sha256"]:
            raise RuntimeError(f"{name}/{fam}: stored predictions differ from the recorded hash")
        out[fam] = r
    return out


def composed_from_units(k, teacher, policy_cids, D, family="interface", units_dir=None):
    """composed_inner for source SRC|<teacher> seed k, family `family`, over the inner units of `policy_cids`
    (same teacher and seed; for the decisions family only class-only policies are admitted)."""
    src = attach_labels(load_inner(unit_of(k, f"SRC|{teacher}"), units_dir)[family], D)
    pols = []
    for c in policy_cids:
        pc = parse_cid(c)
        if pc["teacher"] != teacher:
            raise ValueError(f"REFUSED: {c} is not a policy of teacher {teacher}")
        if family == "decisions" and pc["m"] != 1:
            continue
        pols.append(load_inner(unit_of(k, c), units_dir)["code"])
    return composed_inner(src, pols)


def view_fingerprint(views):
    """Renumbering-invariant content hash of a view set (identical hash -> identical attacks; alias reuse)."""
    return sha_arrays(*[np.asarray(views["X"][w]) for w in PRIMARY_VIEWS])



# ------------------------------------------------------------------ real-data controls (lead runs; inner roles only)
def _noisy(S, seed):
    return AU.noisy_sex(np.asarray(S), seed)


def split_tokens(z, i, bit, collide=True, eta=CONF_ETA):
    """Release copy whose recipient-i token t becomes 2 t + bit (alphabet doubles). collide=True: both copies decode
    to the same probabilities (decoder collision); False: the bit = 1 copy decodes to (1 - eta) q + eta e_d (more
    confident, same decision). Decisions unchanged."""
    zp = {k: np.array(v) for k, v in z.items()}
    t = zp[f"tok{i}"].astype(np.int64)
    a = int(np.asarray(zp[f"alpha{i}"]).ravel()[0])
    b = np.asarray(bit, dtype=np.int64)
    # AMENDMENT_A1 (lead, 2026-10-06): the planted bit is defined only on rows with a valid 0/1 label (AUDIT_FIT /
    # INNER_SELECTION); sealed assessment rows carry -1 labels, which produced bits outside {0, 1} and colliding split
    # tokens (refused by token_decoder_check). Other rows get bit 0; plants are only fitted/scored on inner roles.
    b = np.where((b == 0) | (b == 1), b, 0)
    zp[f"tok{i}"] = 2 * t + b
    zp[f"alpha{i}"] = np.asarray(2 * a)
    if not collide:
        q = zp[f"q{i}"].astype(np.float64)
        e = onehot(zp[f"hard{i}"], q.shape[1])
        zp[f"q{i}"] = np.where(b[:, None] == 1, (1 - eta) * q + eta * e, q)
    return zp


def _roundtrip(zp, tmp, tag):
    p = Path(tmp) / f"{tag}.npz"
    np.savez_compressed(p, **zp)
    z2 = np.load(p)
    exact = all(np.array_equal(z2[k], zp[k]) for k in zp)
    return {k: z2[k] for k in z2.files}, exact


def _split_audit(views, Sp, D, halves, slate, finite=None):
    """Fit AUDIT_FIT, select on half A (whole bank), evaluate on held-out half B, per view."""
    fit_idx, sel_idx = roles(D)
    a, b = halves
    fin = _is_finite(views, finite)
    preds, _, cov, _ = _source_bank(views, Sp, fit_idx, sel_idx, a, slate, 0, fin, None)
    ya, yb = Sp[sel_idx][a], Sp[sel_idx][b]
    selA, banks = _select(preds, [""], ya, a)
    out = {}
    for w in PRIMARY_VIEWS:
        s = selA[w]["auc"]
        out[w] = {"selected_on_A": s["label"], "max_auc_A": s["inner_auc"],
                  "heldout_auc_B": auc1(yb, preds[s["pred_key"]][b])}
    out["coverage"] = _coverage_summary(cov, np.arange(len(sel_idx)))
    return out


def controls_for_release(label, views, D, z_policy=None, slate="final", workdir=None):
    """Null (+ plants when z_policy is a policy release dict) on one release; inner roles, permuted labels."""
    guard_inner(D)
    S = np.asarray(D["sex"])
    halves = AU.null_split(D)
    Sp, perm_sha = AU.frozen_permutation(S, D, halves)
    thr, sd = AU.null_threshold(Sp, D, halves)
    res = {"label": label, "family": views.get("family"), "permutation_sha256": perm_sha,
           "null_threshold": thr, "null_sd_B": sd, "plant_min": PLANT_MIN, "checks": {}}
    t0 = time.time()
    nul = _split_audit(views, Sp, D, halves, slate)
    res["checks"]["null"] = {w: {**nul[w], "ok": nul[w]["heldout_auc_B"] <= thr} for w in PRIMARY_VIEWS}
    if z_policy is not None:
        zz = align(z_policy, D)
        noisy = _noisy(Sp, CONTROL_SEED)
        rng = np.random.default_rng(CONTROL_SEED + 7)
        b1 = rng.integers(0, 2, len(noisy))
        plants = {"CONF_r1": ([(1, noisy, False)], "v1"), "CONF_r2": ([(2, noisy, False)], "v2"),
                  "COLL_r1": ([(1, noisy, True)], "v1"), "COLL_r2": ([(2, noisy, True)], "v2"),
                  "XOR": ([(1, b1, True), (2, b1 ^ noisy, True)], "pair")}
        with tempfile.TemporaryDirectory(dir=workdir) as tmp:
            for pname, (ops, target) in plants.items():
                zp = dict(zz)
                for i, bit, coll in ops:
                    zp = split_tokens(zp, i, bit, collide=coll)
                z2, exact = _roundtrip(zp, tmp, pname)
                V = policy_views(z2, D, meta={"plant": pname})
                r = _split_audit(V, Sp, D, halves, slate)
                entry = {"target_view": target, "serialisation_exact": exact,
                         **{w: r[w] for w in PRIMARY_VIEWS}, "coverage": r["coverage"]}
                if pname.startswith("CONF") or pname == "XOR":
                    dv = decision_views(z2["hard1"], z2["hard2"], D)
                    rd = _split_audit(dv, Sp, D, halves, slate)
                    entry["decisions_only_audit"] = {w: rd[w] for w in PRIMARY_VIEWS}
                    entry["decisions_only_misses_it"] = rd[target]["heldout_auc_B"] <= thr
                if pname.startswith("COLL"):
                    qv = {"family": "decoded_q_only", "tokens": None,
                          "X": {"v1": z2["q1"], "v2": z2["q2"], "pair": np.hstack([z2["q1"], z2["q2"]])}}
                    rq = _split_audit(qv, Sp, D, halves, slate)
                    entry["decoded_probability_only_audit"] = {w: rq[w] for w in PRIMARY_VIEWS}
                if pname == "XOR":
                    entry["locals_null_ok"] = all(r[w]["heldout_auc_B"] <= thr for w in ("v1", "v2"))
                entry["ok"] = bool(exact and r[target]["heldout_auc_B"] > PLANT_MIN)
                res["checks"][pname] = entry
    res["failures"] = [f"null:{w}" for w in PRIMARY_VIEWS if not res["checks"]["null"][w]["ok"]] + \
                      [p for p, e in res["checks"].items() if p != "null" and not e["ok"]]
    res["all_ok"] = not res["failures"]
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def null_calibration(views, D, reps=5, slate="final", seed0=CONTROL_SEED + 100):
    """Repeated shuffled-label nulls on one real release (select half A, evaluate half B); every rep kept."""
    guard_inner(D)
    halves = AU.null_split(D)
    rows = []
    for k in range(reps):
        Sp, sha = AU.frozen_permutation(np.asarray(D["sex"]), D, halves, seed=seed0 + k)
        thr, sd = AU.null_threshold(Sp, D, halves)
        r = _split_audit(views, Sp, D, halves, slate)
        for w in PRIMARY_VIEWS:
            rows.append({"rep": k, "view": w, "heldout_auc_B": r[w]["heldout_auc_B"], "max_auc_A": r[w]["max_auc_A"],
                         "selected_on_A": r[w]["selected_on_A"], "threshold": thr, "sd0": sd,
                         "exceeds": r[w]["heldout_auc_B"] > thr, "permutation_sha256": sha})
    hb = np.array([x["heldout_auc_B"] for x in rows])
    zz = (hb - 0.5) / np.array([x["sd0"] for x in rows])
    return {"summary": {"reps": reps, "tests": len(rows), "exceedances": int(sum(x["exceeds"] for x in rows)),
                        "heldout_mean": float(hb.mean()), "heldout_max": float(hb.max()), "z_mean": float(zz.mean()),
                        "z_sd": float(zz.std(ddof=1)) if len(zz) > 1 else None, "null_z": NULL_Z,
                        "max_auc_A_mean_descriptive": float(np.mean([x["max_auc_A"] for x in rows]))},
            "rows": rows}


PRELOCK_CODE = ("dpc/audit.py", "dpc/utility.py", "dpc/baselines.py", "smf/audit.py", "jcv/audit.py",
                "osf/audit.py", "rgj/finalize.py")
# Fixed control plan (registered before any real-data control runs; chosen by structure, not by results): the largest
# joint code of each teacher (most states, sparsest pair table) and the class-only code of U receive the null and every
# plant; both teachers' five source families and every reference family receive the null; the multi-permutation null
# calibration uses the first policy.
CONTROL_PLAN = {"policies": [(0, "U|JOINT|m8|l1"), (0, "RAW-J_b0.3|JOINT|m8|l1"), (0, "U|CLASS|m1")],
                "sources": [("U", 0), ("RAW-J_b0.3", 0)], "references": [("E", 0), ("F", 0), ("F0", 0)],
                "null_policy": (0, "U|JOINT|m8|l1"), "null_reps": 5}


def run_prelock_controls(D, policies=None, sources=None, references=None, null_policy=None, null_reps=None,
                         units_dir=None, out_path=None, slate="final", workdir=None):
    """STAGE FUNCTION (the LEAD runs it after SELECTION_AND_AUDIT_LOCK; sealed D; inner roles only).

    policies: [(seed, policy cid)] receiving the null AND every plant (CONF_r1/r2, COLL_r1/r2, XOR).
    sources: [(teacher, seed)] whose five source families receive the null.
    references: [(label, seed)] whose score families receive the null (dpc.baselines.reference_view_sets).
    null_policy: (seed, cid) for the multi-permutation null calibration (null_reps permutations).
    Defaults: CONTROL_PLAN. Writes AUDIT_PRELOCK_CHECKS.json (public; refuses private paths) when out_path is given."""
    import platform
    import resource
    plan = CONTROL_PLAN
    policies = plan["policies"] if policies is None else policies
    sources = plan["sources"] if sources is None else sources
    references = plan["references"] if references is None else references
    null_policy = plan["null_policy"] if null_policy is None else null_policy
    null_reps = plan["null_reps"] if null_reps is None else null_reps
    if not D.get("sealed", False):
        raise SystemExit("REFUSED: pre-lock controls run on the sealed D only")
    guard_inner(D)
    wt = Path(__file__).resolve().parents[1]
    code = {f: hashlib.sha256((wt / f).read_bytes()).hexdigest() for f in PRELOCK_CODE if (wt / f).exists()}
    t0, c0 = time.time(), time.process_time()
    out = {"study": "pcrl_decision_preserving_compression_v1", "stage": "controls (inner roles only; sealed D)",
           "roles": {"fit": FIT_ROLE, "select_half_A_evaluate_half_B": SEL_ROLE, "assessment": "never indexed"},
           "slate": slate, "slate_members": [nm for nm, _ in SLATES[slate]()], "cc_alphas": list(CC_ALPHAS),
           "plan": {"policies": [list(x) for x in policies], "sources": [list(x) for x in sources],
                    "references": [list(x) for x in references], "null_policy": list(null_policy) if null_policy
                    else None, "null_reps": null_reps},
           "label": "S* = SEX permuted within AUDIT_FIT, half A and half B (smf.audit.frozen_permutation, "
                    "CONTROL_SEED); plants encode noisy S* (20% random)",
           "thresholds": {"null": "AUC_B <= 0.5 + 3.5 sd0", "plant_min": PLANT_MIN, "null_z": NULL_Z},
           "split": AU._split_receipt(D, AU.null_split(D)), "releases": {}, "code_sha256_at_start": code}
    for k, cid in policies:
        u = unit_of(k, cid)
        z, sha = _load_npz(u, "release.npz", units_dir)
        c = parse_cid(cid)
        V = policy_views(z, D, meta={"kind": "policy", "teacher": c["teacher"], "seed": k, "unit": u, "config": cid})
        out["releases"][f"{cid}|s{k}"] = {"unit": u, "complete_sha256": sha,
                                          **controls_for_release(cid, V, D, z_policy=z, slate=slate, workdir=workdir)}
    for teacher, k in sources:
        t, sha = load_teacher(teacher, k, units_dir)
        for fam in SOURCE_FAMILIES:
            lab = f"SRC|{teacher}|s{k}|{fam}"
            out["releases"][lab] = {"complete_sha256": sha,
                                    **controls_for_release(lab, source_views(t, D, fam), D, slate=slate)}
    if references:
        from dpc import baselines as BL
        for lab_, k in references:
            sets, _, prov = BL.reference_view_sets(lab_, k, D, units_dir=units_dir, with_outputs=True)
            for fam, V in sets.items():
                lab = f"REF|{lab_}|s{k}|{fam}"
                out["releases"][lab] = {"complete_sha256": prov.get("complete_sha256"),
                                        **controls_for_release(lab, V, D, slate=slate)}
    if null_policy:
        k, cid = null_policy
        z, _ = _load_npz(unit_of(k, cid), "release.npz", units_dir)
        out["null_calibration"] = {"release": f"{cid}|s{k}", **null_calibration(policy_views(z, D), D, null_reps,
                                                                                 slate)}
    out["failures"] = [f"{lab}:{f}" for lab, r in out["releases"].items() for f in r["failures"]]
    calib_exc = out.get("null_calibration", {}).get("summary", {}).get("exceedances", 0)
    out["verdict"] = {"all_ok": not out["failures"] and not calib_exc, "failures": out["failures"],
                      "null_calibration_exceedances": calib_exc,
                      "plants_detected": {lab: {p: e["ok"] for p, e in r["checks"].items() if p != "null"}
                                          for lab, r in out["releases"].items() if len(r["checks"]) > 1},
                      "decisions_only_audit_misses_confidence_plant": {
                          lab: {p: e.get("decisions_only_misses_it") for p, e in r["checks"].items()
                                if p.startswith("CONF")} for lab, r in out["releases"].items() if len(r["checks"]) > 1}}
    out["compute"] = {"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "processes": 1,
                      "omp_threads": 1, "python": platform.python_version()}
    if out_path:
        txt = json.dumps(_jsonable(out), indent=1)
        for bad in (str(HOME), "PCRL_eval_cache_private"):
            if bad in txt:
                raise SystemExit("REFUSED: a private path would enter the public pre-lock receipt")
        Path(out_path).write_text(txt + "\n")
    return out


def stage_controls(D, shard_spec=None):
    """dpc.run-compatible entry (LATE "controls"): CONTROL_PLAN, writes the public AUDIT_PRELOCK_CHECKS.json."""
    if shard_spec:
        raise SystemExit("REFUSED: the control stage is one unsharded job")
    wt = Path(__file__).resolve().parents[1]
    return run_prelock_controls(D, out_path=wt / "results" / "pcrl_decision_preserving_compression_v1" /
                                "AUDIT_PRELOCK_CHECKS.json")


# ------------------------------------------------------------------ synthetic fixtures and timing
def synthetic_D(n_fit=6065, n_sel=2235, n_score=0, p1=0.68, seed=0):
    rng = np.random.default_rng(seed)
    n = n_fit + n_sel + n_score
    role = np.array([FIT_ROLE] * n_fit + [SEL_ROLE] * n_sel + ["OSF_DEVELOPMENT_ASSESSMENT"] * n_score)
    D = {"sex": (rng.random(n) < p1).astype(np.int64), "unit": np.arange(n, dtype=np.int64),
         "row_id": np.arange(n, dtype=np.int64) * 3 + 11, "role": role, "sealed": True,
         "idx": {FIT_ROLE: np.arange(n_fit), SEL_ROLE: np.arange(n_fit, n_fit + n_sel),
                 "OSF_DEVELOPMENT_ASSESSMENT": np.arange(n_fit + n_sel, n)}}
    return D


def synthetic_policy(D, states=(16, 48), seed=0, leak=0.0):
    """Study-shaped policy release: per recipient `states` tokens spread over the K classes; leak > 0 tilts token
    frequencies by SEX (synthetic only)."""
    rng = np.random.default_rng(seed)
    n = len(D["row_id"])
    z = {"row_id": D["row_id"]}
    for i, (K, a) in enumerate(zip(KS, states), start=1):
        per = max(a // K, 1)
        w = rng.dirichlet(np.ones(a)) + leak * (D["sex"][:, None] * np.linspace(-1, 1, a)[None] * 0 + 0)
        base = rng.dirichlet(np.ones(a))
        if leak:
            tilt = np.exp(leak * np.linspace(-1, 1, a))
            pm = np.where(D["sex"][:, None] == 1, base * tilt, base / tilt)
            pm = pm / pm.sum(1, keepdims=True)
            u = rng.random(n)[:, None]
            tok = (u > np.cumsum(pm, 1)).sum(1)
        else:
            tok = rng.choice(a, n, p=base)
        del w
        tok = np.minimum(tok, a - 1)
        cls = np.minimum(tok // per, K - 1)
        Q = np.full((a, K), 0.0)
        for t in range(a):
            c = min(t // per, K - 1)
            v = rng.dirichlet(np.ones(K))
            v[c] = v.max() + 0.2
            Q[t] = v / v.sum()
        z[f"tok{i}"], z[f"q{i}"], z[f"hard{i}"], z[f"alpha{i}"] = tok, Q[tok], cls, np.asarray(a)
        assert np.array_equal(Q[tok].argmax(1), cls)
    return z


def timing(states=(16, 48), reps=1, slate="final", seed=0):
    """Seconds for one policy inner audit on study-shaped synthetic data (AUDIT_FIT 6,065 / INNER 2,235 rows)."""
    D = synthetic_D(seed=seed)
    out = []
    for r in range(reps):
        z = synthetic_policy(D, states, seed=seed + r)
        V = policy_views(z, D)
        t0, c0 = time.time(), time.process_time()
        rec = inner_audit(V, D, slate=slate)
        out.append({"wall_s": round(time.time() - t0, 2), "cpu_s": round(time.process_time() - c0, 2),
                    "dims": {w: int(V["X"][w].shape[1]) for w in PRIMARY_VIEWS},
                    "per_view_attacker_seconds": rec["fit_seconds"]})
    return out


def main(argv=None):
    """python -m dpc.audit inner --kind policy|source|reference --cids <cid> ... --seeds 0 1 2   (lead, after the lock;
           prints the record; dpc.run.stage_inner is the unit writer)
    python -m dpc.audit controls [--out results/pcrl_decision_preserving_compression_v1/AUDIT_PRELOCK_CHECKS.json]
    python -m dpc.audit timing [--reps 1] [--states 16 48]                                    (synthetic only)"""
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("inner", "controls", "timing"))
    ap.add_argument("--kind", default="policy")
    ap.add_argument("--cids", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--states", nargs=2, type=int, default=[16, 48])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--units-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.cmd == "timing":
        print(json.dumps(timing(tuple(a.states), a.reps), indent=1))
        return
    from dpc import data as DD
    D = DD.load()
    if a.cmd == "inner":
        for k in a.seeds:
            for cid in a.cids:
                r = inner_unit(a.kind, k, cid, D, a.units_dir)
                print(cid, k, {w: round(r["recovery"]["auc"][w], 4) for w in PRIMARY_VIEWS}, r["wall_s"], flush=True)
    else:
        r = run_prelock_controls(D, units_dir=a.units_dir, out_path=a.out)
        print(json.dumps(r["verdict"], indent=1), r["compute"], flush=True)


if __name__ == "__main__":
    main()
