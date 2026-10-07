"""Synthetic tests of hcal.bank (role D; no real data, no real labels, no unsealing, no outer__ unit).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m hcal.sema --label D:test -- env OMP_NUM_THREADS=1 \\
        PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q tests/pcrl_heldout_calibration_v1/test_bank.py

Fixtures: lra.audit.synthetic_D at small sizes + hcal.bank.synthetic_D's group-closed ATTACK_FIT_NEW (a subset of
AUDIT_FIT), synthetic partition banks with several lookup tables. The pinned "inner" slate (LR, MLP, HGB, DA_LR) is used
for speed; one test runs the pinned FINAL slate end to end. Nothing pinned is edited or monkeypatched.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dpc import audit as DA
from hcal import bank as B
from hcal import data as HD
from hcal import ids as I
from lra import audit as LA

SMALL = dict(n_fit=2000, n_head=50, n_audit=1600, n_sel=800, n_assess=400)
SLATE = "inner"
P_LEG = "U|JOINT|i8o64|l0.1"          # a legacy partition name (labels only; synthetic shapes)
P_LRA = "U|K-JOINT-PAIR|i8o64"
VIEWS = DA.PRIMARY_VIEWS


@pytest.fixture(scope="module")
def DS():
    return B.synthetic_D(seed=1, n_new=1100, **SMALL)


@pytest.fixture(scope="module")
def TS(DS):
    return LA.synthetic_teacher(DS, seed=1)


@pytest.fixture(scope="module")
def BANK(DS, TS):
    return B.synthetic_bank(DS, TS, m=(2, 4), n_tables=3, sex_tilt=0.3, seed=1)


@pytest.fixture(scope="module")
def ROWS(DS):
    return HD.role_rows(DS, "ATTACK_FIT_NEW"), np.asarray(DS["idx"]["INNER_SELECTION"])


@pytest.fixture(scope="module")
def FRESH(DS, BANK, ROWS):
    V = B.fresh_views(*BANK, DS)
    rec, arr = B.fresh_family(V, DS, *ROWS, slate=SLATE)
    return V, rec, arr


def _release(bank, tables, name):
    """lra-format release (row_id, tok_i, q_i, hard_i, alpha_i) decoding with one lookup table."""
    z = {"row_id": np.asarray(bank["row_id"])}
    for i in (1, 2):
        T = tables[name][i - 1]
        z.update({f"tok{i}": bank[f"tok{i}"], f"q{i}": T[bank[f"tok{i}"]], f"hard{i}": bank[f"hard{i}"],
                  f"alpha{i}": np.asarray(bank[f"alpha{i}"])})
    return z


@pytest.fixture(scope="module")
def LEGACY(DS, BANK):
    """Admitted-style lra inner records (schema lra-inner-v1, primary code family fitted on AUDIT_FIT) of the D0 and D1
    original releases of P_LEG, with their stored arrays and views."""
    bank, tables = BANK
    out = []
    for (rid, name), tab in zip(B.legacy_units(0, P_LEG), ("D0", "D1")):
        z = _release(bank, tables, tab)
        V = LA.policy_views(z, DS)
        r, a = LA.inner_family(V, DS, slate=SLATE)
        rec = {"schema": LA.SCHEMA, "kind": "policy", "cid": rid, "seed": 0, "primary_family": "code",
               "inner_unit": name, "recovery": LA._pub(r), "view_fingerprint": DA.view_fingerprint(V)}
        out.append((name, rec, a, V))
    return out


def _auc_fixed(y, p):
    return float(roc_auc_score(np.asarray(y) == 1, np.asarray(p, dtype=np.float64)))


# ====================================================================== fresh views
def test_fresh_views_layout_tables_and_renumbering(DS, BANK):
    bank, tables = BANK
    raw, toks, _, _ = B.fresh_design(bank, tables, DS, expected=list(tables))     # before hygiene
    a1, a2 = int(bank["alpha1"]), int(bank["alpha2"])
    nt = len(tables)
    assert raw["v1"].shape[1] == a1 + nt * 2 + 2 and raw["v2"].shape[1] == a2 + nt * 6 + 6
    for i, a, K in ((1, a1, 2), (2, a2, 6)):
        X = raw[f"v{i}"]
        assert np.all(X[:, :a].sum(1) == 1)                                       # one-hot over the full alphabet
        col = DA.canonical_columns(bank[f"tok{i}"], DS, a)
        assert np.array_equal(X[:, :a].argmax(1), col[bank[f"tok{i}"]])
        for j, nm in enumerate(tables):                                            # registered order
            assert np.array_equal(X[:, a + j * K:a + (j + 1) * K], tables[nm][i - 1][bank[f"tok{i}"]])
        assert np.array_equal(X[:, a + nt * K:], DA.onehot(bank[f"hard{i}"], K))
        assert np.array_equal(toks[f"v{i}"], bank[f"tok{i}"])
    V = B.fresh_views(bank, tables, DS, expected=list(tables))                     # after hygiene
    hyg = V["meta"]["hygiene"]
    for w in ("v1", "v2"):
        assert np.array_equal(V["X"][w], raw[w][:, hyg[w]["kept"]])
        assert np.array_equal(V["tokens"][w], bank[f"tok{w[1]}"])                 # tokens unchanged
    assert np.array_equal(V["X"]["pair"], np.hstack([V["X"]["v1"], V["X"]["v2"]])[:, hyg["pair"]["kept"]])
    # renumbering tokens (with the class and table rows) leaves the design matrices bit-identical
    rng = np.random.default_rng(4)
    bk, tb = dict(bank), {nm: list(v) for nm, v in tables.items()}
    for i in (1, 2):
        a = int(bank[f"alpha{i}"])
        perm = rng.permutation(a)                                                  # old -> new id
        inv = np.argsort(perm)
        bk[f"tok{i}"] = perm[bank[f"tok{i}"]]
        bk[f"class{i}"] = bank[f"class{i}"][inv]
        for nm in tb:
            tb[nm][i - 1] = tables[nm][i - 1][inv]
    V2 = B.fresh_views(bk, tb, DS)
    assert DA.view_fingerprint(V2) == DA.view_fingerprint(V)


def test_view_hygiene_drops_exactly_constant_and_duplicate_columns_on_F():
    F = np.arange(6)
    base = np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.7, 5.0, 6.0, 7.0, 8.0])
    other = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0])
    X = np.stack([base,                                                            # 0 varies on F: kept
                  np.r_[np.full(6, 2.0), [1, 2, 3, 4]],                            # 1 constant on F only: dropped
                  np.r_[base[:6], [0, 0, 0, 0]],                                   # 2 = col 0 on F only: dropped
                  other,                                                           # 3 varies: kept
                  other.copy(),                                                    # 4 duplicate of 3: dropped
                  np.zeros(10),                                                    # 5 all zero: dropped
                  np.r_[other[:6] * 2, [9, 9, 9, 9]],                              # 6 varies (scaled, not equal): kept
                  -other], axis=1)                                                 # 7 varies: kept
    Xh, rec = B.view_hygiene(X, F)
    assert rec["kept"] == [0, 3, 6, 7] and rec["dropped_constant"] == 2 and rec["dropped_duplicate"] == 2
    assert rec["n_in"] == 8 and rec["n_kept"] == 4 and np.array_equal(Xh, X[:, [0, 3, 6, 7]])
    assert rec["kept_sha256"] == DA.sha_arrays(np.asarray([0, 3, 6, 7], dtype=np.int64))
    assert B.view_hygiene(X, F)[1] == rec                                          # deterministic
    # brute force on a random matrix with planted constant / duplicate columns
    rng = np.random.default_rng(3)
    Y = rng.integers(0, 3, (40, 30)).astype(float)
    Y[:20, 5] = 1.0
    Y[:20, 9] = Y[:20, 2]
    Y[:, 17] = Y[:, 4]
    G = np.arange(20)
    _, r = B.view_hygiene(Y, G)
    kept = r["kept"]
    for j in range(Y.shape[1]):
        const = np.all(Y[G, j] == Y[G[0], j])
        dup = any(np.array_equal(Y[G, j], Y[G, k]) for k in kept if k < j)
        assert (j in kept) == (not const and not dup)                              # varying, non-duplicate: kept
    assert all(not np.array_equal(Y[G, a], Y[G, b]) for a in kept for b in kept if a < b)


def test_fresh_view_hygiene_is_label_free_deterministic_and_needs_the_fit_role(DS, BANK, ROWS):
    bank, tables = BANK
    fit, _ = ROWS
    V = B.fresh_views(bank, tables, DS)
    hyg = V["meta"]["hygiene"]
    assert hyg["fit_rows"]["rows"] == len(fit) and hyg["rule"] == B.HYGIENE_RULE
    assert hyg["v2"]["dropped_constant"] > 0                                       # e.g. the absent class's token
    flipped = {**DS, "sex": 1 - np.where(np.asarray(DS["sex"]) < 0, 0, DS["sex"])}
    V2 = B.fresh_views(bank, tables, flipped)                                      # labels changed -> same columns
    assert V2["meta"]["hygiene"] == hyg and DA.view_fingerprint(V2) == DA.view_fingerprint(V)
    assert DA.view_fingerprint(B.fresh_views(bank, tables, DS)) == DA.view_fingerprint(V)
    raw, _, _, _ = B.fresh_design(bank, tables, DS)
    for w in ("v1", "v2"):
        kept = hyg[w]["kept"]
        Xf = raw[w][fit]
        assert all(not np.all(Xf[:, j] == Xf[0, j]) for j in kept)                # every kept column varies on F
        for j in set(range(raw[w].shape[1])) - set(kept):                          # every drop is justified
            assert np.all(Xf[:, j] == Xf[0, j]) or any(np.array_equal(Xf[:, j], Xf[:, k]) for k in kept if k < j)
        assert hyg[w]["n_in"] - hyg[w]["n_kept"] == hyg[w]["dropped_constant"] + hyg[w]["dropped_duplicate"]
    no_roles = {k: v for k, v in DS.items() if k != "hcal"}
    with pytest.raises(B.BankRefused, match="ATTACK_FIT_NEW"):
        B.fresh_views(bank, tables, no_roles)


def test_hygiene_repairs_the_failing_planted_view_at_real_role_sizes():
    """The synthetic CONF_r2 seed-0 fresh view whose DA SVD did not converge (real role sizes) now fits every DA
    reader."""
    from hcal import controls as C
    from smf import audit as SA
    D = B.synthetic_D(seed=0, n_new=4064)
    t = LA.synthetic_teacher(D, seed=0)
    bank, tables = B.synthetic_bank(D, t, m=(8, 64), n_tables=6, sex_tilt=0.1, seed=0)
    halves = SA.null_split(D)
    Sp, _ = C.permuted_sex(D, halves)
    bk, tb = C.split_bank(bank, tables, 2, SA.noisy_sex(Sp, C.CONTROL_SEED), collide=False)
    F = HD.role_rows(D, "ATTACK_FIT_NEW")
    raw, _, _, _ = B.fresh_design(bk, tb, D)
    assert raw["v2"].shape[1] == 684
    V = B.fresh_views(bk, tb, D)
    h = V["meta"]["hygiene"]
    assert h["v2"]["n_kept"] < 684 and h["v2"]["dropped_constant"] >= 11
    for w in ("v2", "pair"):
        X = np.asarray(V["X"][w])[F]
        for fac in (SA.da_lr, SA.da_mlp):                                          # the readers that failed before
            fac(0).fit(X, Sp[F])


def test_fresh_views_refuse_invalid_tables_and_decisions(DS, BANK):
    bank, tables = BANK
    T1, T2 = tables["D0"]
    bad_sum = {**tables, "D0": (T1 * 1.01, T2)}
    with pytest.raises(B.BankRefused, match="sum to one"):
        B.fresh_views(bank, bad_sum, DS)
    swapped = T2.copy()
    swapped[0] = swapped[0][::-1]                                                  # argmax no longer the class
    with pytest.raises(B.BankRefused, match="strict argmax"):
        B.fresh_views(bank, {**tables, "D1": (tables["D1"][0], swapped)}, DS)
    tie = T1.copy()
    c = int(bank["class1"][0])
    tie[0] = 0.0
    tie[0, c] = tie[0, 1 - c] = 0.5                                                # non-strict argmax
    with pytest.raises(B.BankRefused, match="strict argmax"):
        B.fresh_views(bank, {**tables, "D0": (tie, T2)}, DS)
    bk = dict(bank)
    h = bank["hard2"].copy()
    h[5] = (h[5] + 1) % 6
    bk["hard2"] = h
    with pytest.raises(B.BankRefused, match="decision differs"):
        B.fresh_views(bk, tables, DS)
    with pytest.raises(B.BankRefused, match="registered list"):
        B.fresh_views(bank, tables, DS, expected=list(tables)[::-1])
    with pytest.raises(B.BankRefused, match="aligned"):
        B.fresh_views({**bank, "row_id": bank["row_id"][::-1]}, tables, DS)


# ====================================================================== fresh family
def test_fresh_family_record_contract_and_seed_alignment(DS, FRESH, ROWS):
    V, rec, arr = FRESH
    fit, sel = ROWS
    ys = np.asarray(DS["sex"])[sel]
    assert rec["schema"] == B.SCHEMA_FRESH and rec["roles"] == {"fit": "ATTACK_FIT_NEW",
                                                                "select_and_score": "INNER_SELECTION"}
    assert rec["n_fit"] == len(fit) and rec["n_select"] == len(sel) and rec["attacker_seeds"] == [0, 1, 2]
    h = B.role_hashes(DS)
    assert rec["fit_row_id_sha256"] == h["ATTACK_FIT_NEW"] != h["AUDIT_FIT"]
    assert rec["sel_row_id_sha256"] == h["INNER_SELECTION"]
    assert rec["view_fingerprint"] == DA.view_fingerprint(V)
    assert np.array_equal(rec["sel_row_id"], np.asarray(DS["row_id"])[sel])
    P = dict(zip(rec["inner_predictions"]["keys"], rec["inner_predictions"]["P"]))
    for w in VIEWS:
        for crit in B.CRITS:
            A = arr[f"{crit}_{w}"]
            assert A.shape == (3, len(sel))                                        # (seeds, INNER rows)
            s = rec["selection"][w][crit]
            assert np.array_equal(A[0], P[s["pred_key"]])                          # seed 0 = the bank prediction
            if s["attacker"].startswith("CC"):
                assert np.array_equal(A[1], A[0]) and np.array_equal(A[2], A[0])  # deterministic counts
            stat = DA.auc1 if crit == "auc" else DA.ce1
            assert rec[f"{crit}_per_seed"][w] == [stat(ys, p) for p in A]
            assert rec[crit][w] == float(np.mean(rec[f"{crit}_per_seed"][w]))
            assert rec[f"{crit}_seed0"][w] == rec[f"{crit}_per_seed"][w][0]
        # every bank row: fixed-orientation statistics of its stored prediction (independent sklearn AUC)
        for row in rec["tables"][w]:
            assert abs(row["inner_auc"] - _auc_fixed(ys, P[row["pred_key"]])) <= 1e-12
    assert rec["worse_local"] == max(rec["auc"]["v1"], rec["auc"]["v2"])
    pub = B.public_fresh(rec)
    assert "inner_predictions" not in pub and pub["inner_predictions_sha256"] == DA.sha_arrays(
        rec["inner_predictions"]["P"])
    json.dumps(pub, allow_nan=False)
    files = B.fresh_unit_arrays(rec, arr)
    assert set(files) == {"keys_fresh", "P_fresh", "sel_row_id"} | {f"SEL_fresh_{c}_{w}" for c in B.CRITS
                                                                    for w in VIEWS}


def test_pair_bank_holds_coalition_and_both_ignore_recipient_banks(FRESH):
    _, rec, _ = FRESH
    pair = rec["tables"]["pair"]
    cands = [r["candidate"] for r in pair]
    n1, n2 = len(rec["tables"]["v1"]), len(rec["tables"]["v2"])
    k = cands.count("coalition")
    assert cands == ["coalition"] * k + ["ignore_recipient_2"] * n1 + ["ignore_recipient_1"] * n2   # _bank_lists order
    assert {r["view"] for r in pair if r["candidate"] == "ignore_recipient_2"} == {"v1"}
    assert {r["view"] for r in pair if r["candidate"] == "ignore_recipient_1"} == {"v2"}
    att = {r["attacker"] for r in pair if r["candidate"] == "coalition"}
    assert {f"CCpair_alpha{a}" for a in DA.CC_ALPHAS} <= att
    assert {f"CC_alpha{a}" for a in DA.CC_ALPHAS} <= {r["attacker"] for r in rec["tables"]["v1"]}
    assert rec["auc_seed0"]["pair"] >= max(rec["auc_seed0"]["v1"], rec["auc_seed0"]["v2"]) - 1e-12


def _bit_bank(D, bit1, bit2, seed=3):
    """Two tokens per class: token = 2 class + bit; tables decode both copies identically (class-preserving)."""
    t = LA.synthetic_teacher(D, seed=seed)
    bank, tables = {"row_id": np.asarray(D["row_id"])}, {"D0": [None, None], "D1": [None, None]}
    for i, K, bit in ((1, 2, bit1), (2, 6, bit2)):
        d = np.asarray(t[f"d{i}"], dtype=np.int64)
        cls = np.repeat(np.arange(K), 2)
        E = DA.onehot(cls, K)
        tables["D0"][i - 1] = 0.6 * E + 0.4 / K
        tables["D1"][i - 1] = 0.8 * E + 0.2 / K
        bank.update({f"tok{i}": 2 * d + bit, f"hard{i}": d, f"class{i}": cls, f"alpha{i}": np.int64(2 * K)})
    return bank, {k: tuple(v) for k, v in tables.items()}


def test_orientation_is_fixed_an_anti_correlated_reader_stays_below_half(DS, ROWS):
    fit, sel = ROWS
    S = np.asarray(DS["sex"])
    rng = np.random.default_rng(11)
    n = len(S)
    keep = rng.random(n) < 0.9
    bit1 = rng.integers(0, 2, n)
    bit1[fit] = np.where(keep[fit], S[fit], 1 - S[fit])                           # fit rows: bit ~ S
    bit1[sel] = np.where(keep[sel], 1 - S[sel], S[sel])                           # INNER rows: bit ~ 1 - S
    bit2 = rng.integers(0, 2, n)
    V = B.fresh_views(*_bit_bank(DS, bit1, bit2), DS)
    rec, arr = B.fresh_family(V, DS, fit, sel, slate=SLATE)
    ys = S[sel]
    v1 = [r["inner_auc"] for r in rec["tables"]["v1"]]
    assert max(v1) < 0.5                                                           # every reader anti-correlated
    assert rec["auc_seed0"]["v1"] == max(v1) < 0.5                                 # selected = bank max, not flipped
    assert rec["auc"]["v1"] < 0.5
    assert all(abs(_auc_fixed(ys, p) - x) <= 1e-12 for p, x in zip(arr["auc_v1"], rec["auc_per_seed"]["v1"]))
    P = dict(zip(rec["inner_predictions"]["keys"], rec["inner_predictions"]["P"]))
    p = P["v1:CC_alpha1.0"]
    assert p[bit1[sel] == 1].mean() > p[bit1[sel] == 0].mean()                     # P(S=1) column, as fitted


def test_unseen_token_uses_fit_prior_and_unseen_tuple_the_inner_selected_fallback(DS, BANK, ROWS):
    bank, tables = BANK
    fit, sel = ROWS
    S = np.asarray(DS["sex"])
    bk = {k: np.array(v) for k, v in bank.items()}
    a1 = int(bank["alpha1"])
    bk["alpha1"] = np.int64(a1 + 1)
    bk["class1"] = np.append(bank["class1"], 0)                                    # a new class-0 token
    tb = {}
    for nm, (T1, T2) in tables.items():
        tb[nm] = (np.vstack([T1, B._smooth(np.full(2, 0.5), 0, 2)]), T2)
    tok = bk["tok1"].copy()
    inner0 = sel[bank["hard1"][sel] == 0][:20]
    held = DS["hcal"]["roles"]["CALIBRATION_HELDOUT_GROUP_ROWS"]
    held0 = held[bank["hard1"][held] == 0][:10]                                    # in AUDIT_FIT, NOT in the fresh fit
    tok[inner0] = a1
    tok[held0] = a1
    bk["tok1"] = tok
    V = B.fresh_views(bk, tb, DS)
    rec, _ = B.fresh_family(V, DS, fit, sel, slate=SLATE)
    P = dict(zip(rec["inner_predictions"]["keys"], rec["inner_predictions"]["P"]))
    pos = np.flatnonzero(np.isin(sel, inner0))
    prior_fit = float(np.mean(S[fit] == 1))
    for a in DA.CC_ALPHAS:
        assert np.all(P[f"v1:CC_alpha{a}"][pos] == prior_fit)                     # the ATTACK_FIT_NEW prior
    assert rec["coverage"]["v1"]["inner_rows_fallback"] == 20
    # unseen tuples: the rule with the lowest INNER cross-entropy over the unseen-tuple INNER rows (ties in order)
    keys = DA._pair_keys(bk["tok1"], bk["tok2"])
    unseen = ~np.isin(keys[sel], keys[fit])
    assert unseen.sum() >= 20
    ys = S[sel]
    for a in DA.CC_ALPHAS:
        cand = {"local_1": P[f"v1:CC_alpha{a}"], "local_2": P[f"v2:CC_alpha{a}"], "prior": np.full(len(sel), prior_fit)}
        ces = {r: DA.ce1(ys[unseen], cand[r][unseen]) for r in DA.FALLBACK_RULES}
        want = min(DA.FALLBACK_RULES, key=lambda r: (ces[r], DA.FALLBACK_RULES.index(r)))
        assert rec["coverage"]["pair"]["fallback_rule"][f"CCpair_alpha{a}"] == want
        assert np.array_equal(P[f"pair:CCpair_alpha{a}"][unseen], cand[want][unseen])


def test_fit_rows_are_strictly_attack_fit_new(DS, BANK, FRESH, ROWS):
    V, rec, arr = FRESH
    fit, sel = ROWS
    S = np.asarray(DS["sex"])
    outside = np.setdiff1d(DS["idx"]["AUDIT_FIT"], fit)
    assert outside.size and np.array_equal(np.sort(outside), np.sort(DS["hcal"]["roles"][
        "CALIBRATION_HELDOUT_GROUP_ROWS"]))
    y2 = S.copy()
    y2[outside] = 1 - y2[outside]                                                  # perturb AUDIT_FIT labels outside
    rec2, arr2 = B.fresh_family(V, DS, fit, sel, slate=SLATE, y=y2)
    assert np.array_equal(rec2["inner_predictions"]["P"], rec["inner_predictions"]["P"])
    assert all(np.array_equal(arr2[k], arr[k]) for k in arr)
    y3 = S.copy()
    y3[fit[:200]] = 1 - y3[fit[:200]]                                              # inside: predictions change
    rec3, _ = B.fresh_family(V, DS, fit, sel, slate=SLATE, y=y3)
    assert not np.array_equal(rec3["inner_predictions"]["P"], rec["inner_predictions"]["P"])
    with pytest.raises(B.BankRefused, match="ATTACK_FIT_NEW"):
        B.fresh_family(V, DS, DS["idx"]["AUDIT_FIT"], sel, slate=SLATE)            # all historical AUDIT_FIT refused
    with pytest.raises(B.BankRefused):
        B.fresh_family(V, DS, np.append(fit, sel[0]), sel, slate=SLATE)
    with pytest.raises(B.BankRefused, match="INNER_SELECTION"):
        B.fresh_family(V, DS, fit, sel[:-1], slate=SLATE)
    y4 = S.copy()
    y4[fit[3]] = -1
    with pytest.raises(B.BankRefused, match="sealed"):
        B.fresh_family(V, DS, fit, sel, slate=SLATE, y=y4)
    with pytest.raises(B.BankRefused, match="complete fresh view"):
        B.fresh_family(LA.policy_views(_release(*BANK, "D0"), DS), DS, fit, sel, slate=SLATE)


def test_token_only_diagnostic_uses_cell_readers_on_attack_fit_new(DS, BANK, ROWS, FRESH):
    bank, _ = BANK
    _, frec, _ = FRESH
    rec, arr = B.fresh_token_family(B.token_views(bank, DS), DS, *ROWS)
    keys = rec["inner_predictions"]["keys"]
    assert all(k.split(":")[1].startswith("CC") for k in keys) and rec["slate_members"] == []
    assert rec["roles"]["fit"] == "ATTACK_FIT_NEW" and rec["fit_row_id_sha256"] == frec["fit_row_id_sha256"]
    P = dict(zip(keys, rec["inner_predictions"]["P"]))
    F = dict(zip(frec["inner_predictions"]["keys"], frec["inner_predictions"]["P"]))
    for k in keys:                                                                 # same tokens, same fit rows
        assert np.array_equal(P[k], F[k])
    assert all(np.array_equal(a[0], a[1]) and np.array_equal(a[0], a[2]) for a in arr.values())


# ====================================================================== common record and union rule
def test_common_record_is_shared_by_every_decoder_variant(DS, BANK, ROWS, FRESH, LEGACY):
    bank, tables = BANK
    _, rec, _ = FRESH
    legacy = [(n, r) for n, r, _, _ in LEGACY]
    common = B.common_record(legacy, B.public_fresh(rec), partition=P_LEG, seed=0, hashes=B.role_hashes(DS))
    assert [b["name"] for b in common["banks"]] == [n for _, n in B.legacy_units(0, P_LEG)] + ["fresh"]
    assert common["decoder_variants"] == I.decoders_of(P_LEG)
    entries = [B.legacy_entry(n, r) for n, r in legacy] + [B.fresh_entry(rec)]
    for w in VIEWS:
        for crit, best in (("auc", max), ("ce", min)):
            vals = [e[f"{crit}_seed0"][w] for e in entries]
            j = vals.index(best(vals))
            assert common[f"{crit}_seed0"][w] == vals[j]
            assert common[crit][w] == entries[j][crit][w]                          # winner's seed 0-2 mean
            assert common[f"{crit}_per_seed"][w] == entries[j][f"{crit}_per_seed"][w]
            assert common["winner" if crit == "auc" else "ce_winner"][w] == entries[j]["name"]
            assert common["bank_seed0"][crit][w] == {e["name"]: e[f"{crit}_seed0"][w] for e in entries}
    assert common["mean_local"] == (common["auc"]["v1"] + common["auc"]["v2"]) / 2
    variants = B.assign_variants(0, P_LEG, common)
    assert list(variants) == [I.release_id(P_LEG, d) for d in I.decoders_of(P_LEG)]
    receipt = B.check_common(variants, common)
    assert receipt["ok"] and len(receipt["variants"]) == len(I.decoders_of(P_LEG))
    # a variant-specific view (a different table subset over the same tokens) is a different fresh view, but the
    # record stays the partition's common record
    Vsub = B.fresh_views(bank, {"D0": tables["D0"]}, DS)
    assert DA.view_fingerprint(Vsub) != rec["view_fingerprint"]
    bad = dict(variants)
    rid = list(bad)[2]
    tampered = copy.deepcopy(common)
    tampered["auc"]["pair"] += 1e-9
    bad[rid] = tampered
    with pytest.raises(AssertionError, match="do not share"):
        B.check_common(bad, common)
    with pytest.raises(B.BankRefused, match="registered"):
        B.common_record(legacy[::-1], rec, partition=P_LEG, seed=0)


def _fam(auc_ps, ce_ps, fit_sha="F" * 64, sel_sha="S" * 64, slate=SLATE, seeds=(0, 1, 2), label="own:{w}:LR_C1"):
    """A consistent family record (per view: per-seed lists; seed-0 = first; mean = reported)."""
    r = {"family": "code", "fit_row_id_sha256": fit_sha, "sel_row_id_sha256": sel_sha, "slate": slate,
         "attacker_seeds": list(seeds), "auc": {}, "ce": {}, "auc_seed0": {}, "ce_seed0": {}, "auc_per_seed": {},
         "ce_per_seed": {}, "selected": {}, "ce_selected": {}, "selection": {}}
    for w in VIEWS:
        a, c = list(auc_ps[w]), list(ce_ps[w])
        r["auc_per_seed"][w], r["ce_per_seed"][w] = a, c
        r["auc"][w], r["ce"][w] = float(np.mean(a)), float(np.mean(c))
        r["auc_seed0"][w], r["ce_seed0"][w] = a[0], c[0]
        lab = label.format(w=w)
        r["selected"][w] = r["ce_selected"][w] = lab
        view = "v1" if w == "pair" else w
        det = {"candidate": "own", "view": view, "attacker": "LR_C1", "pred_key": f"{view}:LR_C1", "label": lab}
        r["selection"][w] = {"auc": {**det, "inner_auc": a[0], "inner_ce": c[0]},
                             "ce": {**det, "inner_auc": a[0], "inner_ce": c[0]}}
    return r


def _legacy_rec(cid, fam):
    return {"schema": LA.SCHEMA, "kind": "policy", "cid": cid, "seed": 0, "primary_family": "code", "recovery": fam}


def _fresh_rec(fam):
    return {**fam, "schema": B.SCHEMA_FRESH, "family": "fresh", "roles": {"fit": "ATTACK_FIT_NEW"}}


def _const(a0, ce0=0.6, extra=(0.01, -0.01)):
    return ({w: [a0, a0 + extra[0], a0 + extra[1]] for w in VIEWS}, {w: [ce0, ce0, ce0] for w in VIEWS})


def test_union_winner_and_tie_rules():
    (rid0, n0), (rid1, n1) = B.legacy_units(0, P_LEG)
    L0 = _fam(*_const(0.70, 0.60))
    L1 = _fam(*_const(0.70 + 1e-9, 0.60))                                          # beats L0 by > 1e-12
    Fr = _fam(*_const(0.70 + 1e-9 + 5e-13, 0.60 - 1e-9), fit_sha="N" * 64)          # beats L1 by only 5e-13
    rec = B.common_record([(n0, _legacy_rec(rid0, L0)), (n1, _legacy_rec(rid1, L1))], _fresh_rec(Fr),
                          partition=P_LEG, seed=0)
    for w in VIEWS:
        assert rec["winner"][w] == n1                                              # running best, 1e-12 tie rule
        assert rec["auc"][w] == L1["auc"][w] and rec["auc_per_seed"][w] == L1["auc_per_seed"][w]
        assert rec["ce_winner"][w] == "fresh" and rec["ce"][w] == Fr["ce"][w]      # CE: smaller by > 1e-12
        assert rec["winner_detail"][w]["auc"]["kind"] == "legacy" and rec["winner_detail"][w]["auc"]["unit"] == n1
        assert rec["winner_detail"][w]["ce"]["fit_role"] == "ATTACK_FIT_NEW"
        assert rec["winner_detail"][w]["ce"]["stored_key"] == f"SEL_fresh_ce_{w}"
        assert rec["selected"][w] == f"{n1}:{L1['selected'][w]}"
    # exact ties keep the earlier bank (AUC and CE)
    tie = B.common_record([(n0, _legacy_rec(rid0, L0)), (n1, _legacy_rec(rid1, _fam(*_const(0.70, 0.60))))],
                          _fresh_rec(_fam(*_const(0.70, 0.60), fit_sha="N" * 64)))
    assert set(tie["winner"].values()) == {n0} and set(tie["ce_winner"].values()) == {n0}
    # a fresh bank strictly better wins and reports its own seed mean
    Fb = _fam(*_const(0.80, 0.50, extra=(0.03, 0.03)), fit_sha="N" * 64)
    win = B.common_record([(n0, _legacy_rec(rid0, L0))], _fresh_rec(Fb))
    assert set(win["winner"].values()) == {"fresh"} and win["auc"]["v1"] == Fb["auc"]["v1"] > 0.81
    assert win["coalition_minus_best_local"] == win["auc"]["pair"] - max(win["auc"]["v1"], win["auc"]["v2"])


def test_union_refuses_inconsistent_banks():
    (rid0, n0), (rid1, n1) = B.legacy_units(0, P_LEG)
    L0 = _legacy_rec(rid0, _fam(*_const(0.7)))
    Fr = _fresh_rec(_fam(*_const(0.7), fit_sha="N" * 64))

    def refused(legacy, fresh, match):
        with pytest.raises(B.BankRefused, match=match):
            B.common_record(legacy, fresh)
    refused([(n0, {**L0, "schema": "cbp-inner"})], Fr, "schema")
    refused([(n0, {**L0, "primary_family": "token"})], Fr, "complete code interface")
    refused([(n0, L0)], _fresh_rec(_fam(*_const(0.7), sel_sha="X" * 64, fit_sha="N" * 64)), "sel_row_id")
    refused([(n0, L0)], _fresh_rec(_fam(*_const(0.7), slate="final", fit_sha="N" * 64)), "slate")
    refused([(n0, L0)], _fresh_rec(_fam(*_const(0.7), seeds=(0, 1, 3), fit_sha="N" * 64)), "attacker seeds")
    refused([(n0, L0), (n1, _legacy_rec(rid1, _fam(*_const(0.7), fit_sha="G" * 64)))], Fr, "fit rows")
    refused([(n0, L0)], _fresh_rec(_fam(*_const(0.7))), "ATTACK_FIT_NEW only")
    bad = copy.deepcopy(L0)
    bad["recovery"]["auc"]["v1"] += 1e-6                                           # not the seed mean
    refused([(n0, bad)], Fr, "seed mean")
    bad = copy.deepcopy(L0)
    bad["recovery"]["auc_per_seed"]["v2"] = [0.7, 0.7]                             # denominator / seed misalignment
    refused([(n0, bad)], Fr, "per-seed values")
    with pytest.raises(B.BankRefused, match="hold"):
        B.common_record([(n0, {**L0, "cid": rid1}), (n1, _legacy_rec(rid1, _fam(*_const(0.7))))], Fr,
                        partition=P_LEG, seed=0)
    with pytest.raises(B.BankRefused, match="sel_row_id|INNER"):
        B.common_record([(n0, L0)], Fr, hashes={"INNER_SELECTION": "S" * 63 + "x", "AUDIT_FIT": "F" * 64,
                                                "ATTACK_FIT_NEW": "N" * 64})


# ====================================================================== U composition
def _u_record(own, comp, winners, code_ids, k=0):
    """An lra SRC|U-shaped record: recovery = own family with composed values written over auc/ce/seed0/selected (as
    lra.audit.inner_unit does), auc_per_seed / ce_per_seed left as the OWN bank's."""
    r = copy.deepcopy(own)
    r["family"] = "interface"
    r["own"] = {key: copy.deepcopy(own[key]) for key in ("auc", "ce", "auc_seed0", "ce_seed0", "selected",
                                                          "ce_selected")}
    c = {key: copy.deepcopy(comp[key]) for key in ("auc", "ce", "auc_seed0", "ce_seed0")}
    c.update({"winner": dict(winners), "ce_winner": dict(winners), "policies": list(code_ids)})
    c["winner_label"] = {w: (own["selected"][w] if winners[w] == "source" else
                             f"composed[{I.lra_unit(k, winners[w])}]:{comp['selected'][w]}") for w in VIEWS}
    c["ce_winner_label"] = dict(c["winner_label"])
    for key in ("auc", "ce", "auc_seed0", "ce_seed0"):
        r[key] = copy.deepcopy(c[key])
    r["selected"], r["ce_selected"] = dict(c["winner_label"]), dict(c["ce_winner_label"])
    r["composed"] = c
    return {"schema": LA.SCHEMA, "kind": "source", "cid": "SRC|U", "seed": k, "primary_family": "interface",
            "inner_unit": f"aud__tea__s{k}__U", "recovery": r,
            "composed": {"auc": c["auc"], "ce": c["ce"], "winner": c["winner"], "ce_winner": c["ce_winner"],
                         "closure": {"ok": True}, "policies": list(code_ids)}}


def test_compose_u_field_mapping_winner_rule_and_order():
    code = "U|LOCAL|i8o64|l0.01|D1"
    own = _fam({w: [0.70, 0.69, 0.71] for w in VIEWS}, {w: [0.55, 0.56, 0.57] for w in VIEWS})
    cf = _fam({w: [0.72, 0.73, 0.74] for w in VIEWS}, {w: [0.54, 0.55, 0.55] for w in VIEWS})
    # composed: v1 won by the code, v2 / pair by the source's own bank
    winners = {"v1": code, "v2": "source", "pair": "source"}
    comp = copy.deepcopy(own)
    for key in ("auc", "ce", "auc_seed0", "ce_seed0"):
        comp[key]["v1"] = cf[key]["v1"]
    comp["selected"] = cf["selected"]
    u = _u_record(own, comp, winners, ["c"] * 3)
    code_rec = _legacy_rec(code, cf)
    e = B.u_entry(u, {code: code_rec}, strict=False)
    assert e["auc_per_seed"]["v1"] == cf["auc_per_seed"]["v1"]                    # code winner: the code's record
    assert e["auc_per_seed"]["v2"] == own["auc_per_seed"]["v2"]                   # source winner: own per-seed
    assert e["selection"]["v1"]["auc"]["kind"] == "lra_code"
    assert e["selection"]["v1"]["auc"]["unit"] == I.lra_inner(0, code)
    assert e["selection"]["v2"]["ce"]["stored_key"] == "SEL_interface_ce_v2"
    with pytest.raises(B.BankRefused, match="required"):
        B.u_entry(u, {}, strict=False)
    with pytest.raises(B.BankRefused, match="84-code"):
        B.u_entry(u, {code: code_rec}, strict=True)
    ps = I.partitions()
    weak = _fresh_rec(_fam(*_const(0.60, 0.70), fit_sha="N" * 64))
    strong = _fresh_rec(_fam({w: [0.90, 0.91, 0.89] for w in VIEWS}, {w: [0.30, 0.31, 0.32] for w in VIEWS},
                             fit_sha="N" * 64))
    rec = B.compose_u(e, [(ps[0], weak), (ps[5], strong)], seed=0, complete=False)
    assert [b["name"] for b in rec["banks"]] == ["aud__tea__s0__U", f"fresh:{ps[0]}", f"fresh:{ps[5]}"]
    assert set(rec["winner"].values()) == {f"fresh:{ps[5]}"} and rec["auc"]["v1"] == strong["auc"]["v1"]
    assert rec["u_variants"] == I.u_release_ids()
    rec2 = B.compose_u(e, [(ps[0], weak)], complete=False)
    assert rec2["winner"]["v1"] == "aud__tea__s0__U" and rec2["auc"]["v1"] == cf["auc"]["v1"]
    assert rec2["winner_detail"]["v1"]["auc"]["kind"] == "lra_code"
    assert rec2["winner_detail"]["v2"]["auc"]["kind"] == "lra_source"
    with pytest.raises(B.BankRefused, match="order"):
        B.compose_u(e, [(ps[5], strong), (ps[0], weak)], complete=False)
    with pytest.raises(B.BankRefused, match="every partition"):
        B.compose_u(e, [(ps[0], weak)])


# ====================================================================== refit hook
def test_refit_selected_reproduces_stored_inner_predictions_bitwise(DS, FRESH, ROWS):
    V, rec, arr = FRESH
    fit, sel = ROWS
    score = np.asarray(DS["idx"]["HEAD_VALIDATION"])
    e = B.fresh_entry(rec)
    for w in VIEWS:
        for crit in B.CRITS:
            spec = B.spec_for(e, w, crit, arr[f"{crit}_{w}"])
            P_sel, P_score = B.refit_selected(spec, V, DS, DS["sex"], fit, sel, score)
            assert np.array_equal(P_sel, arr[f"{crit}_{w}"])
            assert P_score.shape == (3, len(score), 2)
            assert np.allclose(P_score.sum(2), 1.0) and np.array_equal(P_score[:, :, 0], 1.0 - P_score[:, :, 1])
    # every bank member (slate and cell readers) reproduces its seed-0 bank prediction
    P = dict(zip(rec["inner_predictions"]["keys"], rec["inner_predictions"]["P"]))
    for key, p in P.items():
        view, att = key.split(":")
        spec = {"kind": "fresh", "bank": "fresh", "view": view, "attacker": att, "pred_key": key, "slate": SLATE,
                "attacker_seeds": [0, 1, 2], "stored": p[None]}
        P_sel, _ = B.refit_selected(spec, lambda: V, DS, DS["sex"], fit, sel, score, seeds=(0,))
        assert np.array_equal(P_sel[0], p), key
    # score-row labels are never read
    spec = B.spec_for(e, "pair", "auc", arr["auc_pair"])
    y2 = np.asarray(DS["sex"]).copy()
    y2[score] = 1 - y2[score]
    _, a = B.refit_selected(spec, V, DS, DS["sex"], fit, sel, score)
    _, b = B.refit_selected(spec, V, DS, y2, fit, sel, score)
    assert np.array_equal(a, b)


def test_refit_selected_refuses_mismatch_wrong_rows_and_views(DS, BANK, FRESH, ROWS):
    V, rec, arr = FRESH
    fit, sel = ROWS
    score = np.asarray(DS["idx"]["HEAD_VALIDATION"])
    e = B.fresh_entry(rec)
    stored = arr["auc_v2"].copy()
    stored[1, 7] = np.nextafter(stored[1, 7], 2.0)                                 # one ulp
    with pytest.raises(B.RefitMismatch, match="bitwise"):
        B.refit_selected(B.spec_for(e, "v2", "auc", stored), V, DS, DS["sex"], fit, sel, score)
    spec = B.spec_for(e, "v1", "auc", arr["auc_v1"])
    with pytest.raises(B.BankRefused, match="ATTACK_FIT_NEW"):
        B.refit_selected(spec, V, DS, DS["sex"], DS["idx"]["AUDIT_FIT"], sel, score)
    with pytest.raises(B.BankRefused, match="disjoint"):
        B.refit_selected(spec, V, DS, DS["sex"], fit, sel, np.append(score, sel[0]))
    with pytest.raises(B.BankRefused, match="fingerprint"):
        B.refit_selected(spec, B.fresh_views(BANK[0], {"D0": BANK[1]["D0"]}, DS), DS, DS["sex"], fit, sel, score)
    with pytest.raises(B.BankRefused, match="shape"):
        B.refit_selected(B.spec_for(e, "v1", "auc", arr["auc_v1"][:2]), V, DS, DS["sex"], fit, sel, score)


def test_refit_selected_legacy_bank_on_audit_fit(DS, LEGACY, ROWS):
    _, sel = ROWS
    score = np.asarray(DS["idx"]["HEAD_VALIDATION"])
    name, rec, arrays, V = LEGACY[1]
    e = B.legacy_entry(name, rec)
    for w in VIEWS:
        for crit in B.CRITS:
            spec = B.spec_for(e, w, crit, arrays[f"{crit}_{w}"])
            assert spec["fit_role"] == "AUDIT_FIT" and spec["stored_key"] == f"SEL_code_{crit}_{w}"
            P_sel, _ = B.refit_selected(spec, V, DS, DS["sex"], DS["idx"]["AUDIT_FIT"], sel, score)
            assert np.array_equal(P_sel, arrays[f"{crit}_{w}"])
    with pytest.raises(B.BankRefused, match="AUDIT_FIT"):
        B.refit_selected(B.spec_for(e, "v1", "auc", arrays["auc_v1"]), V, DS, DS["sex"],
                         HD.role_rows(DS, "ATTACK_FIT_NEW"), sel, score)


# ====================================================================== admitted store helpers
def test_load_legacy_and_stored_predictions_from_a_hash_complete_store(tmp_path, DS, BANK, LEGACY, FRESH):
    _, frec, _ = FRESH
    for name, rec, arrays, _ in LEGACY:
        files = {"inner_preds.npz": {**{f"SEL_code_{k}": v for k, v in arrays.items()}}}
        LA._save_unit(tmp_path, name, files, rec)
    got = B.load_legacy(0, P_LEG, units_dir=tmp_path)
    assert [g[0] for g in got] == [n for _, n in B.legacy_units(0, P_LEG)]
    common = B.common_record([(n, r) for n, r, _ in got], frec, partition=P_LEG, seed=0, hashes=B.role_hashes(DS))
    for w in VIEWS:
        d = common["winner_detail"][w]["auc"]
        if d["kind"] == "legacy":
            stored = B.stored_predictions(d, units_dir=tmp_path)
            assert np.array_equal(stored, dict(zip([n for n, *_ in LEGACY], [a for _, _, a, _ in LEGACY]))[
                d["unit"]][f"auc_{w}"])
    assert B.legacy_units(0, P_LRA) == [(f"{P_LRA}|D1", f"aud__new__s0__{I.safe(P_LRA)}_D1")]
    # the refit view of a legacy winner is rebuilt from its admitted release through lra.audit.policy_views
    for (name, rec, arrays, V), tab in zip(LEGACY, ("D0", "D1")):
        LA._save_unit(tmp_path, name[len("aud__"):], {"release.npz": _release(*BANK, tab)}, {})
        d = B.spec_for(B.legacy_entry(name, rec), "v2", "auc", arrays["auc_v2"])
        load = B.views_loader(d, DS, units_dir=tmp_path)
        assert DA.view_fingerprint(load()) == rec["view_fingerprint"]
        P_sel, _ = B.refit_selected(d, load, DS, DS["sex"], DS["idx"]["AUDIT_FIT"], DS["idx"]["INNER_SELECTION"],
                                    DS["idx"]["HEAD_VALIDATION"])
        assert np.array_equal(P_sel, arrays["auc_v2"])
    t = LA.synthetic_teacher(DS, seed=1)
    LA._save_unit(tmp_path, "tea__s0__U", {"teacher.npz": t}, {})
    Vu = B.views_loader({"kind": "lra_source", "unit": "aud__tea__s0__U"}, DS, units_dir=tmp_path)()
    assert Vu["family"] == "interface" and Vu["X"]["v2"].shape[1] == 2 * 6
    with pytest.raises(B.BankRefused, match="fresh_views loader"):
        B.views_loader({"kind": "fresh", "unit": "fresh"}, DS)
    name, rec, arrays, _ = LEGACY[0]
    LA._save_unit(tmp_path, LEGACY[1][0], {"inner_preds.npz": {}}, {**rec})        # wrong cid in the D1 slot
    with pytest.raises(B.BankRefused, match="holds"):
        B.load_legacy(0, P_LEG, units_dir=tmp_path)
    (tmp_path / name / "record.json").write_text("{}")                             # no longer hash-complete
    with pytest.raises(SystemExit, match="hash-complete"):
        B.load_legacy(0, P_LEG, units_dir=tmp_path)


# ====================================================================== the pinned FINAL slate end to end
def test_fresh_family_with_the_pinned_final_slate(DS, FRESH, ROWS):
    V, _, _ = FRESH
    rec, arr = B.fresh_family(V, DS, *ROWS, slate="final")
    members = [nm for nm, _ in DA.SLATES["final"]()]
    assert rec["slate_members"] == members and len(members) == 15
    att = {r["attacker"] for r in rec["tables"]["v1"]}
    assert set(members) | {f"CC_alpha{a}" for a in DA.CC_ALPHAS} == att
    assert all(a.shape == (3, len(ROWS[1])) for a in arr.values())
    B.public_fresh(rec)
