"""[lra port of lcr/assess.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
The single locked assessment opening on OSF_DEVELOPMENT_ASSESSMENT for the learned-decoder constrained-release study
(lra; role D; prompt sections 11, 12, 13 Stage 6).

PROVENANCE. A COPY of cbp/assess.py at the cbp final commit 7f3ec67b2ecd86d474e2ff27167091af9923f572 (itself a copy of
qpc/assess.py at d0c8a45, adapted from dpc/assess.py at 0a7b05a5; same gate, same unit contents, same
dpc.audit.final_audit). cbp/, qpc/ and dpc/ are never edited and nothing in them is monkeypatched at runtime.
DOCUMENTED DIFF against cbp/assess.py:
  S1  docstrings; study branch research/pcrl-adult-learned-decoder-release-v1 and the registered lock path
      results/pcrl_adult_learned_decoder_release_v1/EVALUATION_LOCK.json;
  S2  binding: lra.audit, lra.baselines, lra.run and lra.lock (module-hash check); unsealing ONLY through
      lra.data.load(unseal=True), whose gate admits only the caller module lra.assess and only after the lra
      EVALUATION_LOCK is committed and byte-identical on origin/research/pcrl-adult-learned-decoder-release-v1;
  S3  CHAIN lists every scoring file: the lra chain (lra/assess.py, lra/audit.py, lra/baselines.py, lra/data.py,
      lra/run.py, lra/lock.py) plus every imported qpc / dpc / osf / smf / jcv / rgj file it loads (cbp's list plus
      rgj/data.py and smf/data.py, which osf.data imports);
  S4  code release units are pol__ (D0), dec__ (D1 fixed-map) and new__ (fitted) s{k}__<cid>; a unit name is mapped to
      its config id through the REGISTERED lra bank only (never by string surgery); composed_for REFUSES a composed
      code outside the registered 83-code composition bank of its teacher (lra.audit L3; none for RAW-J) and reads
      the composed winners from the lra source inner unit lra.run.inner_name(k, "SRC|<teacher>") = aud__tea__...;
  S5  code releases are scored with all three lra.audit families by default (code = complete interface, PRIMARY;
      token-only and probability-only diagnostics; spec["families"] may restrict to a subset that contains the
      primary); preds.npz keys follow the cbp rule (primary unprefixed, others P_<crit>_<family>_<view>);
  S6  outer_unit refuses to save a record containing a nonfinite float, and refuses nonfinite prediction / utility
      arrays (technical failure; new JSON is finite);
  S7  the restore hook refit_selected_attacker rebuilds dec__ / new__ releases and the token / prob diagnostic
      families as well (lra.audit.code_view_sets);
  S8  (lra.audit L4b) the token-only diagnostic family of a code is scored with the cell readers only
      (lra.audit.final_audit_cells: CC / CCpair on the exact token identities, same dual selection and
      (3, n, 2) prediction contract); the complete-interface (primary) and probability-only families use
      dpc.audit.final_audit with the full FINAL slate, unchanged.
Everything else -- outer-unit contents (P_auc / P_ce arrays of shape (3, n, 2) per view and family, utility arrays,
labels, U anchor), the composed-winner freeze check, jobs_from_lock (only the locked list), load_outer and the restore
hook refit_selected_attacker(units_root, D, release_unit, outer_unit, view="pair", attacker_seed=None) -- is the cbp
logic unchanged.

This is the ONLY lra module that indexes OSF_DEVELOPMENT_ASSESSMENT rows or reads their labels.

GATE. REFUSES unless results/pcrl_adult_learned_decoder_release_v1/EVALUATION_LOCK.json is committed,
byte-identical to its last commit in the working tree, and that commit is contained in
origin/research/pcrl-adult-learned-decoder-release-v1 (git fetch, `git branch -r --contains`, and the blob on
origin equals the working-tree bytes). `open_assessment` also requires lock["locked_code_files"] = {relpath: sha256}
to list EVERY file of the scoring chain (CHAIN) with a matching hash now, and every worktree module loaded by this
process to be locked with its current hash. Only then does `load_unsealed()` call lra.data.load(unseal=True) from
this module. The unsealed assessment role must equal lock["assessment_role"] (rows, groups, row_id_sha256 of
osf.data.manifest). `outer_unit` refuses a sealed D, re-verifies the lock (same commit and sha256, still pushed)
before every unit, and refuses a release unit whose COMPLETE.json file map differs from
lock["seeds"][k]["unit_file_sha256"][unit] (when recorded).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema --label A:assess -- \\
        env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.assess \\
        --evaluation-lock results/pcrl_adult_learned_decoder_release_v1/EVALUATION_LOCK.json \\
        [--seeds 0 1 2] [--labels ...] \\
        [--shard i/n]

LOCK SCHEMA READ (written by the lead's lra.eval_lock):
  lock["seeds"][str(k)] = {"score": {label: spec}, "u_label": "SRC|U",
                           "composed_policies": [code units] (each source composes with its own teacher's codes) or
                                                {teacher: [cids|units]},
                           "unit_file_sha256": {unit: COMPLETE.json files map} (optional, checked when present)}
  spec = {"kind": "policy" | "source" | "reference", "cid": <config id>, "unit": <unit name> (optional, checked)}
         (+ optional "families": subset of the release's families; the primary family is always scored)
  lock["locked_code_files"] = {relpath: sha256};  lock["assessment_role"] = {"rows", "groups", "row_id_sha256"}
  composed_policies[teacher] must contain the locked selected codes of that teacher and seed AND every code of
  lra.audit.composed_freeze_list(k, teacher) (refused otherwise), and nothing outside the registered composition bank
  (S4). The decisions family composes only with the class-only code. Exactness note (role D): every code outside the
  freeze list lost every inner (family, view, criterion) comparison on the SAME seed-0 AUDIT_FIT fits and
  INNER_SELECTION rows that final_audit reselects on, so composing with the freeze list plus the scored codes selects
  the same final attackers as composing with all 83 (and costs far less); composing with all 83 remains valid.

PER RELEASE (dpc.audit.final_audit: the same FINAL slate + cell readers; attackers FITTED on AUDIT_FIT, SELECTED on
INNER_SELECTION over the whole bank (own + composed), the AUC- and CE-selected attackers refit at attacker seeds 0, 1, 2
and scored on the assessment rows; orientation fixed; nothing flipped or clamped). Utility: qpc.utility.release_utility
on the assessment rows (allowlist procedure "assessment") with U of the same seed as the anchor.

UNIT outer__s{k}__{safe(label)}; preds.npz (private; assessment rows in D order; n rows):
  assess_row_id, assess_unit (exact-record group = bootstrap unit), sex, race, y_income, y_occ
  const_class (2,), const1, const2 (n,)        OSF_DEFENSE_FIT majority-class constant predictions
  hard1, hard2 (n,), prob1 (n, 2), prob2 (n, 6) the release's decisions and probabilities (q_i for codes)
  ll1, ll2, br1, br2 (n,)                       per-row true-label log loss (clip 1e-12) and Brier
  u_hard1, u_hard2, u_prob1, u_prob2, u_ll1, u_ll2, u_br1, u_br2   U's anchor on the same rows
  P_auc_{v1,v2,pair}, P_ce_{v1,v2,pair} (3, n, 2)   [P(S=0), P(S=1)] of the AUC- / CE-selected attacker at attacker
                                                    seeds 0, 1, 2, PRIMARY family (score = column 1)
  P_auc_<family>_{w}, P_ce_<family>_{w}         every secondary family (codes: token, prob)
record.json: label, seed, spec, provenance, lock commit / sha256, utility, per family the selected attackers (label,
inner AUC / CE, scored AUC per seed -- descriptive; endpoints are recomputed from preds.npz by lra.infer), coverage
and fallback counts (fit, inner and scored rows), composed policies, preds keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from lra import audit as AU                      # S2
from lra import baselines as BL
from dpc import audit as DA
from qpc import utility as UT

WT = Path(__file__).resolve().parents[1]
STUDY_BRANCH = "research/pcrl-adult-learned-decoder-release-v1"
LOCK_NAME = "EVALUATION_LOCK.json"
LOCK_REL = "results/pcrl_adult_learned_decoder_release_v1/EVALUATION_LOCK.json"
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
CHAIN = ("lra/assess.py", "lra/audit.py", "lra/baselines.py", "lra/data.py", "lra/run.py", "lra/lock.py",   # S3
         "qpc/utility.py", "qpc/data.py", "dpc/audit.py", "dpc/utility.py", "dpc/baselines.py", "dpc/data.py",
         "osf/data.py", "osf/audit.py", "smf/audit.py", "smf/data.py", "jcv/audit.py", "jcv/finalize.py",
         "rgj/finalize.py", "rgj/data.py")
_OPENED = None


# ------------------------------------------------------------------ the lock gate
def _git(repo, *args, text=True):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lock_is_pushed(path, repo=WT, branch=STUDY_BRANCH, rel_required=LOCK_REL, fetch=True):
    """{"ok": bool, "reason" | "commit", "sha256", ...}: the registered lock, committed, unmodified, on origin."""
    p = Path(path).resolve()
    repo = Path(repo).resolve()
    if p.name != LOCK_NAME:
        return {"ok": False, "reason": f"lock file must be named {LOCK_NAME}"}
    if not p.is_file():
        return {"ok": False, "reason": "lock file missing"}
    try:
        rel = str(p.relative_to(repo))
    except ValueError:
        return {"ok": False, "reason": "lock file is outside the study repository"}
    if rel_required is not None and rel != rel_required:
        return {"ok": False, "reason": f"lock file is not the registered {rel_required}"}
    if fetch:
        _git(repo, "fetch", "-q", "origin", branch)
    commit = _git(repo, "log", "-1", "--format=%H", "--", rel).stdout.strip()
    if not commit:
        return {"ok": False, "reason": "lock file is not committed"}
    blob = _git(repo, "show", f"{commit}:{rel}", text=False)
    if blob.returncode != 0 or blob.stdout != p.read_bytes():
        return {"ok": False, "reason": "working-tree lock differs from its last commit"}
    if _git(repo, "status", "--porcelain", "--", rel).stdout.strip():
        return {"ok": False, "reason": "lock file has uncommitted changes"}
    remotes = _git(repo, "branch", "-r", "--contains", commit).stdout.split()
    if f"origin/{branch}" not in remotes:
        return {"ok": False, "reason": f"lock commit {commit[:12]} is not on origin/{branch}"}
    rb = _git(repo, "show", f"origin/{branch}:{rel}", text=False)
    if rb.returncode != 0 or rb.stdout != p.read_bytes():
        return {"ok": False, "reason": "the lock on origin differs from the working-tree lock"}
    return {"ok": True, "commit": commit, "sha256": _sha(p), "path": str(p), "repo": str(repo), "branch": branch,
            "rel": rel}


def verify_code(lock, repo=WT, chain=CHAIN, check_modules=True):
    """Every CHAIN file listed with a matching hash now (refused otherwise); every loaded worktree module locked."""
    listed = lock.get("locked_code_files") or {}
    missing = [f for f in chain if f not in listed]
    if missing:
        raise SystemExit(f"REFUSED: the lock does not list the scoring-chain code hashes of {missing}")
    rec = {}
    for f, want in sorted(listed.items()):
        have = _sha(Path(repo) / f) if (Path(repo) / f).exists() else None
        if f in chain and have != want:
            raise SystemExit(f"REFUSED: {f} differs from the locked code hash")
        rec[f] = "matches lock" if have == want else ("missing now" if have is None else
                                                      "CHANGED after lock (not in the scoring chain)")
    if check_modules:
        from lra import lock as LK                # S2
        bad = LK.check_loaded_modules(listed)
        if bad:
            raise SystemExit(f"REFUSED: loaded worktree modules are not locked: {bad[:10]}")
        rec["_loaded_modules"] = "every loaded worktree module is locked with its current hash"
    return rec


def open_assessment(lock_path, repo=WT, branch=STUDY_BRANCH, rel_required=LOCK_REL, check_code=True, chain=CHAIN,
                    fetch=True):
    """Verify the pushed lock and the locked code, then open the assessment for this process."""
    global _OPENED
    v = lock_is_pushed(lock_path, repo, branch, rel_required, fetch)
    if not v["ok"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} is not committed and pushed ({v['reason']})")
    lock = json.loads(Path(lock_path).read_text())
    v["code"] = verify_code(lock, repo, chain) if check_code else "not checked (test)"
    v.update({"rel_required": rel_required, "lock": lock, "fetch": fetch})
    _OPENED = v
    return lock


def close_assessment():
    global _OPENED
    _OPENED = None


def _require_open():
    if _OPENED is None:
        raise SystemExit(f"REFUSED: the development assessment is sealed; no verified pushed {LOCK_NAME}")
    v = lock_is_pushed(_OPENED["path"], _OPENED["repo"], _OPENED["branch"], _OPENED["rel_required"],
                       _OPENED.get("fetch", True))
    if not v["ok"] or v["sha256"] != _OPENED["sha256"] or v["commit"] != _OPENED["commit"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} changed or is no longer pushed since it was opened")
    return _OPENED


def load_unsealed():
    """lra.data.load(unseal=True) -- called from this module (lra.data's caller gate) after open_assessment()."""
    opened = _require_open()
    from lra import data as CD                    # S2
    from osf import data as OD
    D = CD.load(unseal=True)
    if D.get("sealed", True):
        raise SystemExit("REFUSED: lra.data did not unseal the assessment labels")
    ar = (opened.get("lock") or {}).get("assessment_role") or {}
    m = OD.manifest(D)[ASSESS]
    for key in ("rows", "groups", "row_id_sha256"):
        if ar.get(key) is not None and m[key] != ar[key]:
            raise SystemExit(f"REFUSED: the assessment {key} differs from the lock")
    return D


# ------------------------------------------------------------------ releases
def safe(label):
    return str(label).replace("*", "star").replace("/", "_").replace("|", "_").replace(" ", "_")


def _unit_files(unit, units_dir=None):
    return json.loads((BL.units_dir(units_dir) / unit / "COMPLETE.json").read_text())["files"]


def _policy_cid_of(u):
    """S4: config id of a code unit name (pol__ / dec__ / new__ s{k}__<cid with | -> _>) through the REGISTERED lra bank
    only, or a registered config id given directly; anything else is REFUSED."""
    reg = AU.registered_composition_bank()
    if "|" in u:
        if u not in reg:
            raise ValueError(f"REFUSED: {u!r} is not a registered code")
        return u
    parts = str(u).split("__", 2)
    if len(parts) != 3 or parts[0] not in AU.CODE_PREFIXES or not parts[1].startswith("s"):
        raise ValueError(f"cannot parse code unit {u!r}")
    k = int(parts[1][1:])
    m = {AU.unit_of(k, c): c for c in reg}
    if u not in m:
        raise ValueError(f"REFUSED: {u!r} is not a unit of the registered code bank")
    return m[u]


def composed_for(k, teacher, seed_entry, D, units_dir=None, check_freeze=True):
    """[(code unit, complete code views, class_only)] of lock composed_policies[teacher] (same teacher and seed only).
    The list must contain every frozen composed winner of the inner source bank (lra.audit.composed_freeze_list) and
    nothing outside the registered composition bank of the teacher (S4)."""
    cp = seed_entry.get("composed_policies") or {}
    bad = [u for u in (cp if isinstance(cp, list) else [x for v in cp.values() for x in v])
           if str(u).startswith(tuple(f"{p}__" for p in AU.CODE_PREFIXES)) and
           not str(u).startswith(tuple(f"{p}__s{k}__" for p in AU.CODE_PREFIXES))]
    if bad:
        raise SystemExit(f"REFUSED: composed policies {bad[:3]} are not units of seed {k}")
    try:
        if isinstance(cp, dict):
            lst = [_policy_cid_of(u) for u in cp.get(teacher, [])]
        else:                                # lra.eval_lock: one list of code units; keep this teacher's codes
            lst = [c for c in (_policy_cid_of(u) for u in cp) if AU.parse_cid(c)["teacher"] == teacher]
    except ValueError as e:                  # S4: an unregistered code / unit is outside the registered bank
        raise SystemExit(f"REFUSED: composed policies outside the registered composition bank ({e})")
    iu = AU.inner_of(k, f"SRC|{teacher}")                                             # S4: aud__tea__...
    locked = seed_entry.get("unit_file_sha256") or {}
    if iu in locked and locked[iu] != _unit_files(iu, units_dir):
        raise SystemExit(f"REFUSED: {iu} (composed winners) differs from the lock")
    reg = [c for c in AU.registered_composition_bank() if AU.parse_cid(c)["teacher"] == teacher]     # S4
    outside = [c for c in lst if c not in reg]
    if outside:
        raise SystemExit(f"REFUSED: composed policies {outside[:3]} are outside the registered composition bank")
    if check_freeze:
        need = AU.composed_freeze_list(k, teacher, units_dir)
        miss = [c for c in need if c not in lst]
        if miss:
            raise SystemExit(f"REFUSED: composed winners {miss} of SRC|{teacher} s{k} are not frozen in the lock")
    out = []
    for cid in lst:
        c = AU.parse_cid(cid)
        if c["kind"] != "policy" or c["teacher"] != teacher:
            raise SystemExit(f"REFUSED: composed policy {cid} is not a policy of teacher {teacher}")
        unit = AU.unit_of(k, cid)
        z, _ = BL.load_unit_npz(unit, "release.npz", units_dir)
        V = AU.lazy_policy_views(z, D, meta={"kind": "policy", "teacher": teacher, "seed": k, "unit": unit,
                                             "config": cid})
        out.append((unit, V, AU.is_class_only(cid)))
    return out


def outer_unit(D, k, label, spec, units_dir=None, memo=None, slate="final", check_freeze=True):
    """Score one frozen release on OSF_DEVELOPMENT_ASSESSMENT (refuses unless the pushed lock was opened and D is
    unsealed). See the module docstring for the unit contents."""
    from jcv.finalize import save_unit, unit_complete
    opened = _require_open()
    if D.get("sealed", True):
        raise SystemExit("REFUSED: D is sealed; use load_unsealed() after open_assessment()")
    root = BL.units_dir(units_dir)
    name = f"outer__s{k}__{safe(label)}"
    if unit_complete(root / name):
        return json.loads((root / name / "record.json").read_text())
    t0, c0 = time.time(), time.process_time()
    seed_entry = (opened.get("lock") or {}).get("seeds", {}).get(str(k), {})
    locked = seed_entry.get("unit_file_sha256") or {}
    kind, cid = spec["kind"], spec["cid"]
    unit = AU.unit_of(k, cid)
    if spec.get("unit") not in (None, unit):
        raise SystemExit(f"REFUSED: spec unit {spec.get('unit')} != {unit} for {cid}")
    if unit in locked and locked[unit] != _unit_files(unit, units_dir):
        raise SystemExit(f"REFUSED: unit {unit} files differ from the hashes recorded in the lock")
    primary, sets, out, prov, _ = AU.view_sets(kind, k, cid, D, units_dir)
    fams = spec.get("families") or list(sets)
    if primary not in fams:
        raise SystemExit("REFUSED: the primary family must be scored")
    u_label = seed_entry.get("u_label", "SRC|U")
    u_t, u_prov = BL.load_teacher(AU.parse_cid(u_label)["teacher"], k, D, units_dir)
    u_out = BL.source_outputs(u_t)
    if f"tea__s{k}__U" in locked and locked[f"tea__s{k}__U"] != _unit_files(f"tea__s{k}__U", units_dir):
        raise SystemExit("REFUSED: the U anchor unit differs from the lock")
    composed, comp_meta = [], {}
    if kind == "source":
        teacher = AU.parse_cid(cid)["teacher"]
        composed = composed_for(k, teacher, seed_entry, D, units_dir, check_freeze)
        comp_meta = {"teacher": teacher, "units": [u for u, _, _ in composed],
                     "class_only_units": [u for u, _, co in composed if co]}
    a = np.asarray(D["idx"][ASSESS])
    S = np.asarray(D["sex"])
    memo = {} if memo is None else memo
    fam_rec, preds = {}, {}
    for fam in fams:
        comp = [(u, v) for u, v, co in composed if fam != "decisions" or co]
        if kind == "policy" and fam == "token":          # L4b: the token-only diagnostic uses the cell readers only
            r, P = AU.final_audit_cells(sets[fam], D, a)
        else:
            r, P = DA.final_audit(sets[fam], D, a, composed=comp, memo=memo, slate=slate)
        fam_rec[fam] = DA._jsonable(r)
        for key, arr in P.items():
            crit, w = key.split("_", 1)
            preds[f"P_{crit}_{w}" if fam == primary else f"P_{crit}_{fam}_{w}"] = arr
    util = UT.release_utility(out, D, ASSESS, u_out, procedure="assessment")
    yI, yO = np.asarray(D["y"]["income"])[a], np.asarray(D["y"]["occupation_group"])[a]
    cst = np.array([UT.constant_class(D, 0), UT.constant_class(D, 1)], dtype=np.int64)
    preds.update({"assess_row_id": np.asarray(D["row_id"])[a], "assess_unit": np.asarray(D["unit"])[a], "sex": S[a],
                  "race": np.asarray(D["race"])[a] if "race" in D else np.full(len(a), -1),
                  "y_income": yI, "y_occ": yO, "const_class": cst,
                  "const1": np.full(len(a), cst[0], dtype=np.int64), "const2": np.full(len(a), cst[1], dtype=np.int64)})
    for i, y in ((1, yI), (2, yO)):
        for pre, src in (("", out), ("u_", u_out)):
            Pr = np.asarray(src[f"p{i}"], dtype=np.float64)[a]
            pr = UT.per_row(Pr, y)
            preds.update({f"{pre}hard{i}": np.asarray(src[f"hard{i}"])[a], f"{pre}prob{i}": Pr,
                          f"{pre}ll{i}": pr["ll"], f"{pre}br{i}": pr["br"]})
    dp = None
    if kind == "policy":
        dp = UT.decision_preservation(out, u_t if AU.parse_cid(cid)["teacher"] == "U" else
                                      BL.load_teacher(AU.parse_cid(cid)["teacher"], k, D, units_dir)[0], D)
    rec = {"unit": name, "seed": k, "label": label, "spec": spec, "u_label": u_label,
           "release_provenance": prov, "u_release_provenance": u_prov,
           "evaluation_lock": {"commit": opened["commit"], "sha256": opened["sha256"]},
           "roles": {"attacker_fit": DA.FIT_ROLE, "attacker_selection": DA.SEL_ROLE, "scored": ASSESS},
           "primary_family": primary, "families": fam_rec, "composed": comp_meta, "utility": DA._jsonable(util),
           "decision_preservation": DA._jsonable(dp), "n_assessment": int(len(a)),
           "n_assessment_groups": int(len(np.unique(np.asarray(D["unit"])[a]))), "slate": slate,
           "preds_keys": sorted(preds), "wall_s": round(time.time() - t0, 1),
           "cpu_s": round(time.process_time() - c0, 1)}
    rec = DA._jsonable(rec)
    bad = AU._nonfinite_paths(rec)                # S6: new JSON is finite (a nonfinite value is a technical failure)
    bad += [f"preds.{x}" for x, v in preds.items() if np.asarray(v).dtype.kind == "f" and
            not np.all(np.isfinite(np.asarray(v)))]
    if bad:
        raise RuntimeError(f"TECHNICAL FAILURE: nonfinite values in {name}: {bad[:5]}")
    save_unit(root / name, {"preds.npz": lambda p: np.savez_compressed(p, **preds)}, rec)
    return json.loads((root / name / "record.json").read_text())


def jobs_from_lock(lock, seeds=(0, 1, 2), labels=None, shard=None):
    """(seed, label, spec) of the LOCKED score list only, optionally one shard i/n of it."""
    jobs = [(k, lb, sp) for k in seeds for lb, sp in lock["seeds"][str(k)].get("score", {}).items()
            if not labels or lb in labels]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        jobs = [j for t, j in enumerate(jobs) if t % n == i]
    return jobs


def load_outer(k, label, units_dir=None):
    """(record, preds) of a completed outer unit (hash-complete), for lra.infer."""
    from jcv.finalize import unit_complete
    d = BL.units_dir(units_dir) / f"outer__s{k}__{safe(label)}"
    if not unit_complete(d):
        raise SystemExit(f"REFUSED: {d.name} is missing or not hash-complete")
    z = np.load(d / "preds.npz", allow_pickle=False)
    return json.loads((d / "record.json").read_text()), {x: z[x] for x in z.files}


# ------------------------------------------------------------------ restore hook (F's closeout)
def _views_of_unit(units_root, unit, D, family):
    root = Path(units_root)
    if unit.startswith(tuple(f"{p}__" for p in AU.CODE_PREFIXES)):                         # S7: pol__ / dec__ / new__
        z = np.load(root / unit / "release.npz", allow_pickle=False)
        sets = AU.code_view_sets({k: z[k] for k in z.files}, D)
        return sets[family if family in sets else AU.PRIMARY_CODE_FAMILY]
    if unit.startswith("tea__"):
        z = np.load(root / unit / "teacher.npz", allow_pickle=False)
        return BL.source_view_sets({k: z[k] for k in z.files}, D, families=(family,))[family]
    if unit.startswith("ref__"):
        z = np.load(root / unit / "reference.npz", allow_pickle=False)
        label, k = unit.split("__")[2], int(unit.split("__")[1][1:])
        return BL.reference_view_sets(label, k, D, arrays={x: z[x] for x in z.files})[family]
    raise ValueError(f"unknown release unit {unit!r}")


def refit_selected_attacker(units_root, D, release_unit, outer_unit, view="pair", attacker_seed=None, crit="auc",
                            family=None, slate="final"):
    """Closeout restore hook: from <units_root>/<outer_unit>/record.json take the `crit`-selected attacker of `view`
    (primary family unless `family`), rebuild its source view from the backed-up release unit (or, for a composed
    winner, from the composed code unit named in its candidate), refit it on AUDIT_FIT at the attacker seed(s) exactly
    as dpc.audit.final_audit does (predict on INNER_SELECTION + assessment rows, keep the assessment rows; cell readers
    recomputed from their deterministic counts and frozen INNER fallback rule), and return [P(S=0), P(S=1)] on the
    assessment rows: (3, n, 2) for attacker_seed=None, else (n, 2) -- compared bitwise with preds.npz P_<crit>_<view>.
    D may be sealed: only AUDIT_FIT / INNER_SELECTION SEX is read; no assessment label is read."""
    root = Path(units_root)
    rec = json.loads((root / outer_unit / "record.json").read_text())
    fam = family or rec["primary_family"]
    sc = rec["families"][fam]["scored"][view][crit]
    cand, src_view, att = sc["candidate"], sc["source_view"], sc["attacker"]
    unit = release_unit
    if cand.startswith("composed["):
        unit = cand[len("composed["):cand.index("]")]
        V = _views_of_unit(root, unit, D, AU.PRIMARY_CODE_FAMILY)       # composed readers use the complete code view
    else:
        V = _views_of_unit(root, unit, D, fam)
    yy = np.asarray(D["sex"])
    fit_idx, sel_idx = DA.roles(D)
    a = np.asarray(D["idx"][ASSESS])
    pred_idx = np.concatenate([sel_idx, a])
    sel_pos, score_pos = np.arange(len(sel_idx)), np.arange(len(sel_idx), len(pred_idx))
    seeds = DA.ATT_SEEDS if attacker_seed is None else (attacker_seed,)
    out = []
    for sd in seeds:
        if str(att).startswith("CC"):
            toks = V["tokens"]
            if src_view == "pair":
                P, _ = DA.cc_pair(toks["v1"], toks["v2"], yy, fit_idx, pred_idx, sel_pos)
            else:
                P, _ = DA.cc_local(toks[src_view], yy, fit_idx, pred_idx)
            p1 = P[att][score_pos]
        else:
            X = np.asarray(V["X"][src_view], dtype=np.float64)
            m = dict(DA.SLATES[slate]())[att](sd).fit(X[fit_idx], yy[fit_idx])
            p1 = DA._p1(m, X[pred_idx])[score_pos]
        out.append(np.stack([1.0 - p1, p1], 1))
    return np.stack(out) if attacker_seed is None else out[0]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--shard", default=None, help="i/n over the locked (seed, label) list")
    ap.add_argument("--units-dir", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    L = open_assessment(a.evaluation_lock)
    D = load_unsealed()
    for k, label, spec in jobs_from_lock(L, a.seeds, a.labels, a.shard):
        r = outer_unit(D, k, label, spec, a.units_dir)            # memo per unit (bounded memory)
        pf = r["primary_family"]
        print(r["unit"], {w: round(r["families"][pf]["scored"][w]["auc"]["auc_mean"], 4) for w in DA.PRIMARY_VIEWS},
              flush=True)


if __name__ == "__main__":
    main()
