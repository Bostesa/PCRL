"""The single locked assessment on OSF_DEVELOPMENT_ASSESSMENT (audit/baseline owner; adapted from osf.assess).

This is the ONLY dpc module that indexes OSF_DEVELOPMENT_ASSESSMENT rows or reads their labels.

Gate. REFUSES unless results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json is committed, byte-identical
to its last commit in the working tree, and that commit is contained in
origin/research/pcrl-decision-preserving-compression-v1 (`git branch -r --contains`; fetch first). `open_assessment`
also requires lock["locked_code_files"] = {relpath: sha256} to list EVERY file of the scoring chain (CHAIN) with a
matching hash now (other listed files are re-hashed and any change is recorded, not refused). Only after this passes
does `load_unsealed()` call dpc.data.load(unseal=True) -- from this module, which dpc.data's own unseal gate requires,
and which repeats the fetch + byte-identity check on origin. The unsealed role must match lock["assessment_role"]
(rows, groups, row_id_sha256 of osf.data.manifest) when recorded. `outer_unit` refuses a sealed D and re-verifies the
lock (same commit and sha256, still pushed) before every unit; a release unit whose COMPLETE.json file map differs from
lock["seeds"][k]["unit_file_sha256"][unit] (when recorded) is refused.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.assess --evaluation-lock \
        results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json [--seeds 0 1 2] [--labels ...] [--shard i/n]

Lock schema read: lock["seeds"][str(k)] = {"score": {label: spec}, "u_label": "SRC|U", "composed_policies":
{"U": [policy units or cids], "RAW-J_b0.3": [...]}, "unit_file_sha256": {unit: files map} (optional)};
spec = {"kind": "policy" | "source" | "reference", "unit": <unit name>, "cid": <config id>} (+ optional
"families": [subset of the release's families]). ONE source spec scores ALL five source families
(interface = primary, complete, scores, probs, decisions) in one unit; the composed policy code readers of the same
teacher join the interface / complete / scores / probs banks, and only class-only (m = 1) policies join the decisions
bank. lock["assessment_role"] = osf.data.manifest(D)["OSF_DEVELOPMENT_ASSESSMENT"] (re-checked after unsealing).

Per release (dpc.audit.final_audit: attackers FITTED on AUDIT_FIT, SELECTED on INNER_SELECTION over the whole bank,
refit at attacker seeds 0, 1, 2 and scored on the assessment rows; orientation fixed; the scored coalition attacker is
never clamped to a local result): the primary family and every secondary family, with the same FINAL slate + cell
readers on finite views. Utility: dpc.utility.release_utility on the assessment rows with U (u_label) as the anchor.

Unit outer__s{k}__{safe(label)} (safe: "*" -> "star", "/", "|", " " -> "_"). preds.npz (private; assessment rows in D
order; n rows):
  assess_row_id, assess_unit (exact-record group = bootstrap unit), sex, race, y_income, y_occ
  const_class (2,), const1, const2 (n,)               the OSF_DEFENSE_FIT majority-class constant predictions
  hard1, hard2 (n,), prob1 (n,2), prob2 (n,6)          the release's decisions and probabilities (q_i for policies)
  ll1, ll2, br1, br2 (n,)                              per-row true-label log loss (1e-12 clip) and Brier
  u_hard1, u_hard2, u_prob1, u_prob2, u_ll1, u_ll2, u_br1, u_br2     U's anchor on the same rows
  P_auc_{v1,v2,pair}, P_ce_{v1,v2,pair} (3, n, 2)      [P(S=0), P(S=1)] of the AUC- / CE-selected attacker, attacker
                                                       seeds 0, 1, 2, PRIMARY family (score = column 1)
  P_auc_<family>_{w}, P_ce_<family>_{w}                every secondary family
record.json: label, seed, spec, provenance, lock commit / sha256, utility, per family the selected attackers (label,
inner AUC / CE, scored AUC per seed (descriptive; endpoints are recomputed from preds.npz)), coverage and fallback
counts (inner and scored rows), composed policies, preds keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from dpc import audit as AU
from dpc import utility as UT

WT = Path(__file__).resolve().parents[1]
STUDY_BRANCH = "research/pcrl-decision-preserving-compression-v1"
LOCK_NAME = "EVALUATION_LOCK.json"
LOCK_REL = "results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json"
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
CHAIN = ("dpc/assess.py", "dpc/audit.py", "dpc/utility.py", "dpc/baselines.py", "dpc/data.py", "osf/data.py",
         "osf/audit.py", "smf/audit.py", "smf/data.py", "jcv/audit.py", "jcv/finalize.py", "rgj/finalize.py",
         "rgj/data.py")
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


def verify_code(lock, repo=WT, chain=CHAIN):
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
    """dpc.data.load(unseal=True) -- called from this module (dpc.data's caller gate) after open_assessment()."""
    opened = _require_open()
    from dpc import data as DD
    from osf import data as OD
    D = DD.load(unseal=True)
    if D.get("sealed", True):
        raise SystemExit("REFUSED: dpc.data did not unseal the assessment labels")
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
    d = AU.udir(unit, units_dir)
    return json.loads((d / "COMPLETE.json").read_text())["files"]


def _policy_cid_of(u):
    """Config id of a policy unit name pol__s{k}__<cid with | -> _> (or a cid given directly)."""
    if "|" in u:
        return u
    body = u.split("__", 2)[2]
    for t in ("RAW-J_b0.3", "U"):
        if body.startswith(t + "_"):
            rest = body[len(t) + 1:].split("_")
            return "|".join([t] + rest)
    raise ValueError(f"cannot parse policy unit {u!r}")


def composed_for(k, teacher, seed_entry, D, units_dir=None):
    """[(policy unit, code views, class_only)] of lock composed_policies[teacher] (same teacher and seed only)."""
    lst = (seed_entry.get("composed_policies") or {}).get(teacher, [])
    out = []
    for u in lst:
        cid = _policy_cid_of(u)
        c = AU.parse_cid(cid)
        if c["kind"] != "policy" or c["teacher"] != teacher:
            raise SystemExit(f"REFUSED: composed policy {u} is not a policy of teacher {teacher}")
        unit = AU.unit_of(k, cid)
        _, sets, _, _ = AU.view_sets("policy", k, cid, D, units_dir)
        out.append((unit, sets["code"], c["m"] == 1))
    return out


def outer_unit(D, k, label, spec, units_dir=None, memo=None):
    """Score one frozen release on OSF_DEVELOPMENT_ASSESSMENT (refuses unless the pushed lock was opened and D is
    unsealed). See the module docstring for the unit contents."""
    from rgj import finalize as FN
    opened = _require_open()
    if D.get("sealed", True):
        raise SystemExit("REFUSED: D is sealed; use load_unsealed() after open_assessment()")
    name = f"outer__s{k}__{safe(label)}"
    if FN.unit_complete(AU.udir(name, units_dir)):
        return json.loads((AU.udir(name, units_dir) / "record.json").read_text())
    t0, c0 = time.time(), time.process_time()
    seed_entry = (opened.get("lock") or {}).get("seeds", {}).get(str(k), {})
    locked = seed_entry.get("unit_file_sha256") or {}
    kind, cid = spec["kind"], spec["cid"]
    unit = AU.unit_of(k, cid)
    if spec.get("unit") not in (None, unit):
        raise SystemExit(f"REFUSED: spec unit {spec.get('unit')} != {unit} for {cid}")
    if unit in locked and locked[unit] != _unit_files(unit, units_dir):
        raise SystemExit(f"REFUSED: unit {unit} files differ from the hashes recorded in the lock")
    primary, sets, out, prov = AU.view_sets(kind, k, cid, D, units_dir)
    prov = {kk: v for kk, v in prov.items() if kk != "z"}
    fams = spec.get("families") or list(sets)
    if primary not in fams:
        raise SystemExit("REFUSED: the primary family must be scored")
    u_label = seed_entry.get("u_label", "SRC|U")
    u_spec = (seed_entry.get("score") or {}).get(u_label) or {"kind": "source", "cid": u_label}
    _, _, u_out, u_prov = AU.view_sets(u_spec["kind"], k, u_spec["cid"], D, units_dir)
    composed, comp_meta = [], {}
    if kind == "source":
        teacher = AU.parse_cid(cid)["teacher"]
        composed = composed_for(k, teacher, seed_entry, D, units_dir)
        comp_meta = {"teacher": teacher, "units": [u for u, _, _ in composed],
                     "class_only_units": [u for u, _, co in composed if co]}
    a = np.asarray(D["idx"][ASSESS])
    S = np.asarray(D["sex"])
    memo = {} if memo is None else memo
    fam_rec, preds = {}, {}
    for fam in fams:
        comp = [(u, v) for u, v, co in composed if fam != "decisions" or co]
        r, P = AU.final_audit(sets[fam], D, a, composed=comp, memo=memo)
        fam_rec[fam] = AU._jsonable(r)
        for key, arr in P.items():
            crit, w = key.split("_", 1)
            preds[f"P_{crit}_{w}" if fam == primary else f"P_{crit}_{fam}_{w}"] = arr
    cst, prior = UT.constants(D)
    util = UT.release_utility(out, D, a, u_out)
    yI, yO = np.asarray(D["y"]["income"])[a], np.asarray(D["y"]["occupation_group"])[a]
    preds.update({"assess_row_id": np.asarray(D["row_id"])[a], "assess_unit": np.asarray(D["unit"])[a], "sex": S[a],
                  "race": np.asarray(D["race"])[a], "y_income": yI, "y_occ": yO,
                  "const_class": np.array([cst[0], cst[1]], dtype=np.int64),
                  "const1": np.full(len(a), cst[0], dtype=np.int64), "const2": np.full(len(a), cst[1], dtype=np.int64)})
    for i, y in ((1, yI), (2, yO)):
        for pre, src in (("", out), ("u_", u_out)):
            Pr = np.asarray(src[f"p{i}"], dtype=np.float64)[a]
            pr = UT.per_row(Pr, y)
            preds.update({f"{pre}hard{i}": np.asarray(src[f"hard{i}"])[a], f"{pre}prob{i}": Pr,
                          f"{pre}ll{i}": pr["ll"], f"{pre}br{i}": pr["br"]})
    rec = {"unit": name, "seed": k, "label": label, "spec": spec, "u_label": u_label, "release_provenance": prov,
           "u_release_provenance": {kk: v for kk, v in u_prov.items() if kk != "z"},
           "evaluation_lock": {"commit": opened["commit"], "sha256": opened["sha256"], "code": opened.get("code")},
           "roles": {"attacker_fit": AU.FIT_ROLE, "attacker_selection": AU.SEL_ROLE, "scored": ASSESS},
           "primary_family": primary, "families": fam_rec, "composed": comp_meta, "utility": util,
           "n_assessment": int(len(a)), "n_assessment_groups": int(len(np.unique(np.asarray(D["unit"])[a]))),
           "preds_keys": sorted(preds), "wall_s": round(time.time() - t0, 1),
           "cpu_s": round(time.process_time() - c0, 1)}
    FN.save_unit(AU.udir(name, units_dir), {"preds.npz": lambda p: np.savez_compressed(p, **preds)}, rec)
    return json.loads((AU.udir(name, units_dir) / "record.json").read_text())


def jobs_from_lock(lock, seeds=(0, 1, 2), labels=None, shard=None):
    jobs = [(k, lb, sp) for k in seeds for lb, sp in lock["seeds"][str(k)].get("score", {}).items()
            if not labels or lb in labels]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        jobs = [j for t, j in enumerate(jobs) if t % n == i]
    return jobs


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--shard", default=None, help="i/n over the (seed, label) list")
    ap.add_argument("--units-dir", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    L = open_assessment(a.evaluation_lock)
    D = load_unsealed()
    for k, label, spec in jobs_from_lock(L, a.seeds, a.labels, a.shard):
        r = outer_unit(D, k, label, spec, a.units_dir)            # memo per unit (bounded memory)
        pf = r["primary_family"]
        print(r["unit"], {w: round(r["families"][pf]["scored"][w]["auc"]["auc_mean"], 4) for w in AU.PRIMARY_VIEWS},
              flush=True)


if __name__ == "__main__":
    main()
