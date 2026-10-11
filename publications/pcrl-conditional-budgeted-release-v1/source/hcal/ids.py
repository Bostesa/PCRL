"""Shared paths and configuration IDs of the held-out calibration study (hcal). No data is loaded here.

PRIVATE STORE. Every private artifact lives under the task-specific cache named by the environment variable
PCRL_HCAL_PRIVATE_CACHE (default <HOME>/PCRL_eval_cache_private/hcal_v1; HOME itself is never repurposed). Tracked files
refer to it as <PRIVATE_CACHE>/hcal_v1.

FROZEN BANK (prompt section 6; 57 partition pairs per source seed, 171 partition/seed units):
  legacy (27)  U|DIRECT-TASK|i8o64, U|FINE-TASK|i8o64, U|CLASS|i1o1,
               U|{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{0.01,0.025,0.04,0.06,0.08,0.1}   (lambda-major, family-minor)
  lra (30)     U|C-TASK|i8o64, U|W-{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{lambda} (same order),
               U|K-{LOCAL,SEQ-12,SEQ-21,JOINT-SINGLE,JOINT-PAIR}|i8o64
ORIGINAL RELEASES (84 per seed; admitted unchanged, lra IDs): legacy D0 = <partition> (27), legacy D1 = <partition>|D1
(27), lra fits = <partition>|D1 (30). Their tokens are the partition's tokens; only the decoded vectors differ.
DECODER VARIANTS (release ID = <partition>|<decoder>; the partition's tokens and decisions are unchanged):
  D0             legacy original mean-teacher decoder (admitted; the legacy TEACHER-MEAN decoder is this, bitwise)
  D1             original lra learned decoder (kappa 32 on OSF_DEFENSE_FIT labels; admitted, never overwritten)
  MEAN           TEACHER-MEAN decoder of an lra partition (newly evaluated mean-decoded control; never a nominee)
  H-TOKEN32      per-token kappa-32 objective on CALIBRATION_HELDOUT task labels (fixed teacher prior mu_t)
  T-TOKEN32      the same objective on CALIBRATION_TRAIN_MATCHED (4 fixed diagnostic partitions only; diagnostic)
  H-GLOBAL-TEMP  one inverse temperature per task, map and seed on CALIBRATION_HELDOUT
  H-CLASS-TEMP   one inverse temperature per predicted class (>= 50 calibration representatives, else alpha = 1)
CONTINUOUS U: SRC|U (identity, U0), SRC|U|H-GLOBAL-TEMP, SRC|U|H-CLASS-TEMP. References (descriptive): SRC|RAW-J_b0.3,
REF|E, REF|F, REF|F0.
"""
from __future__ import annotations

import os
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_heldout_calibration_v1"
PKG = WT / REL
BRANCH = "research/pcrl-heldout-calibration-v1"
STUDY = "pcrl_heldout_calibration_v1"
PRIV = Path(os.environ.get("PCRL_HCAL_PRIVATE_CACHE") or (Path.home() / "PCRL_eval_cache_private" / "hcal_v1"))
RUN = PRIV / "run"
UNITS = RUN / "units"
ADM = PRIV / "admitted"
ADM_UNITS = ADM / "units"          # verified copies of the lra units (never written after admission)
LRA_PRIV = Path.home() / "PCRL_eval_cache_private" / "lra_v1"
LRA_COPY = Path.home() / "PCRL_eval_cache_private" / "lra_v1_local_copy_20261007"

SOURCE_TIP = "976202546a100f2b89a863cb3785c3465cf76c6a"       # lra final tip (worktree base)
SOURCE_EVIDENCE = "1dee33253af03bb5efcaa652774cb222df642aa3"  # lra evidence commit (resolved full SHA)
SOURCE_REL = "results/pcrl_adult_learned_decoder_release_v1"

SEEDS = (0, 1, 2)
TASKS = ("income", "occupation")
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
CONSTRAINED = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT-SINGLE", "JOINT-PAIR")
NEW_DECODERS = ("H-TOKEN32", "H-GLOBAL-TEMP", "H-CLASS-TEMP")
DIAG_DECODER = "T-TOKEN32"
ALL_DECODERS = ("D0", "D1", "MEAN") + NEW_DECODERS + (DIAG_DECODER,)
U_DECODERS = ("identity", "H-GLOBAL-TEMP", "H-CLASS-TEMP")
U_ID = "SRC|U"
REFERENCES = ("SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0")
DIAGNOSTIC_PARTITIONS = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1", "U|JOINT|i8o64|l0.1")
PRIMARY_DIAGNOSTIC_PARTITION = "U|JOINT|i8o64|l0.1"
TASK_ONLY_PARTITIONS = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1", "U|C-TASK|i8o64")


def g(x):
    return f"{x:g}"


def legacy_partitions():
    return (["U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1"]
            + [f"U|{f}|i8o64|l{g(lam)}" for lam in LAMS for f in PRIVACY])


def lra_partitions():
    return (["U|C-TASK|i8o64"] + [f"U|W-{f}|i8o64|l{g(lam)}" for lam in LAMS for f in PRIVACY]
            + [f"U|K-{a}|i8o64" for a in CONSTRAINED])


def partitions():
    """The 57 frozen partition IDs of one seed, in registered order (legacy first)."""
    return legacy_partitions() + lra_partitions()


def is_legacy(p):
    return p in legacy_partitions()


def privacy_trained(p):
    """True iff the partition's assignment search used SEX (legacy privacy families, W- and K- families)."""
    fam = p.split("|")[1]
    return fam in PRIVACY or fam.startswith(("W-", "K-"))


def task_only(p):
    return p in TASK_ONLY_PARTITIONS


def decoders_of(p):
    """Registered decoder variants of a partition, in registered (bank) order."""
    base = ["D0", "D1"] if is_legacy(p) else ["D1", "MEAN"]
    out = base + list(NEW_DECODERS)
    if p in DIAGNOSTIC_PARTITIONS:
        out.append(DIAG_DECODER)
    return out


def release_id(p, dec):
    """Release ID of (partition, decoder). Original releases keep their lra IDs."""
    if dec == "D0":
        assert is_legacy(p)
        return p
    if dec == "D1":
        return f"{p}|D1"
    return f"{p}|{dec}"


def parse_release(rid):
    """(partition, decoder) of a code release ID, or (None, U decoder) for continuous U."""
    if rid == U_ID:
        return None, "identity"
    if rid.startswith(U_ID + "|"):
        return None, rid[len(U_ID) + 1:]
    for dec in ALL_DECODERS:
        if dec != "D0" and rid.endswith("|" + dec):
            p = rid[: -len(dec) - 1]
            if p in partitions():
                return p, dec
    if rid in legacy_partitions():
        return rid, "D0"
    raise ValueError(f"unknown release id {rid!r}")


def original_ids():
    """The 84 original release IDs of one seed (lra code_ids order: D0 27, D1 fixed-map 27, lra fits 30)."""
    return legacy_partitions() + [f"{p}|D1" for p in legacy_partitions()] + [f"{p}|D1" for p in lra_partitions()]


def code_release_ids():
    """Every registered code release of one seed (all partitions x all their decoder variants)."""
    return [release_id(p, d) for p in partitions() for d in decoders_of(p)]


def u_release_ids():
    return [U_ID] + [f"{U_ID}|{d}" for d in U_DECODERS[1:]]


def nominee_capable(rid):
    """P* pool membership (prompt section 9): privacy-trained partition with an admitted ORIGINAL decoder (D0 legacy,
    D1 any) or a newly registered HELD-OUT decoder (H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP). MEAN of an lra partition
    is a control and T-TOKEN32 is diagnostic: never nominees."""
    if rid.startswith(("SRC|", "REF|")):
        return False
    p, dec = parse_release(rid)
    return privacy_trained(p) and dec in ("D0", "D1") + NEW_DECODERS


def task_only_candidate(rid):
    """T* pool membership: task-only partitions with every admitted original / mean / held-out / shared variant
    (T-TOKEN32 is diagnostic only), and continuous U with its two shared calibrations."""
    if rid in u_release_ids():
        return True
    if rid.startswith(("SRC|", "REF|")):
        return False
    p, dec = parse_release(rid)
    return task_only(p) and dec != DIAG_DECODER


def safe(x):
    return str(x).replace("|", "_").replace("*", "star").replace("/", "_").replace(" ", "_")


# lra unit names (admitted copies keep their names)
def lra_unit(k, rid):
    p, dec = parse_release(rid)
    if dec == "D0":
        return f"pol__s{k}__{safe(rid)}"
    if dec == "D1":
        return (f"dec__s{k}__{safe(rid)}" if is_legacy(p) else f"new__s{k}__{safe(rid)}")
    raise ValueError(f"{rid} is not an original lra release")


def lra_inner(k, rid):
    return f"aud__{lra_unit(k, rid)}"


def partition_original_units(k, p):
    """Original admitted release units of a partition (bank order: legacy D0 then D1; lra D1)."""
    return [lra_unit(k, release_id(p, d)) for d in (("D0", "D1") if is_legacy(p) else ("D1",))]
