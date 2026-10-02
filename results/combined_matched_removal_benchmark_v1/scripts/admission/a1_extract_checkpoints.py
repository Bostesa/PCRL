"""A1: selective, hash-verified extraction of the Round-4 Adult/HMDA checkpoints (no fitting, no drive writes).

- final.pt seeds 1, 2 per dataset: extracted from fl-PCRL-main-checkpoints.tar into the private staging dir.
  Seed 0 final.pt is already in ~/PCRL_eval_cache_private/checkpoints/ and is only re-hashed.
- best.pt seeds 0-2 per dataset: extracted for the row-provenance fingerprint only (per_seed_results.json
  health/accuracy were computed on the reloaded best.pt over the test split). Never used as a benchmark encoder.
- Every extracted member is sha256-verified against the drive inventory; a mismatch is a hard failure.
- Round-5 (Adult/HMDA) and Round-7 (Diabetes) headline final.pt/best.pt: inventory hash + size only (no extraction).
Admitted finals are copied into ~/PCRL_eval_cache_private/checkpoints/ by a3_lineage.py after lineage passes.
Writes notes/admission/checkpoint_extraction.json (hashes and sizes only).
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission_common import (CKPT_DIR, CKPT_TAR, DATASETS, NOTES, PROV_CKPT, SEEDS, STAGING, drive_retry,  # noqa
                              inventory, sha256_file, write_json)


def main():
    inv = inventory()
    STAGING.mkdir(parents=True, exist_ok=True)
    PROV_CKPT.mkdir(parents=True, exist_ok=True)
    want = {}
    for d in DATASETS:
        for s in SEEDS:
            for f in ("final.pt", "best.pt"):
                m = f"checkpoints/v2_{d}_s{s}/{f}"
                if m not in inv:
                    raise SystemExit(f"inventory has no member {m}")
                want[m] = inv[m]
    def local_copy(m):
        """Existing private copy (admitted final in checkpoints/, provenance best in provenance_ckpt/), if any."""
        d, sk = m.split("/")[1][3:-3], m.split("/")[1][-2:]
        return (CKPT_DIR / f"v2_{d}_{sk}_final.pt") if m.endswith("final.pt") else (PROV_CKPT / f"v2_{d}_{sk}_best.pt")

    to_extract = [m for m in want if not (local_copy(m).exists() and sha256_file(local_copy(m)) == want[m]["sha256"])]
    pending = [m for m in to_extract if not (STAGING / m).exists() or sha256_file(STAGING / m) != want[m]["sha256"]]
    if pending:
        # bsdtar --fast-read: stop once every operand has matched; extracts only these members
        drive_retry(lambda: subprocess.run(["tar", "-x", "-q", "-C", str(STAGING), "-f", str(CKPT_TAR), *pending],
                                           check=True), f"extract {len(pending)} members")
    rec = {"archive": str(CKPT_TAR).replace(str(Path.home()), "~"),
           "inventory": "storage-relocation-20260930/inventories/fl-PCRL-main-checkpoints.json.gz", "members": {}}
    for m, meta in want.items():
        d = m.split("/")[1][3:-3]
        if m in to_extract:
            p = STAGING / m
            got = sha256_file(p)
            src = "extracted to private staging"
        else:
            p = local_copy(m)
            got = sha256_file(p)
            src = (f"existing private copy {str(p).replace(str(Path.home()), '~')} (reused, re-hashed; "
                   + ("s0 extracted 2026-10-01 by the preparation" if "_s0/final" in m else
                      "extracted from the drive by this script 2026-10-02") + ")")
        ok = got == meta["sha256"] and p.stat().st_size == meta["size"]
        if not ok:
            raise SystemExit(f"HASH/SIZE MISMATCH {m}: {got} vs inventory {meta['sha256']}")
        if m.endswith("best.pt"):
            tgt = PROV_CKPT / f"v2_{d}_{m.split('/')[1][-2:]}_best.pt"
            if not tgt.exists() or sha256_file(tgt) != got:
                shutil.copyfile(p, tgt)
            assert sha256_file(tgt) == got
        rec["members"][m] = {"sha256": got, "size": meta["size"], "inventory_sha256_match": ok, "source": src,
                             "inventory_mtime_ns": meta["mtime_ns"]}
    # extension inventory (hash only)
    ext = {}
    for m, meta in sorted(inv.items()):
        for tag in ("v2_adult_ROUND5_s", "v2_hmda_ROUND5_s", "v2_diabetes_ROUND7_s"):
            if m.startswith(f"checkpoints/{tag}") and m.endswith((".pt",)) and m.split("/")[-1] in ("final.pt", "best.pt"):
                ext[m] = {"sha256": meta["sha256"], "size": meta["size"], "mtime_ns": meta["mtime_ns"],
                          "extracted": False}
    rec["extension_inventory_round5_round7"] = ext
    write_json(NOTES / "checkpoint_extraction.json", rec)
    print(f"verified {len(rec['members'])} Round-4 members; {len(ext)} extension members inventoried")


if __name__ == "__main__":
    main()
