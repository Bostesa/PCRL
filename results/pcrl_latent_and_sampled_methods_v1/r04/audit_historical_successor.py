"""Verify the sanitized R01 historical audit changes only path identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .independent_a_source import sha_file


DROP = {"path", "receipt_path", "clean_path", "clean_relative_path",
        "source_fine_path", "source_fine_cache_relative_path", "path_roots"}


def _scientific(value):
    if isinstance(value, dict):
        return {key: _scientific(inner) for key, inner in value.items() if key not in DROP}
    if isinstance(value, list):
        return [_scientific(item) for item in value]
    return value


def _rooted(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("sanitized audit path is not rooted relative")
    absolute = (root / path).resolve(strict=True)
    if not absolute.is_relative_to(root.resolve(strict=True)):
        raise ValueError("sanitized audit path escapes its admitted root")
    return absolute


def run(args: argparse.Namespace) -> dict:
    old_path = Path(args.old_audit).resolve(strict=True)
    new_path = Path(args.new_audit).resolve(strict=True)
    amendment_path = Path(args.custody_amendment).resolve(strict=True)
    old_sha, new_sha = sha_file(old_path), sha_file(new_path)
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    if (old_sha != amendment["old_public_audit_sha256"]
            or new_sha != amendment["new_public_audit_sha256"]
            or amendment.get("status") != "PATH_IDENTITY_SANITIZED_SCIENTIFIC_HASHES_UNCHANGED"
            or _scientific(old) != _scientific(new)):
        raise ValueError("historical audit scientific fields changed")
    scientific = json.dumps(_scientific(new), sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False).encode()
    scientific_sha = hashlib.sha256(scientific).hexdigest()
    if scientific_sha != amendment["scientific_fields_canonical_sha256"]:
        raise ValueError("R01 custody scientific canonical digest differs")
    portfolio = Path(args.portfolio_package).resolve(strict=True)
    cache = Path(args.cache_root).resolve(strict=True)
    campaign = Path(args.campaign).resolve(strict=True)
    checked = 0
    for collection in ("pstar", "d0"):
        entries = ([(f"s{seed}", old[collection][f"s{seed}"], new[collection][f"s{seed}"])
                    for seed in (0, 1)] if collection == "pstar" else
                   [(f"{key}:s{seed}", old[collection][key][f"s{seed}"],
                     new[collection][key][f"s{seed}"])
                    for key in sorted(old[collection]) for seed in (0, 1)])
        for _, before, after in entries:
            pairs = [(before["receipt_path"], after["receipt_path"], portfolio,
                      before["receipt_sha256"])]
            for field in ("banks", "roles"):
                pairs.extend((record["path"], after[field][key]["path"], portfolio,
                              record["sha256"])
                             for key, record in before[field].items())
            pairs.extend((record["path"], after["source_paths_sha256"][key]["path"],
                          cache, record["sha256"])
                         for key, record in before["source_paths_sha256"].items())
            for prior, relative, root, expected_sha in pairs:
                resolved = _rooted(root, relative)
                if resolved != Path(prior).resolve(strict=True) or sha_file(resolved) != expected_sha:
                    raise ValueError("sanitized historical path maps to different bytes")
                checked += 1
    for seed in (0, 1):
        before, after = old["public_centroids"][f"s{seed}"], new["public_centroids"][f"s{seed}"]
        for old_key, relative_key, root, digest_key in (
            ("clean_path", "clean_relative_path", campaign, "clean_sha256"),
            ("source_fine_path", "source_fine_cache_relative_path", cache,
             "source_fine_sha256")):
            resolved = _rooted(root, after[relative_key])
            if resolved != Path(before[old_key]).resolve(strict=True) or sha_file(resolved) != before[digest_key]:
                raise ValueError("sanitized public centroid source path differs")
            checked += 1
    for name, change in amendment["fixed_public_map_receipt_sha256_changes"].items():
        receipt_path = campaign / "r01" / name
        npz_path = receipt_path.with_suffix(".npz")
        record = json.loads(receipt_path.read_text())
        if (sha_file(receipt_path) != change["new_sha256"]
                or sha_file(npz_path) != change["npz_sha256_unchanged"]
                or record["npz_sha256"] != change["npz_sha256_unchanged"]
                or record["source_audit_sha256"] != new_sha):
            raise ValueError("sanitized fixed public map bytes differ")
    return {"schema": "pcrl-r04-historical-audit-successor-v1",
            "status": "PATH_ONLY_SUCCESSOR_PARITY_PASS",
            "old_public_audit_sha256": old_sha, "new_public_audit_sha256": new_sha,
            "custody_amendment_sha256": sha_file(amendment_path),
            "scientific_fields_canonical_sha256": scientific_sha,
            "rooted_source_files_checked": checked,
            "fixed_public_map_npz_bytes_unchanged": len(amendment["fixed_public_map_receipt_sha256_changes"]),
            "historical_replay_receipts_preserved": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-audit", "new-audit", "custody-amendment", "portfolio-package",
                 "cache-root", "campaign", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("R04 historical successor receipt exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
