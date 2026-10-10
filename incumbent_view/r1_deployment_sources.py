"""Hash-only inventory for later queued raw83 replay and snapshot admission.

The public inventory uses logical aliases. The private map alone contains
absolute source paths. No person-level archive is opened by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ARMS = ("O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21")
CODE_DIRS = ("qpc", "hcal", "lra", "dpc", "cbp", "jcv", "rgj",
             "results/pcrl_latent_and_sampled_methods_v1/r01",
             "results/pcrl_latent_and_sampled_methods_v1/r02",
             "results/pcrl_latent_and_sampled_methods_v1/r03",
             "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl",
             "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/controls",
             "results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1")
CODE_FILES = ("results/pcrl_bank_anchored_release_v1/bank_anchored/r2_laws.py",
              "results/pcrl_incumbent_view_risk_v1/r0/compute_queue.py")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path: Path, obj: dict, mode: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest(path)


def catalog(repo: Path) -> tuple[list[dict], dict[str, str]]:
    repo = repo.resolve(strict=True)
    root = repo / "results/pcrl_incumbent_view_risk_v1"
    source = root / "r1/.private/P0_SOURCE_PATHS.json"
    original = json.loads(source.read_text())
    entries: dict[str, dict] = {}
    paths: dict[str, str] = {}

    def add(alias: str, path: str | Path, expected: str | None = None) -> None:
        p = Path(path).absolute()
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"missing or symlink source: {alias}")
        sha = digest(p)
        if expected is not None and sha != expected:
            raise ValueError(f"source manifest hash differs: {alias}")
        record = {"alias": alias, "sha256": sha, "size_bytes": p.stat().st_size}
        if alias in entries and (entries[alias] != record or paths[alias] != str(p)):
            raise ValueError(f"duplicate source alias: {alias}")
        entries[alias], paths[alias] = record, str(p)

    add("private/r1_source_manifest.json", source)
    add("private/stream_roots.json", root / "r2/.private/stream_roots_v1.json")
    add("locks/admission.json", root / "r0/ADMISSION_LOCK.json")
    add("locks/campaign.json", root / "r0/CAMPAIGN_LOCK.json")
    reader_index_path = root / "r1/HISTORICAL_PSTAR_READER_INDEX.json"
    add("public/historical_reader_index.json", reader_index_path)
    reader_index = json.loads(reader_index_path.read_text())
    for seed in (0, 1):
        s = original["seeds"][str(seed)]
        for key, pin in sorted(s["pins"].items()):
            if key in ("model", "head0", "head1", "record", "complete"):
                alias = f"frozen/s{seed}/teacher/{Path(pin['path']).name}"
            else:
                alias = f"frozen/s{seed}/{key}/{Path(pin['path']).name}"
            add(alias, pin["path"], pin["sha256"])
        add(f"banks/s{seed}/P0_PUBLIC_BANK.json", s["public_bank"]["path"],
            s["public_bank"]["sha256"])
        role = s["roles"]["OSF_DEFENSE_FIT"]
        add(f"role/s{seed}/FIT_P0_RELEASE.npz", role["path"], role["sha256"])
        sample = root / "r1/.private" / f"R1_P0_S{seed}_ADMISSION/OSF_DEFENSE_FIT/sample_input.npz"
        add(f"role/s{seed}/raw32.npz", sample)
        historical = reader_index["seeds"][str(seed)]["historical"]
        older = repo.parent / "pcrl-bank-anchored-release-v1"
        for key in ("model", "predictions_inner", "view_inner", "reader_receipt"):
            item = historical[key]
            add(f"reader/s{seed}/{key}/{Path(item['alias']).name}",
                older / item["alias"], item["sha256"])
        inner = reader_index["seeds"][str(seed)]["p0_role_only"]["INNER_SELECTION"]
        for key in ("release", "receipt"):
            item = inner[key]
            add(f"reader/s{seed}/p0_inner_{key}/{Path(item['alias']).name}",
                repo / item["alias"], item["sha256"])
        for arm in ARMS:
            unit = f"R2_{arm}_S{seed}_FIT"
            base = root / "r2/.private/units" / unit
            for name in ("policy.npz", "POLICY_RECEIPT.json"):
                add(f"policies/{arm}/s{seed}/{name}", base / "policy" / name)
            for draw in range(3):
                release = base / f"OSF_DEFENSE_FIT_R{draw}"
                for name in ("release_codes.npz", "RELEASE_RECEIPT.json"):
                    add(f"policies/{arm}/s{seed}/draw{draw}/{name}", release / name)
    for directory in CODE_DIRS:
        for path in sorted((repo / directory).rglob("*.py")):
            add("code/" + str(path.relative_to(repo)), path)
    for path in CODE_FILES:
        add("code/" + path, repo / path)
    for path in sorted((repo / "incumbent_view").glob("r[12]*.py")):
        add("code/" + str(path.relative_to(repo)), path)
    for directory in (repo / "results", repo / "results/pcrl_bank_anchored_release_v1",
                      repo / "results/pcrl_latent_and_sampled_methods_v1",
                      repo / "incumbent_view"):
        init = directory / "__init__.py"
        if init.is_file():
            add("code/" + str(init.relative_to(repo)), init)
    return [entries[k] for k in sorted(entries)], paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve(strict=True)
    root = repo / "results/pcrl_incumbent_view_risk_v1/r1"
    records, paths = catalog(repo)
    private = {"schema": "IVR1-r1-deployment-source-map-v1", "version": args.version,
               "entries": [{**record, "path": paths[record["alias"]]} for record in records]}
    private_path = root / ".private" / f"DEPLOYMENT_SOURCES_{args.version}.json"
    private_sha = write_once(private_path, private, 0o600)
    public = {"schema": "IVR1-r1-deployment-source-index-v1", "version": args.version,
              "entries": records, "private_map_sha256": private_sha,
              "selection_status": "unbound; final nominee/control require locked receipts",
              "release_contract": {"seeds": [0, 1], "realizations": [0, 1, 2],
                                   "deterministic_P0": "three exact aliases",
                                   "fixed_joint_diagnostic": "V_J"}}
    public_path = root / f"DEPLOYMENT_SOURCE_INDEX_{args.version}.json"
    public_sha = write_once(public_path, public, 0o644)
    print(json.dumps({"public_index": str(public_path), "public_sha256": public_sha,
                      "private_map": str(private_path), "private_sha256": private_sha,
                      "entry_count": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
