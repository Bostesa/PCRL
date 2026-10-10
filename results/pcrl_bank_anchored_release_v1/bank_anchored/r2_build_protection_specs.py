"""Build byte-pinned protection queue specs without opening role arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
PACKAGE = HERE.parents[1]
R2 = PACKAGE / "r2"
OWN = (
    "r2_laws", "r2_fit_core", "r2_factor_projection", "r2_sequential",
    "r2_source_deterministic", "r2_source_fit_product",
    "r2_source_fit_local", "r2_source_seq_train", "r2_dispatch",
    "r2_package", "r2_materialize", "r2_worker", "r2_build_protection_specs",
)
SOURCE = (
    "r01/role_store", "r01/tstar_source",
    "sampled_channel_v1/impl/bank", "sampled_channel_v1/impl/channel",
    "sampled_channel_v1/impl/fit", "sampled_channel_v1/impl/release",
    "sampled_channel_v1/impl/source_bridge",
    "sampled_channel_v1/controls/control_dispatch",
    "sampled_channel_v1/controls/deterministic_policy",
    "sampled_channel_v1/controls/deterministic_trainer",
    "sampled_channel_v1/controls/factor_controls",
    "sampled_channel_v1/controls/fit_local",
    "sampled_channel_v1/controls/fit_product",
    "sampled_channel_v1/controls/seq_train",
    "sampled_channel_v1/controls/task_rate",
)
ARMS = ("B0", "T", "D", "P", "L", "J", "S12", "S21")
INDEX = R2 / "R2_PROTECTION_SOURCE_INDEX.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise PermissionError("source is not a regular file")
    return {"path": str(path), "sha256": sha(path)}


def code_inventory():
    names = ([f"results/pcrl_bank_anchored_release_v1/bank_anchored/{v}.py"
              for v in OWN] +
             [f"results/pcrl_latent_and_sampled_methods_v1/{v}.py"
              for v in SOURCE] +
             ["results/pcrl_bank_anchored_release_v1/r0/compute_queue.py"])
    return [{"relative_path": name, "sha256": sha(REPO / name)} for name in names]


def write_index():
    if INDEX.exists():
        raise FileExistsError("source index is write-once")
    inventory = code_inventory()
    out = {"schema": "BAR1-r2-protection-source-index-v1",
           "source_lock_sha256": sha(PACKAGE / "CAMPAIGN_LOCK.json"),
           "unit_ids": [f"R2_{arm}_S{seed}_FIT" for arm in ARMS for seed in (0, 1)],
           "recipe_seeds": {f"R2_{arm}_S{seed}_FIT": 2026101200 +
                            100*seed + ARMS.index(arm)
                            for arm in ARMS for seed in (0, 1)},
           "whole_arm_cpu_ceiling_s": 300,
           "code": inventory,
           "builder_command": "python -m results.pcrl_bank_anchored_release_v1.bank_anchored.r2_build_protection_specs --amendment PATH --role-manifest PATH --portfolio-manifest PATH --source-audit PATH --stream-roots PATH"}
    INDEX.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    return out


def write_private(path, payload):
    if path.exists() or path.is_symlink():
        raise FileExistsError("protection config/spec is write-once")
    if stat.S_IMODE(path.parent.stat().st_mode) & 0o077:
        raise PermissionError("private spec parent is not owner-only")
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)
        stream.flush(); os.fsync(stream.fileno())
    return sha(path)


def build(args):
    index = json.loads(INDEX.read_text())
    if (index.get("schema") != "BAR1-r2-protection-source-index-v1" or
            index["code"] != code_inventory() or
            index["source_lock_sha256"] != sha(PACKAGE / "CAMPAIGN_LOCK.json")):
        raise PermissionError("R2 source index differs before protection specs")
    amendment = Path(args.amendment).resolve(strict=True)
    role_manifest = Path(args.role_manifest).resolve(strict=True)
    portfolio_manifest = Path(args.portfolio_manifest).resolve(strict=True)
    source_audit = Path(args.source_audit).resolve(strict=True)
    roots = Path(args.stream_roots).resolve(strict=True)
    stream_manifest = REPO / "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
    role = json.loads(role_manifest.read_text())
    private_root = role_manifest.parent.parent
    outdir = R2 / ".private"
    units = outdir / "units"
    if not outdir.is_dir() or stat.S_IMODE(outdir.stat().st_mode) & 0o077:
        raise PermissionError("R2 private custody root is missing or public")
    units.mkdir(mode=0o700, exist_ok=True)
    if stat.S_IMODE(units.stat().st_mode) & 0o077:
        raise PermissionError("R2 unit custody root is not owner-only")
    code = [record(REPO / v["relative_path"]) for v in index["code"]]
    created = {}
    for arm in ARMS:
        for seed in (0, 1):
            unit = f"R2_{arm}_S{seed}_FIT"
            identity = R2 / f"R2_FULL_FIT_IDENTITY_S{seed}_RECEIPT.json"
            pins = {"campaign_lock": PACKAGE / "CAMPAIGN_LOCK.json",
                    "protection_amendment": amendment,
                    "role_manifest": role_manifest,
                    "portfolio_manifest": portfolio_manifest,
                    "source_audit": source_audit,
                    "stream_roots": roots,
                    "stream_manifest": stream_manifest,
                    "identity_receipt": identity,
                    "worker_source": PACKAGE / "bank_anchored/r2_worker.py"}
            refs = [("OSF_DEFENSE_FIT", f"references_s{seed}"),
                    ("OSF_DEFENSE_FIT", "income"),
                    ("OSF_DEFENSE_FIT", "occupation"),
                    ("ATTACK_FIT_NEW", f"references_s{seed}"),
                    ("INNER_SELECTION", f"references_s{seed}")]
            if arm not in ("B0", "T"):
                refs.append(("OSF_DEFENSE_FIT", "sex"))
            role_files = [private_root / role["role_file_relative_path"][f"{r}:{field}"]
                          for r, field in refs]
            source_root = portfolio_manifest.parent / \
                f"shared/.private/tstar_score_bank_s{seed}_v1"
            source_files = [source_root / name for name in
                            ("BANK_ADMISSION_RECEIPT.json", "bank_r1.npz",
                             "bank_r2.npz", "tokens_OSF_DEFENSE_FIT.npz",
                             "tokens_ATTACK_FIT_NEW.npz",
                             "tokens_INNER_SELECTION.npz")]
            fit_receipt = R2 / f"{unit}_FIT_RECEIPT.json"
            unit_receipt = R2 / f"{unit}_UNIT_RECEIPT.json"
            config = {"schema": "BAR1-r2-protection-job-v1", "unit_id": unit,
                      "arm_id": arm, "source_seed": seed,
                      "recipe_seed": index["recipe_seeds"][unit],
                      "cpu_seconds": 300,
                      "source_lock_sha256": index["source_lock_sha256"],
                      "protection_amendment_sha256": sha(amendment),
                      "pins": {k: record(v) for k, v in pins.items()},
                      "private_output_dir": str(units / unit),
                      "fit_receipt": str(fit_receipt),
                      "unit_receipt": str(unit_receipt)}
            config_path = outdir / f"{unit}_CONFIG.json"
            config_sha = write_private(config_path, config)
            seen = {}
            for item in [record(config_path), *code,
                         *(record(v) for v in pins.values()),
                         *(record(v) for v in role_files),
                         *(record(v) for v in source_files)]:
                seen[item["path"]] = item
            spec = {"unit_id": unit, "stage": "protection",
                    "argv": [str(Path(args.python).resolve(strict=True)), "-m",
                             "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_worker",
                             "--config", str(config_path)],
                    "cwd": str(REPO), "env": {}, "max_wall_s": 600,
                    "cpu_reserve_s": 600, "max_rss_gib": 4.0,
                    "projected_output_bytes": 134217728,
                    "deps": [f"R2_FULL_FIT_IDENTITY_S{seed}"],
                    "outputs": [str(fit_receipt), str(unit_receipt)],
                    "sources": list(seen.values()),
                    "amendment_sha256": sha(amendment)}
            spec_path = outdir / f"{unit}_QUEUE_SPEC.json"
            spec_sha = write_private(spec_path, spec)
            created[unit] = {"config_sha256": config_sha,
                             "queue_spec_sha256": spec_sha}
    summary = {"schema": "BAR1-r2-protection-spec-receipt-v1",
               "source_index_sha256": sha(INDEX),
               "amendment_sha256": sha(amendment),
               "unit_sha256": created}
    summary_path = R2 / "R2_PROTECTION_SPEC_RECEIPT.json"
    if summary_path.exists():
        raise FileExistsError("protection spec receipt is write-once")
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--portfolio-manifest", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--stream-roots", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.index_only:
        out = write_index()
    else:
        if any(getattr(args, k) is None for k in
               ("amendment", "role_manifest", "portfolio_manifest",
                "source_audit", "stream_roots")):
            parser.error("all source paths and amendment are required")
        out = build(args)
    print(json.dumps({"schema": out["schema"], "units":
                      out.get("unit_ids", list(out.get("unit_sha256", {})))},
                     sort_keys=True))


if __name__ == "__main__":
    main()
