"""Read unchanged 2018 archived kernels and write aggregate radius brackets."""

import argparse
import datetime
import hashlib
import json
import os
import tempfile

import numpy as np

from .finite import information_radius_bracket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.load(open(args.manifest))
    result = {"source": "unchanged 2018 mechanisms only", "manifest": args.manifest,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "anchors": {}}
    for anchor, entry in manifest["anchors"].items():
        result["anchors"][anchor] = {}
        for method in ("Q", "D17"):
            item = entry["maps"][method]["Q.npz"]
            with open(item["path"], "rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"kernel hash mismatch: {anchor}/{method}")
            with np.load(item["path"], allow_pickle=False) as data:
                q = data["Q"]
            bracket = information_radius_bracket(q)
            result["anchors"][anchor][method] = {"file_sha256": digest,
                                                    "shape": list(q.shape), **bracket}
    directory = os.path.dirname(args.output)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".radius-", delete=False) as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temp = f.name
    os.replace(temp, args.output)


if __name__ == "__main__":
    main()
