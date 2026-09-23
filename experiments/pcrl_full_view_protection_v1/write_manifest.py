"""Hash only owned public research files; exclude the manifest and handoff."""

import datetime
import hashlib
import json
import os
import tempfile


ROOTS = ("experiments/pcrl_full_view_protection_v1",
         "tests/pcrl_full_view_protection_v1",
         "results/pcrl_full_view_protection_v1")
EXCLUDE = {"MANIFEST.json", "HANDOFF.json", "REMOTE_VERIFICATION.json"}


def main():
    entries = {}
    for root in ROOTS:
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [item for item in subdirs if item != "__pycache__"]
            for name in files:
                if name in EXCLUDE or name.endswith(".pyc"):
                    continue
                path = os.path.join(directory, name)
                with open(path, "rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                entries[path] = {"sha256": digest, "bytes": os.path.getsize(path)}
    result = {"schema": 1,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "branch": "research/pcrl-full-view-protection-v1",
              "base_sha": "5e154e5c4fdaeb23d327a0ebefe838525f1a19cb",
              "protocol_commit": "63f288382",
              "pre_suite_amendment_commit": "e47310785",
              "post_suite_amendment_commit": "6daff016d",
              "sources": {"completed_2016": "5e154e5c4fdaeb23d327a0ebefe838525f1a19cb",
                          "claims_audit": "33124f965861ccfcaa710aff4a56880c5ab3957e",
                          "supervised_2018": "f4bdf4cd5bf74c634feeec50aef78bff249667e4"},
              "files": dict(sorted(entries.items()))}
    path = "results/pcrl_full_view_protection_v1/MANIFEST.json"
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), prefix=".manifest-",
                                 delete=False) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp = handle.name
    os.replace(temp, path)


if __name__ == "__main__":
    main()
