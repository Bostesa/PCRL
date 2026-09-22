# Archive validation and recovery closure repair — 2026-09-22

A task-owned lock key produced the SHA-256 filename `d0a1e406e19a5a47b01bd7db5aabc8b3c1bd75be5145fb4582a7e28201653c66.lock`. The original archive guard rejected the substring `2016` in that checksum even though the file contained only 92 bytes of job metadata. No sealed-year data was accessed.

The repaired guard accepts that substring only within the exact native lock namespace, requires at most 4096 bytes of strict PID/key/UTC JSON, checks that the recognized job key hashes to the filename, and validates the same bytes during inventory, packing and every readback. The general sealed-year, symlink, traversal, checksum and no-overwrite checks remain. Locks are preserved.

A separate closure repair includes all C retry claims, staged artifacts and failed originals in the representative restore, plus every installed C map and its four pinned preparation/finer-table dependencies across anchors. Other nonrepresentative audit trees remain stream-verified. The semantic C verifier must run separately on the original and restored roots, irrespective of the selected nominee.

The original source SHA-256 was `e344ad4b0dcdd3af3cae6ddbd5d27eb6cbe7a300bdcd6410aada5be3014b99e2`; the final source is `b5d7aec24b51d987d5151e6fb8d54ca9a6ccb5bbad1866d392e70e27322a29b5`. Both patches, original observation, synthetic tests and independent review are preserved privately. These repairs change no fitted model, scientific source fingerprint, constraint, selection or inference rule. They were applied after the original A/C fit controllers closed.
