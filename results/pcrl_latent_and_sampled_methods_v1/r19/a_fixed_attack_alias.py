"""Join corrected A attack attempts to original scientific cells without erasure."""

from queue_unit_status import SHA256, UNIT_ID


ATTEMPT_FIELDS = ("logical_unit_id", "physical_unit_id", "attempt_kind", "attempt_status",
                  "charged_cpu_s", "wall_s", "sampled_tree_rss_peak_bytes",
                  "queue_receipt_sha256", "attack_index_sha256", "controls_status")


def _valid_id(value):
    return isinstance(value, str) and bool(UNIT_ID.fullmatch(value)) and ".." not in value


def _valid_hash(value):
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def _attempt(logical_id, physical_id, kind, queue_row, attack_index="", controls=""):
    return {"logical_unit_id": logical_id, "physical_unit_id": physical_id,
            "attempt_kind": kind, "attempt_status": queue_row["status"],
            "charged_cpu_s": queue_row["charged_cpu_s"], "wall_s": queue_row["wall_s"],
            "sampled_tree_rss_peak_bytes": queue_row["sampled_tree_rss_peak_bytes"],
            "queue_receipt_sha256": queue_row["receipt_sha256"],
            "attack_index_sha256": attack_index, "controls_status": controls}


def attach_aliases(logical_rows, queued, correction, completion, output_hashes,
                   *, expected_aliases=12, expected_cells=18,
                   expected_old_failed=2, expected_old_unqueued=10):
    """Return logical rows and separate physical attempts from hash-only evidence."""
    if completion.get("status") != "ALL_18_SCIENTIFIC_CELLS_HAVE_COMPLETE_PHYSICAL_ATTACK":
        raise ValueError("fixed attack completion status mismatch")
    cells = completion.get("cells")
    mapping = correction.get("physical_attempts")
    if (not isinstance(cells, list) or len(cells) != expected_cells or
            not isinstance(mapping, dict) or len(mapping) != expected_aliases or
            completion.get("scientific_cell_count") != expected_cells or
            completion.get("corrected_physical_done_count") != expected_aliases or
            completion.get("original_physical_done_count") != expected_cells - expected_aliases or
            completion.get("original_failed_preserved_count") != expected_old_failed or
            completion.get("original_unqueued_superseded_count") != expected_old_unqueued):
        raise ValueError("fixed attack cell count mismatch")
    rows = [dict(row) for row in logical_rows]
    by_id = {row.get("unit_id"): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate logical unit")
    seen_cells = set()
    seen_aliases = set()
    attempts = []
    old_failed_count = 0
    old_unqueued_count = 0
    for cell in cells:
        logical_id = cell.get("scientific_cell_id")
        physical_id = cell.get("physical_unit_id")
        if (not _valid_id(logical_id) or not _valid_id(physical_id) or
                logical_id in seen_cells or logical_id not in by_id or
                by_id[logical_id].get("kind") not in ("reference_attack", "diagnostic_attack")):
            raise ValueError("fixed attack logical identity mismatch")
        seen_cells.add(logical_id)
        controls = cell.get("control_status")
        if controls != {"r1": "PASS", "r2": "PASS", "pair": "PASS"}:
            raise ValueError("fixed attack controls did not pass")
        q = queued.get(physical_id)
        if (not isinstance(q, dict) or q.get("unit_id") != physical_id or
                q.get("stage") != "a" or q.get("status") != "done" or
                q.get("receipt_sha256") != cell.get("queue_receipt_sha256") or
                not _valid_hash(q.get("receipt_sha256"))):
            raise ValueError("physical queue receipt mismatch")
        index_sha = cell.get("attack_index_sha256")
        if not _valid_hash(index_sha) or index_sha not in output_hashes.get(physical_id, set()):
            raise ValueError("physical attack index hash mismatch")
        original_status = cell.get("original_physical_status")
        row = by_id[logical_id]
        row["physical_unit_id"] = physical_id
        row["physical_queue_receipt_sha256"] = q["receipt_sha256"]
        row["attack_index_sha256"] = index_sha
        row["original_attempt_status"] = original_status
        row["scientific_status"] = "FIXED_ATTACK_COMPLETE_CONTROLS_PASS_REPLAY_PENDING"
        if physical_id == logical_id:
            if original_status != "SAME_PHYSICAL_ID" or row.get("status") != "done":
                raise ValueError("original completed attack mismatch")
            attempts.append(_attempt(logical_id, physical_id, "ORIGINAL_DONE", q,
                                     index_sha, "PASS_ALL_THREE_VIEWS"))
        else:
            match = mapping.get(physical_id)
            if (not isinstance(match, dict) or match.get("original_attack_unit_id") != logical_id or
                    match.get("original_attempt_status") != original_status):
                raise ValueError("attack alias mapping mismatch")
            seen_aliases.add(physical_id)
            old_q = queued.get(logical_id)
            if original_status == "FAILED_TECHNICAL_PRESERVED":
                if (not isinstance(old_q, dict) or old_q.get("status") != "failed" or
                        old_q.get("receipt_sha256") !=
                        correction.get("old_attempt_failed_queue_receipt_sha256", {}).get(logical_id) or
                        row.get("status") != "failed"):
                    raise ValueError("old failed attack receipt mismatch")
                row["original_queue_receipt_sha256"] = old_q["receipt_sha256"]
                attempts.append(_attempt(logical_id, logical_id, "ORIGINAL_FAILED", old_q))
                old_failed_count += 1
            elif original_status == "NOT_ENQUEUED_SOURCE_CORRECTION":
                if old_q is not None or row.get("status") != "PLANNED_NOT_ENQUEUED":
                    raise ValueError("old unqueued attack mismatch")
                old_unqueued_count += 1
            else:
                raise ValueError("invalid old attack status")
            row["status"] = "COMPLETE_VIA_R1_ALIAS"
            attempts.append(_attempt(logical_id, physical_id, "R1_DONE", q,
                                     index_sha, "PASS_ALL_THREE_VIEWS"))
    if (seen_aliases != set(mapping) or old_failed_count != expected_old_failed or
            old_unqueued_count != expected_old_unqueued):
        raise ValueError("attack alias coverage mismatch")
    attempts.sort(key=lambda row: (row["logical_unit_id"], row["physical_unit_id"]))
    return rows, attempts
