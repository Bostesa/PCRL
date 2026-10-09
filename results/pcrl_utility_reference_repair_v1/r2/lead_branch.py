"""Pure prospective two-lead qualification on already aligned utility rows."""

from __future__ import annotations

LEADS = ("M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS")
TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")


def qualify_historical_leads(rows: dict) -> tuple[str, ...]:
    """Return exactly the historical masks passing all point clauses."""
    if not isinstance(rows, dict) or set(rows) != {"0", "1"}:
        raise ValueError("lead rows need both paired seeds")
    for seed in ("0", "1"):
        masks = rows[seed]
        if not isinstance(masks, dict) or set(masks) != set(LEADS):
            raise ValueError("lead rows need both frozen mask identities")
        for mask in LEADS:
            tasks = masks[mask]
            if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
                raise ValueError("lead rows need both task identities")
            for task in TASKS:
                item = tasks[task]
                refs = item.get("references") if isinstance(item, dict) else None
                if not isinstance(refs, dict) or set(refs) != set(REFERENCES):
                    raise ValueError("lead reference coverage differs")
                for reference in REFERENCES:
                    status = refs[reference]
                    if (not isinstance(status, dict)
                            or type(status.get("point_eligible")) is not bool):
                        raise ValueError("lead point eligibility must be boolean")
    return tuple(mask for mask in LEADS if all(
        rows[seed][mask][task]["references"][reference]["point_eligible"]
        for seed in ("0", "1")
        for task in TASKS
        for reference in REFERENCES
    ))

