"""One registered margin vector for matched score or feature comparisons.

Pure in-memory arithmetic. A runner supplies only S4-admitted role arrays and
the actual complete-view fresh-reader predictions. This module neither picks
an arm/control nor opens a file; S1's lock supplies the ordered control IDs.
S4 recomputes every endpoint independently from private saved arrays.
"""

from __future__ import annotations

import re

import numpy as np

from attack_eval import oriented_auc
from common_eval import assert_same_people, task_metrics


SCHEMA = "pcrl-portfolio-common-margin-vector-v1"
VIEWS = ("r1", "r2", "pair")
TASKS = (("income", 1), ("occupation", 2))
REFERENCES = ("U0", "Ucal")


def margin_names(control_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not control_ids or len(set(control_ids)) != len(control_ids) or \
            any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", x) is None for x in control_ids):
        raise ValueError("predeclared matched-control IDs required")
    names = [f"{task}:{ref}:{metric}" for task, _ in TASKS for ref in REFERENCES
             for metric in ("ACC", "NLL", "BRIER")]
    names += [f"{control}:{view}:{metric}" for control in control_ids
              for view, metric in (("r1", "LOCAL"), ("r2", "LOCAL"), ("pair", "BENEFIT"))]
    return tuple(names)


def _one_release(record: dict, y: dict, positions: np.ndarray) -> dict:
    task = {name: task_metrics(record[f"prob_{r}"][positions], y[name][positions], task=name)
            for name, r in TASKS}
    sex = np.asarray(y["sex"], dtype=np.int64)[positions]
    auc = {}
    for view in VIEWS:
        readers = record["reader_predictions"].get(view)
        if not isinstance(readers, dict) or not readers:
            raise ValueError("required complete-view reader bank missing")
        auc[view] = max(oriented_auc(sex, np.asarray(pred)[positions])["fixed_alias_max_auc"]
                        for pred in readers.values())
    return {"task": task, "auc": auc}


def calculate_margins(seed_records: dict[int, dict], control_ids: tuple[str, ...],
                      positions) -> dict:
    """Seed-mean, paired-person margins; return registered order and values.

    `seed_records[k]` has `truth`, `candidate`, `controls`, and `references`
    with exact ordered row_id/unit. The same positions are used for all seeds.
    Repeated positions from a group bootstrap are allowed only after the full
    unsliced identity checks. U0/Ucal references are fixed per seed, not fitted
    or selected here. The caller includes all potentially eligible controls.
    """
    names = margin_names(control_ids)
    if not seed_records:
        raise ValueError("paired source seeds missing")
    pos = np.asarray(positions, dtype=np.int64)
    output = []
    first_truth = seed_records[min(seed_records)]["truth"]
    for seed in sorted(seed_records):
        item = seed_records[seed]
        truth = item["truth"]
        candidate = item["candidate"]
        controls = item["controls"]
        references = item["references"]
        if set(controls) != set(control_ids):
            raise ValueError("registered matched-control coverage missing")
        assert_same_people(first_truth, truth, candidate, references,
                           *[controls[c] for c in control_ids])
        n = len(truth["row_id"])
        if pos.ndim != 1 or not len(pos) or np.any((pos < 0) | (pos >= n)):
            raise ValueError("paired resample positions invalid")
        for name in ("income", "occupation", "sex"):
            if np.asarray(truth[name]).shape != (n,):
                raise ValueError("truth label shape mismatch")
        cand = _one_release(candidate, truth, pos)
        ref_metrics = {ref: {task: task_metrics(references[f"{ref}_{r}"][pos],
                                                 np.asarray(truth[task])[pos], task=task)
                             for task, r in TASKS} for ref in REFERENCES}
        ctl = {name: _one_release(controls[name], truth, pos) for name in control_ids}
        margin = []
        for task, _ in TASKS:
            for ref in REFERENCES:
                c, u = cand["task"][task], ref_metrics[ref][task]
                margin.extend((c["accuracy"] - u["accuracy"] + 0.010,
                               0.010 - (c["nll"] - u["nll"]),
                               0.005 - (c["brier"] - u["brier"])))
        for name in control_ids:
            for view in ("r1", "r2"):
                margin.append(0.010 - (cand["auc"][view] - ctl[name]["auc"][view]))
            margin.append(ctl[name]["auc"]["pair"] - cand["auc"]["pair"] - 0.020)
        output.append(np.asarray(margin, dtype=np.float64))
    values = np.mean(np.stack(output), axis=0)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise ValueError("common margin vector invalid")
    return {"schema": SCHEMA, "names": names, "values": values,
            "source_seeds": tuple(sorted(seed_records)), "resampled_rows": len(pos)}
