"""Prospective complete family, including every unrun comparison slot."""
from itertools import combinations

ARMS = ("P0", "O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21")
V_FAMILIES = ("V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21")
SEEDS = (0, 1)
TASKS = ("income", "occupation")
REFERENCES = ("U0", "Ucal")
METRICS = ("accuracy", "nll", "brier")
VIEWS = ("r1", "r2", "pair")
RECOVERY_SUMMARIES = ("s0_d0", "s0_d1", "s0_d2", "s1_d0", "s1_d1", "s1_d2", "s0_mean", "s1_mean", "mean")
K = 948


def ordered_clause_ids():
    utility = tuple(f"s{s}:utility:{a}:{t}:{r}:{m}"
                    for s in SEEDS for a in ARMS for t in TASKS
                    for r in REFERENCES for m in METRICS)
    recovery = tuple(f"{s}:recovery:{a}:{b}:{v}:auc"
                     for s in RECOVERY_SUMMARIES for a, b in combinations(ARMS, 2) for v in VIEWS)
    assert len(utility) == 192 and len(recovery) == 756
    assert len(set(utility + recovery)) == K
    return utility + recovery


def oriented_recovery_slot(candidate, control, view, summary="mean"):
    """Return slot/sign for benefit control minus candidate."""
    if candidate == control or candidate not in ARMS or control not in ARMS or view not in VIEWS or summary not in RECOVERY_SUMMARIES:
        raise ValueError("comparison is outside frozen family")
    first, second = sorted((candidate, control), key=ARMS.index)
    return f"{summary}:recovery:{first}:{second}:{view}:auc", (1 if control == first else -1)
