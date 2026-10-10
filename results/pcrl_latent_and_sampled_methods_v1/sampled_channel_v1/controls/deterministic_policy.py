"""Exact one-hot public release law for separately fitted deterministic gates.

Import with the shared sampled_channel_v1/impl directory on PYTHONPATH. This
subclass keeps the R12 cached wire adapter while changing the emitted law.
"""

import numpy as np

from channel import JointPolicy, _entropy_margin, expected_cost, bayes_recovery
from fit import TaskOnlyData, ProtectedData


class DeterministicJointPolicy(JointPolicy):
    law_kind = "deterministic_argmax_v1"

    def sparse_law(self, score1, score2, cell1: int, cell2: int):
        # Parent performs exact score, fine-cell, class and action validation.
        pairs, _ = self._pairs_and_probabilities(score1, score2, cell1, cell2)
        features = np.r_[score1, score2, _entropy_margin(score1),
                         _entropy_margin(score2)]
        slots = [4 * a + b for a in range(len(self.actions1[cell1]))
                 for b in range(len(self.actions2[cell2]))]
        winner = int(np.argmax(self.gate.logits(features)[slots]))
        i, j = pairs[winner]
        return ((int(i), int(j), 1.0),)


def replay_deployed_hard(data, policy: DeterministicJointPolicy):
    """Independent hard-law replay for checking R12's canonical exact_metrics.

    Both functions use the policy's deployed sparse_law; vectorized softmax
    replay_prepared remains specific to stochastic training diagnostics.
    """
    if not isinstance(policy, DeterministicJointPolicy):
        raise TypeError("hard replay requires the deployed hard policy")
    if isinstance(data, ProtectedData):
        task, sex = data.task, data.sex
    elif type(data) is TaskOnlyData:
        task, sex = data, None
    else:
        raise TypeError("hard replay requires admitted task or protected rows")
    laws = []
    losses = []
    for row in range(len(task.y1)):
        law = policy.sparse_law(task.score1[row], task.score2[row],
                                int(task.cell1[row]), int(task.cell2[row]))
        laws.append(law)
        cost = expected_cost(policy.bank1, policy.bank2, law,
                             y1=int(task.y1[row]), y2=int(task.y2[row]))
        losses.append((cost["r1"]["nll"], cost["r1"]["brier"],
                       cost["r2"]["nll"], cost["r2"]["brier"]))
    means = np.asarray(losses, dtype=np.float64).mean(axis=0)
    result = dict(zip(("r1_nll", "r1_brier", "r2_nll", "r2_brier"),
                      map(float, means)))
    _, counts = np.unique(np.asarray([law[0][:2] for law in laws], dtype=np.int64),
                          axis=0, return_counts=True)
    mass = counts / len(laws)
    result["task_rate"] = float(-(mass * np.log(mass)).sum())
    if sex is not None:
        recovery = bayes_recovery(laws, sex)
        result.update({"r1_bayes": recovery["r1"],
                       "r2_bayes": recovery["r2"],
                       "pair_bayes": recovery["pair"]})
        result["protected_objective"] = (recovery["pair"] +
                                         .5 * (recovery["r1"] + recovery["r2"]))
    result["max_law_residual"] = 0.0
    result["law_kind"] = policy.law_kind
    return result
