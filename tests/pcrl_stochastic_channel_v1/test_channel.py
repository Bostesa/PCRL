"""Finite-channel machinery: conditional-information algebra, the simultaneous-view nullspace
feasibility test, and the fixed-cost convex program.

Every guarantee established here is a statement about a **specified finite model**. None of it
transfers automatically to continuous `H` or to the ACS population; the coarse-conditioning fixture
below is the counterexample that makes that explicit. No data is read.
"""
import numpy as np
import pytest

from experiments.pcrl_stochastic_channel_v1 import channel as ch

LOG2 = float(np.log(2))


# ---------------------------------------------------------------- models used across the file
def review_toy():
    """The review's interior-probability toy: P(S=1|T) = (1/10, 2/5, 9/10), Y = 1[T == 1]."""
    p_t = np.full(3, 1 / 3)
    p_s1 = np.array([0.1, 0.4, 0.9])
    leak = np.zeros((2, 1, 3))                       # [S, C, T]; C is trivial
    leak[1, 0, :] = p_t * p_s1
    leak[0, 0, :] = p_t * (1 - p_s1)
    task = np.zeros((2, 1, 3))                       # [Y, C, T]
    task[1, 0, 1] = p_t[1]
    task[0, 0, 0] = p_t[0]
    task[0, 0, 2] = p_t[2]
    return leak, task


def xor_model():
    """S, C independent fair bits, T encodes (s, c) as 2s + c.

    Local role sees nothing, so I(S;Z) = 0 for Z = S xor C. Coalition role sees C, and then Z
    determines S exactly. Individually private, jointly disclosing.
    """
    local = np.zeros((2, 1, 4))
    coalition = np.zeros((2, 2, 4))
    for s in (0, 1):
        for c in (0, 1):
            local[s, 0, 2 * s + c] = 0.25
            coalition[s, c, 2 * s + c] = 0.25
    return local, coalition


def xor_channel():
    """Deterministic Z = S xor C, as a channel on T = 2s + c."""
    q = np.zeros((4, 2))
    for s in (0, 1):
        for c in (0, 1):
            q[2 * s + c, s ^ c] = 1.0
    return q


# ---------------------------------------------------------------- conditional mutual information
def test_cmi_is_zero_for_a_constant_channel():
    leak, _ = review_toy()
    q = np.zeros((3, 2))
    q[:, 0] = 1.0
    assert ch.cmi(leak, q) == pytest.approx(0.0, abs=1e-12)


def test_cmi_reproduces_the_reviewed_stochastic_release_exactly():
    """P(Z=1|T) = (1, 0, 11/13) gives exact independence and 0.4854855270534466 nats of task info."""
    leak, task = review_toy()
    q1 = np.array([1.0, 0.0, 11 / 13])
    q = np.column_stack((1 - q1, q1))
    assert ch.cmi(leak, q) == pytest.approx(0.0, abs=1e-12)
    assert ch.cmi(task, q) == pytest.approx(0.4854855270534466, abs=1e-12)


def test_cmi_reproduces_a_reviewed_deterministic_partition():
    """Partition [0,1,0] of the same toy: 0.004487966604102323 leakage, 0.6365141682948128 task."""
    leak, task = review_toy()
    q = np.zeros((3, 2))
    q[[0, 2], 0] = 1.0
    q[1, 1] = 1.0
    assert ch.cmi(leak, q) == pytest.approx(0.004487966604102323, abs=1e-12)
    assert ch.cmi(task, q) == pytest.approx(0.6365141682948128, abs=1e-12)


def test_cmi_conditions_correctly_on_a_nontrivial_view():
    local, coalition = xor_model()
    q = xor_channel()
    assert ch.cmi(local, q) == pytest.approx(0.0, abs=1e-12)       # individually private
    assert ch.cmi(coalition, q) == pytest.approx(LOG2, abs=1e-12)  # jointly disclosing


def test_the_reverse_counterexample_also_holds():
    """B = S and Z = S: the coalition increment is zero while the local one is log 2.

    Both directions occur, so neither role constraint implies the other and both are necessary.
    """
    local = np.zeros((2, 1, 2))
    coalition = np.zeros((2, 2, 2))
    for s in (0, 1):
        local[s, 0, s] = 0.5
        coalition[s, s, s] = 0.5                     # the coalition already sees B = S
    q = np.eye(2)
    assert ch.cmi(local, q) == pytest.approx(LOG2, abs=1e-12)
    assert ch.cmi(coalition, q) == pytest.approx(0.0, abs=1e-12)


def test_coarse_conditioning_hides_the_leak_entirely():
    """`I(S;Z|bin(C)) = 0` does not imply `I(S;Z|C) = 0`. This is why fitted-model constraints are
    never advertised as population conditional-information certificates."""
    _, coalition = xor_model()
    coarse = coalition.sum(axis=1, keepdims=True)    # put both C values in one bin
    q = xor_channel()
    assert ch.cmi(coarse, q) == pytest.approx(0.0, abs=1e-12)
    assert ch.cmi(coalition, q) == pytest.approx(LOG2, abs=1e-12)


def test_cmi_is_convex_in_the_channel():
    """Spot-check of the property the whole program rests on, on a random chord."""
    leak, _ = review_toy()
    rng = np.random.default_rng(0)
    for _ in range(25):
        a, b = (ch.random_channel(3, 2, rng) for _ in range(2))
        lam = rng.uniform(0.05, 0.95)
        mixed = lam * a + (1 - lam) * b
        assert ch.cmi(leak, mixed) <= lam * ch.cmi(leak, a) + (1 - lam) * ch.cmi(leak, b) + 1e-12


# ---------------------------------------------------------------- nullspace feasibility
def test_feasibility_matches_the_reviewed_toy():
    """A useful perfectly private binary release exists: ker(A) is not contained in ker(B)."""
    leak, task = review_toy()
    out = ch.feasible_direction([leak], task)
    assert out['feasible'] is True
    q1 = 0.5 + out['scale'] * out['direction']
    q = np.column_stack((1 - q1, q1))
    assert ch.cmi(leak, q) == pytest.approx(0.0, abs=1e-10)
    assert ch.cmi(task, q) > 1e-6


def test_infeasible_when_the_code_determines_the_protected_attribute():
    """If T determines S, ker(A) is exactly the constants and every constant map is useless."""
    leak = np.zeros((3, 1, 3))
    task = np.zeros((2, 1, 3))
    for t in range(3):
        leak[t, 0, t] = 1 / 3
        task[t % 2, 0, t] = 1 / 3
    out = ch.feasible_direction([leak], task)
    assert out['feasible'] is False
    assert out['null_dim'] == 1                      # the constants only


def test_simultaneous_roles_are_stricter_than_either_alone():
    """The XOR channel is feasible for the local role alone and infeasible once the coalition
    role is added -- the intersection over roles, not a single nullspace."""
    local, coalition = xor_model()
    task = np.zeros((2, 1, 4))
    for s in (0, 1):
        for c in (0, 1):
            task[s ^ c, 0, 2 * s + c] = 0.25         # the task IS S xor C
    alone = ch.feasible_direction([local], task)
    both = ch.feasible_direction([local, coalition], task)
    assert alone['feasible'] is True
    assert both['feasible'] is False
    assert both['null_dim'] < alone['null_dim']


def test_direction_stays_in_the_probability_box():
    leak, task = review_toy()
    out = ch.feasible_direction([leak], task)
    q1 = 0.5 + out['scale'] * out['direction']
    assert q1.min() >= -1e-12 and q1.max() <= 1 + 1e-12


def test_feasibility_reports_sampling_sensitivity_of_near_null_directions():
    """Estimated models have no exact nullspace; the tolerance and the spectrum must be reported."""
    leak, task = review_toy()
    out = ch.feasible_direction([leak], task)
    assert 'singular_values' in out and len(out['singular_values']) == 3
    assert out['tolerance'] > 0
    assert 'null_dim_by_tolerance' in out           # how the answer moves with the threshold


# ---------------------------------------------------------------- convex program
def test_solver_beats_every_deterministic_private_map_at_delta_zero():
    """The randomization payoff, through the solver.

    Cost is 0 when Z matches Y = 1[T == 1] and 1 otherwise. The only perfectly private
    deterministic map is constant, whose cost is P(Y = 1) = 1/3. A randomized channel must do
    strictly better while holding leakage at zero.
    """
    leak, task = review_toy()
    y = np.array([0, 1, 0])
    cost = np.array([[0.0 if z == y[t] else 1.0 for z in (0, 1)] for t in range(3)])
    out = ch.solve(roles={'local': leak}, cost=cost, deltas={'local': 0.0}, n_z=2)
    assert out['status'] in ('optimal', 'optimal_inaccurate')
    assert out['objective'] < 1 / 3 - 1e-3
    assert ch.cmi(leak, out['Q']) < 1e-7
    assert ch.cmi(task, out['Q']) > 1e-3


def test_solver_honours_both_roles_and_refuses_the_xor_channel():
    local, coalition = xor_model()
    y = np.array([0, 1, 1, 0])                       # the task is exactly S xor C
    cost = np.array([[0.0 if z == y[t] else 1.0 for z in (0, 1)] for t in range(4)])
    local_only = ch.solve(roles={'local': local}, cost=cost, deltas={'local': 0.0}, n_z=2)
    both = ch.solve(roles={'local': local, 'coalition': coalition},
                    cost=cost, deltas={'local': 0.0, 'coalition': 0.0}, n_z=2)
    assert local_only['objective'] < 1e-4            # XOR is free for the local role alone
    assert both['objective'] > 0.4                   # and unavailable once the coalition is bound
    assert ch.cmi(coalition, both['Q']) < 1e-6


def test_solver_reports_feasibility_residuals_and_an_optimality_gap():
    """A solver success string alone is insufficient (REGISTRATION G4)."""
    leak, task = review_toy()
    cost = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    out = ch.solve(roles={'local': leak}, cost=cost, deltas={'local': 0.01}, n_z=2)
    assert out['row_sum_residual'] < 1e-8
    assert out['negativity_residual'] < 1e-8
    assert out['constraint_violation']['local'] < 1e-7
    assert out['optimality_gap'] is None or out['optimality_gap'] < 1e-6
    assert out['solver'] and out['verified_cmi']['local'] <= 0.01 + 1e-7


def test_a_looser_budget_never_costs_more():
    """Monotonicity of the value function: the feasible set grows with delta."""
    leak, _ = review_toy()
    cost = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    vals = [ch.solve(roles={'local': leak}, cost=cost, deltas={'local': d}, n_z=2)['objective']
            for d in (0.0, 0.01, 0.05)]
    assert vals[0] >= vals[1] - 1e-7 >= vals[2] - 2e-7


def test_lp_variant_is_at_least_as_restrictive_as_the_information_budget():
    """Pointwise likelihood-ratio constraints control the whole output law, not an average."""
    leak, _ = review_toy()
    cost = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    eps = 0.05
    lp = ch.solve_lp(roles={'local': leak}, cost=cost, epsilons={'local': eps}, n_z=2)
    assert lp['status'] in ('optimal', 'optimal_inaccurate')
    achieved = ch.cmi(leak, lp['Q'])
    mi = ch.solve(roles={'local': leak}, cost=cost, deltas={'local': achieved}, n_z=2)
    assert mi['objective'] <= lp['objective'] + 1e-6      # the MI budget is the weaker constraint
    assert lp['max_log_ratio'] <= eps + 1e-6


# ---------------------------------------------------------------- the released object
def test_release_is_a_sampled_one_hot_token_not_the_probability_vector():
    """REGISTRATION G3. Releasing Q[t,:] is a different mechanism and may disclose T."""
    rng = np.random.default_rng(0)
    q = np.array([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]])
    t = np.array([0, 1, 2, 1, 0])
    out = ch.sample_release(q, t, rng)
    assert out.shape == (5, 2)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert np.all(out.sum(1) == 1.0)
    assert np.all(out[2] == [1.0, 0.0])              # a degenerate row is deterministic
    assert not np.allclose(out[:2], q[t[:2]])        # emphatically not the probability vector


def test_release_check_rejects_a_probability_vector():
    q = np.array([[0.2, 0.8], [0.5, 0.5]])
    t = np.array([0, 1])
    assert ch.is_sampled_token(ch.sample_release(q, t, np.random.default_rng(1))) is True
    assert ch.is_sampled_token(q[t]) is False
    assert ch.is_sampled_token(np.array([[0.5, 0.5], [1.0, 0.0]])) is False


def test_sampling_is_unbiased_for_the_channel():
    rng = np.random.default_rng(7)
    q = np.array([[0.25, 0.75], [0.9, 0.1]])
    t = np.repeat([0, 1], 20000)
    out = ch.sample_release(q, t, rng)
    assert out[:20000].mean(0) == pytest.approx(q[0], abs=0.01)
    assert out[20000:].mean(0) == pytest.approx(q[1], abs=0.01)
