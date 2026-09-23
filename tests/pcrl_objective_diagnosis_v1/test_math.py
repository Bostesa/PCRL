import numpy as np

from analysis.pcrl_objective_diagnosis_v1.math import (
    affine_dual_certificate, cmi, cmi_gradient, information_radius_bound,
)


def test_cmi_gradient_on_simplex_direction():
    p = np.array([[[.1, .2], [.05, .15]], [[.2, .05], [.1, .15]]])
    q = np.array([[.3, .7], [.8, .2]])
    d = np.array([[1., -1.], [-1., 1.]])
    eps = 1e-6
    observed = (cmi(p, q+eps*d)-cmi(p, q-eps*d))/(2*eps)
    assert abs(observed - float(np.sum(cmi_gradient(p, q)*d))) < 1e-8


def test_dual_certificate_brackets_known_binary_solution():
    p = np.zeros((2, 1, 2))
    p[0, 0, 0] = p[1, 0, 1] = .5
    cost = np.array([[0., .5], [.5, 0.]])
    q = np.array([[.8, .2], [.2, .8]])
    budget = cmi(p, q)
    out = affine_dual_certificate(cost, {"s": p}, {"s": budget}, q)
    assert out["status"] == "dual_affine_bound"
    assert out["lower_bound"] <= out["primal_objective"] + 1e-9
    assert out["absolute_gap"] < 1e-5
    assert information_radius_bound(q, np.array([1., 1.])) >= budget-1e-12
