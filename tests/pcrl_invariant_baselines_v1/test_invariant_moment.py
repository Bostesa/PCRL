"""Validation fixtures for the rotation-invariant moment refinement.

These establish that the code computes the stated finite objective and that the
objective has the invariance the method claims. They do **not** certify privacy.

No fixture parameter is tuned to make a measure look successful. Two fixtures exist
precisely to show the penalty can be **wrong** (conditional-null concentration and
nuisance misspecification); their expected behaviour is declared in the fixture body,
not inferred after the fact.

Tolerances are the ones declared in ``results/pcrl_invariant_baselines_v1/PROTOCOL.md``
section 4.1 and hashed into ``PROTOCOL_FREEZE.json`` before any fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.pcrl_nonlinear_rank_v1.nonlinear_moment import ProtectedRole
from experiments.pcrl_invariant_baselines_v1.invariant_moment import (
    BANDS, KernelBlock, QuadraticFeatures, build_features, packed_weights, quadratic_pairs,
    quadratic_value, quadratic_value_and_dz, quadratic_value_reference, reference_bandwidth,
    subset_class_support)
from experiments.pcrl_invariant_baselines_v1.objective import (FAMILY_INVARIANT, FAMILY_ORIGINAL,
                                                               InvariantRolePenalty,
                                                               PolicyObjective, canonicalize,
                                                               qr_retract, spectral_solution,
                                                               tangent_project)

RNG = np.random.default_rng(20260918)

# ---- declared tolerances (PROTOCOL.md section 4.1) --------------------------------
TOL_PACKED_REL = 1e-12
TOL_KERNEL_REL = 1e-12
TOL_GRAD_REL = 1e-6
FD_STEP = 1e-6
TOL_ROTATION_DERIVATIVE_REL = 1e-9
TOL_CHUNK_REL = 1e-12
TOL_FEASIBILITY = 1e-13


def invariance_ok(before: float, after: float) -> bool:
    """``abs(delta) <= 1e-10 * max(1, abs(value))`` -- the declared invariance tolerance."""
    return abs(after - before) <= 1e-10 * max(1.0, abs(before))


def whiten(x):
    x = x - x.mean(0)
    cov = x.T @ x / len(x)
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    keep = values > max(1e-12, 1e-10 * values[-1])
    return x @ (vectors[:, keep] / np.sqrt(values[keep]))


def make_role(name, basis, residual, gram_dim, classes=None, gram=None):
    classes = classes if classes is not None else residual.shape[1]
    g = np.zeros((gram_dim, gram_dim)) if gram is None else gram
    return ProtectedRole(name=name, view=name.split('/')[0], basis=basis, residual=residual,
                         linear_gram=g, n_valid=len(basis), classes=classes,
                         support=np.full(classes, len(basis) // classes), unsupported=[],
                         raw_trace=1.0, zero_trace=False)


def make_kernel(name, role, n_rows, sigma, seed=0, r=4):
    # Synthetic fixtures use their own role names, so the role index is passed
    # explicitly; every production fit derives it from the frozen ROLE_ORDER.
    from experiments.pcrl_invariant_baselines_v1.invariant_moment import ROLE_ORDER
    index = ROLE_ORDER.index(name) if name in ROLE_ORDER else 0
    return KernelBlock.build(name, role, np.arange(n_rows), sigma, seed=seed, r=r,
                             role_index=index)


def orthogonal(r, rng):
    return qr_retract(rng.normal(size=(r, r)))


def sign_flip(r, rng):
    return np.diag(rng.choice([-1.0, 1.0], size=r))


def permutation(r, rng):
    return np.eye(r)[rng.permutation(r)]


# =================================================================== 1. packed identity
def test_packed_sqrt2_weights_equal_the_frobenius_norm():
    """Check 1: the packed sqrt(2) block IS sum_kc ||M_jkc||_F^2."""
    for r in (3, 5, 8):
        v = whiten(RNG.normal(size=(600, r + 4)))
        w = qr_retract(RNG.normal(size=(v.shape[1], r)))
        quadratic = QuadraticFeatures.build(r)
        basis = np.column_stack((np.ones(len(v)), RNG.uniform(size=(len(v), 2))))
        residual = RNG.normal(size=(len(v), 3)) * 0.4
        role = make_role('A/x', basis, residual, v.shape[1])
        bundle = build_features(quadratic, v, w)
        packed = quadratic_value(bundle, role.weights(), None, role.n_valid)
        reference = quadratic_value_reference(quadratic, bundle, role.weights(), None, role.n_valid)
        assert abs(packed - reference) <= TOL_PACKED_REL * max(1.0, abs(reference))

        # ...and a fully independent path: build M by explicit accumulation.
        z = bundle.z
        total = 0.0
        weights = role.weights()
        for column in range(weights.shape[1]):
            m = (z * weights[:, [column]]).T @ z / role.n_valid
            total += float(np.sum(m * m))
        assert abs(packed - total) <= TOL_PACKED_REL * max(1.0, abs(total))


def test_equal_weight_monomials_are_NOT_the_frobenius_norm():
    """The predecessor's defect, reproduced so the repair is attributable."""
    r = 6
    pairs = quadratic_pairs(r)
    w = packed_weights(pairs)
    assert w[[a == b for a, b in pairs]].tolist() == [1.0] * r
    off = w[[a != b for a, b in pairs]]
    assert np.allclose(off, np.sqrt(2.0))
    # Equal weights would compute sum_a M_aa^2 + sum_{a<b} M_ab^2, which is strictly
    # smaller than ||M||_F^2 whenever any off-diagonal moment is nonzero.
    assert not np.allclose(w, 1.0)


# =================================================================== 2. kernel reference
def test_kernel_block_matches_an_independent_dense_gram_formula():
    """Check 2: the precomputed L path equals the explicit double sum."""
    n, r = 120, 4
    v = whiten(RNG.normal(size=(n, r + 3)))
    w = qr_retract(RNG.normal(size=(v.shape[1], r)))
    basis = np.column_stack((np.ones(n), RNG.uniform(size=(n, 2))))
    residual = RNG.normal(size=(n, 3)) * 0.5
    role = make_role('A/SEX', basis, residual, v.shape[1], classes=3)
    z = v @ w
    sigma = float(np.median([np.linalg.norm(z[i] - z[j]) for i in range(40) for j in range(i + 1, 40)]))
    block = make_kernel('A/SEX', role, n, sigma, r=r)
    fast = block.value(z)
    slow = block.value_dense_reference(z, basis, residual)
    assert abs(fast - slow) <= TOL_KERNEL_REL * max(1.0, abs(slow))


def test_kernel_block_is_a_squared_kernel_feature_moment_via_explicit_feature_map():
    """The block equals sum_kc ||(1/m) sum_i phi(z_i) b_k(i) e_c(i)||^2.

    Verified with an EXPLICIT finite feature map: use a single linear kernel
    (K = <z_i, z_l>), for which phi is the identity, so the moment is computable
    directly. This checks the algebra of L = (BB') * (EE'), not the Gaussian.
    """
    n, r = 80, 3
    z = RNG.normal(size=(n, r))
    basis = np.column_stack((np.ones(n), RNG.uniform(size=(n, 2))))
    residual = RNG.normal(size=(n, 2))
    l_matrix = (basis @ basis.T) * (residual @ residual.T)
    linear_form = float(np.sum((z @ z.T) * l_matrix)) / n ** 2
    moment_form = 0.0
    for k in range(basis.shape[1]):
        for c in range(residual.shape[1]):
            moment = (z * (basis[:, [k]] * residual[:, [c]])).sum(0) / n
            moment_form += float(moment @ moment)
    assert abs(linear_form - moment_form) <= TOL_KERNEL_REL * max(1.0, abs(moment_form))


def test_kernel_keeps_the_declared_v_statistic_diagonal():
    """The diagonal i == l is retained. Declared convention, never swapped on outcome."""
    n, r = 60, 3
    v = whiten(RNG.normal(size=(n, r + 2)))
    w = qr_retract(RNG.normal(size=(v.shape[1], r)))
    basis = np.ones((n, 1))
    residual = RNG.normal(size=(n, 2))
    role = make_role('A/SEX', basis, residual, v.shape[1], classes=2)
    z = v @ w
    block = make_kernel('A/SEX', role, n, 1.0, r=r)
    kernel = block.kernel(z[block.row_index])
    assert np.allclose(np.diag(kernel), 1.0)          # exp(0) averaged over three bands
    with_diagonal = float(np.sum(kernel * block.l_matrix))
    without = float(np.sum((kernel - np.diag(np.diag(kernel))) * block.l_matrix))
    assert with_diagonal != without
    assert block.value(z) == pytest.approx(with_diagonal / block.m ** 2, rel=1e-15)
    assert block.diagnostics['estimator'] == 'V-statistic (diagonal retained)'


def test_three_frozen_bandwidths_are_the_declared_multiples():
    block = make_kernel('A/SEX', make_role('A/SEX', np.ones((50, 1)), RNG.normal(size=(50, 2)), 5),
                        50, 2.5, r=3)
    assert np.allclose(block.diagnostics['bandwidths'], np.array(BANDS) * 2.5)


def test_bandwidth_failure_is_explicit_not_silent():
    with pytest.raises(ValueError, match='bandwidth diagnostic failure'):
        reference_bandwidth(np.ones((20, 3)), seed=0, r=3)


# =================================================================== 3. invariance
def build_objective(r=4, q=9, n=500, policy='C1', seed=0, family=FAMILY_INVARIANT):
    v = whiten(RNG.normal(size=(n, q)))
    q = v.shape[1]
    quadratic = QuadraticFeatures.build(r)
    w_ref = qr_retract(RNG.normal(size=(q, r)))
    sigma = reference_bandwidth(v @ w_ref, seed, r)['sigma_reference']
    local, coalition = {}, {}
    for index, name in enumerate(('A/SEX', 'A/RAC1P', 'A/public_coverage', 'AB/SEX', 'AB/RAC1P')):
        classes = 9 if 'RAC1P' in name else 2
        width = 6 if name.startswith('A/') else 10
        basis = np.column_stack((np.ones(n), RNG.uniform(size=(n, width - 1))))
        residual = RNG.normal(size=(n, classes)) * 0.3
        gram = RNG.normal(size=(q, q))
        gram = (gram @ gram.T) / (q * n)
        role = make_role(name, basis, residual, q, classes=classes, gram=gram)
        block = make_kernel(name, role, n, sigma, seed=seed, r=r)
        penalty = InvariantRolePenalty.build(role, quadratic, block, v, None, w_ref)
        (local if name.startswith('A/') else coalition)[name] = penalty
    objective = PolicyObjective(utility=np.eye(q) / q, local=local, coalition=coalition,
                                policy=policy, family=family, features=v, quadratic=quadratic)
    return objective, v, w_ref


@pytest.mark.parametrize('r', (3, 5, 8))
@pytest.mark.parametrize('seed', (0, 1))
def test_every_component_and_the_objective_are_invariant_under_many_transforms(r, seed):
    """Check 3: MANY rotations, sign flips and permutations -- not one chosen Q."""
    objective, v, w_ref = build_objective(r=r, seed=seed)
    w = qr_retract(RNG.normal(size=(v.shape[1], r)))
    rng = np.random.default_rng(1000 + 10 * seed + r)
    transforms = ([('rotation', orthogonal(r, rng)) for _ in range(8)]
                  + [('sign', sign_flip(r, rng)) for _ in range(4)]
                  + [('permutation', permutation(r, rng)) for _ in range(4)])

    base_bundle = objective.bundle_for(w)
    base_loss = objective.loss(w)
    base_utility = float(np.trace(w.T @ objective.utility @ w))
    base_parts = {name: penalty.nonlinear_value(base_bundle)[1]
                  for name, penalty in {**objective.local, **objective.coalition}.items()}

    for kind, transform in transforms:
        rotated = w @ transform
        assert np.max(abs(rotated.T @ rotated - np.eye(r))) < 1e-13, kind
        bundle = objective.bundle_for(rotated)
        assert invariance_ok(base_loss, objective.loss(rotated)), f'{kind}: objective moved'
        assert invariance_ok(base_utility,
                             float(np.trace(rotated.T @ objective.utility @ rotated))), kind
        for name, penalty in {**objective.local, **objective.coalition}.items():
            detail = penalty.nonlinear_value(bundle)[1]
            for key in ('quadratic_block_raw', 'kernel_block_raw'):
                assert invariance_ok(base_parts[name][key], detail[key]), f'{kind} {name} {key}'
            assert invariance_ok(penalty.linear_value(w), penalty.linear_value(rotated)), kind


def test_reconstruction_is_invariant_too():
    """Reconstruction of the teacher residual depends only on the subspace."""
    objective, v, _ = build_objective(r=5)
    r = 5
    w = qr_retract(RNG.normal(size=(v.shape[1], r)))
    target = RNG.normal(size=(len(v), 6))
    rng = np.random.default_rng(77)
    def mse(mapping):
        z = v @ mapping
        coefficient = np.linalg.lstsq(z, target, rcond=1e-10)[0]
        return float(np.sum((target - z @ coefficient) ** 2) / len(z))
    base = mse(w)
    for _ in range(6):
        assert invariance_ok(base, mse(w @ orthogonal(r, rng)))


def test_the_predecessor_objective_is_NOT_invariant():
    """Control: the defective family must still move, or the repair proves nothing."""
    from experiments.pcrl_nonlinear_rank_v1.nonlinear_moment import ChiFeatures
    n, r, q = 500, 5, 9
    v = whiten(RNG.normal(size=(n, q)))
    q = v.shape[1]
    w = qr_retract(RNG.normal(size=(q, r)))
    chi = ChiFeatures.fit(v @ w, seed=0)
    basis = np.column_stack((np.ones(n), RNG.uniform(size=(n, 5))))
    residual = RNG.normal(size=(n, 2)) * 0.3
    role = make_role('A/SEX', basis, residual, q, classes=2)
    from experiments.pcrl_nonlinear_rank_v1.nonlinear_moment import (build_features as old_build,
                                                                     role_moments as old_moments,
                                                                     block_squared_norms)
    def old_quadratic(mapping):
        bundle = old_build(chi, v, mapping)
        m = old_moments(role, bundle, None)
        return block_squared_norms(m, chi)[0]
    base = old_quadratic(w)
    moved = max(abs(old_quadratic(w @ orthogonal(r, np.random.default_rng(s))) - base)
                for s in range(5))
    assert moved > 1e-6 * max(1.0, abs(base)), (
        'the predecessor quadratic block should be rotation SENSITIVE; if it is not, '
        'this fixture is not exercising the defect it is meant to reproduce')


# =================================================================== 4/5. gradients
def test_analytic_gradient_matches_central_finite_differences():
    """Check 4: analytic dL/dW against central differences at the declared tolerance."""
    objective, v, _ = build_objective(r=4, n=300)
    q, r = v.shape[1], 4
    w = qr_retract(RNG.normal(size=(q, r)))
    _, grad, _ = objective.loss_and_grad(w)
    rng = np.random.default_rng(4242)
    for _ in range(12):
        direction = rng.normal(size=(q, r))
        direction /= np.linalg.norm(direction)
        plus = objective.loss(w + FD_STEP * direction)
        minus = objective.loss(w - FD_STEP * direction)
        numeric = (plus - minus) / (2 * FD_STEP)
        analytic = float(np.sum(grad * direction))
        assert abs(numeric - analytic) <= TOL_GRAD_REL * max(1.0, abs(numeric))


def test_gradient_is_near_zero_along_pure_rotation_directions():
    """Check 5: with L a function on the Grassmannian, <G, W A> = 0 for skew A."""
    for r in (3, 5):
        objective, v, _ = build_objective(r=r, n=300)
        w = qr_retract(RNG.normal(size=(v.shape[1], r)))
        _, grad, _ = objective.loss_and_grad(w)
        rng = np.random.default_rng(900 + r)
        for _ in range(8):
            a = rng.normal(size=(r, r))
            a = a - a.T                                   # skew-symmetric
            direction = w @ a
            inner = abs(float(np.sum(grad * direction)))
            bound = (TOL_ROTATION_DERIVATIVE_REL * np.linalg.norm(grad)
                     * np.linalg.norm(direction))
            assert inner <= bound, f'r={r}: rotation derivative {inner:.3e} > {bound:.3e}'


def test_quadratic_dz_matches_the_closed_form_of_method_equation_14():
    """dD_quad/dz_i = (4/n) sum_kc w_ikc M_jkc z_i -- an independent derivation."""
    n, r = 200, 4
    v = whiten(RNG.normal(size=(n, r + 3)))
    w = qr_retract(RNG.normal(size=(v.shape[1], r)))
    quadratic = QuadraticFeatures.build(r)
    basis = np.column_stack((np.ones(n), RNG.uniform(size=(n, 2))))
    residual = RNG.normal(size=(n, 2)) * 0.5
    role = make_role('A/SEX', basis, residual, v.shape[1], classes=2)
    bundle = build_features(quadratic, v, w)
    _, dz = quadratic_value_and_dz(quadratic, bundle, role.weights(), None, role.n_valid, 1.0)

    weights = role.weights()
    z = bundle.z
    moments = [(z * weights[:, [c]]).T @ z / role.n_valid for c in range(weights.shape[1])]
    expected = np.zeros_like(z)
    for c, m in enumerate(moments):
        expected += (4.0 / role.n_valid) * weights[:, [c]] * (z @ m)
    assert np.max(abs(dz - expected)) <= 1e-10 * max(1.0, np.max(abs(expected)))


# =================================================================== 8. solver replay
def test_original_family_is_still_the_closed_form_global_optimum():
    """Check 8: the original-moment family remains an exact fixed-matrix eigenproblem."""
    objective, v, _ = build_objective(r=5, family=FAMILY_ORIGINAL)
    r = 5
    a = objective.spectral_matrix()
    w, values = spectral_solution(a, r)
    assert np.max(abs(w.T @ w - np.eye(r))) <= TOL_FEASIBILITY
    assert objective.loss(w) == pytest.approx(-float(values[:r].sum()), rel=1e-12, abs=1e-14)
    rng = np.random.default_rng(31337)
    for _ in range(20):
        other = qr_retract(rng.normal(size=(v.shape[1], r)))
        assert objective.loss(other) >= objective.loss(w) - 1e-12


def test_canonicalisation_and_retraction_are_the_predecessor_functions():
    """The solver pieces are imported, not reimplemented: identity is asserted."""
    from experiments.pcrl_nonlinear_rank_v1 import objective as old
    from experiments.pcrl_invariant_baselines_v1 import objective as new
    assert new.canonicalize is old.canonicalize
    assert new.qr_retract is old.qr_retract
    assert new.tangent_project is old.tangent_project
    assert new.stiefel_descent is old.stiefel_descent
    assert new.spectral_solution is old.spectral_solution
    assert new.perturbed_start is old.perturbed_start
    assert new.POLICIES == {'L1': (1.0, 0.0), 'L2': (2.0, 0.0), 'C1': (1.0, 1.0)}


# =================================================================== 9. numerical stability
def test_chunk_size_invariance_and_repeated_serial_evaluation():
    """Check 9: alternate chunk sizes and repeated serial evaluations on fixed objects."""
    objective, v, _ = build_objective(r=4, n=700)
    w = qr_retract(RNG.normal(size=(v.shape[1], 4)))
    first = objective.loss(w)
    for _ in range(5):
        assert objective.loss(w) == first          # bitwise, same process, fixed objects
    for chunk in (64, 512, 4096, 100000):
        objective.chunk = chunk
        value = objective.loss(w)
        assert abs(value - first) <= TOL_CHUNK_REL * max(1.0, abs(first)), chunk


def test_kernel_subset_indices_are_frozen_and_reproducible():
    role = make_role('A/RAC1P', np.ones((3000, 1)), RNG.normal(size=(3000, 9)), 5, classes=9)
    a = make_kernel('A/RAC1P', role, 3000, 1.0, seed=1, r=16)
    b = make_kernel('A/RAC1P', role, 3000, 1.0, seed=1, r=16)
    c = make_kernel('A/RAC1P', role, 3000, 1.0, seed=2, r=16)
    assert np.array_equal(a.subset_index, b.subset_index)
    assert not np.array_equal(a.subset_index, c.subset_index)
    assert a.m == 512 and len(np.unique(a.subset_index)) == 512
    assert a.diagnostics['subset_seed'] == 20260930 + 100 * 1 + 10 * 16 + 1


def test_subset_support_is_recorded_not_manufactured():
    """Rare classes may have zero subset support. That is reported, never padded."""
    n = 3000
    labels = np.zeros(n, dtype=int)
    labels[:5] = 3                                     # a class with 5 rows in 3000
    residual = np.zeros((n, 9))
    residual[np.arange(n), labels] = 1.0
    residual -= residual.mean(0)
    role = make_role('A/RAC1P', np.ones((n, 1)), residual, 5, classes=9)
    role.support = np.bincount(labels, minlength=9)
    block = make_kernel('A/RAC1P', role, n, 1.0, seed=0, r=16)
    record = subset_class_support(role, block, labels)
    assert sum(record['subset_support']) == block.m
    assert record['population_support'][3] == 5
    # whatever the realised support is, it is reported and not repaired
    assert set(record['subset_zero_support_classes']) == {
        c for c in range(9) if record['subset_support'][c] == 0}


# =================================================================== 7. falsification fixtures
# Carried over from the predecessor and re-pointed at the repaired objective. Two of
# these exist to show the penalty can be WRONG; their expected behaviour is declared
# here, in advance, not inferred from what the numbers turn out to be.

def _blocks(role, v, w, r, seed=3, sigma=None):
    quadratic = QuadraticFeatures.build(r)
    bundle = build_features(quadratic, v, w)
    if sigma is None:
        sigma = reference_bandwidth(bundle.z, seed, r)['sigma_reference']
    block = make_kernel(role.name, role, len(v), sigma, seed=seed, r=r)
    quad = quadratic_value(bundle, role.weights(), None, role.n_valid)
    return quad, block.value(bundle.z), bundle


def test_magnitude_leakage_is_seen_by_the_repaired_blocks_but_not_by_first_moments():
    """S depends on |z|, so every FIRST moment of z vanishes while the blocks fire."""
    n = 8000
    z = RNG.normal(size=n)
    p = 1 / (1 + np.exp(-(z ** 2 - 1.0)))          # depends on magnitude only
    s = (RNG.uniform(size=n) < p).astype(int)
    v = whiten(np.column_stack((z, RNG.normal(size=(n, 2)))))
    w = np.linalg.lstsq(v, z, rcond=None)[0].reshape(-1, 1)
    w /= np.linalg.norm(w)
    onehot = np.eye(2)[s]
    residual = onehot - onehot.mean(0)
    basis = np.ones((n, 1))
    role = make_role('A/S', basis, residual, v.shape[1], classes=2)
    zz = (v @ w).ravel()
    first_moment = abs(float(np.mean(zz[:, None] * residual)))
    quad, kernel, _ = _blocks(role, v, w, 1)
    assert first_moment < 0.02, 'the first moment should be near zero by construction'
    assert quad > 1e-4, 'the quadratic block must see magnitude dependence'
    assert kernel > 1e-6, 'the kernel block must see magnitude dependence'


def test_xor_with_side_information_needs_the_interacted_basis():
    """Z alone says nothing about S; Z interacted with H does. The intercept is retained."""
    n = 8000
    h = RNG.integers(0, 2, n).astype(float)
    z = RNG.normal(size=n)
    s = ((z > 0).astype(int) ^ h.astype(int))
    v = whiten(np.column_stack((z, RNG.normal(size=(n, 2)))))
    w = np.linalg.lstsq(v, z, rcond=None)[0].reshape(-1, 1)
    w /= np.linalg.norm(w)
    onehot = np.eye(2)[s]
    residual = onehot - onehot.mean(0)
    constant = make_role('A/S', np.ones((n, 1)), residual, v.shape[1], classes=2)
    interacted = make_role('A/S', np.column_stack((np.ones(n), h)), residual, v.shape[1], classes=2)
    quadratic = QuadraticFeatures.build(1)
    bundle = build_features(quadratic, v, w)
    sigma = reference_bandwidth(bundle.z, 3, 1)['sigma_reference']
    kc = make_kernel('A/S', constant, n, sigma, seed=3, r=1).value(bundle.z)
    ki = make_kernel('A/S', interacted, n, sigma, seed=3, r=1).value(bundle.z)
    assert ki > 3 * kc, ('the constant basis should barely see XOR disclosure while the '
                         f'H-interacted basis exposes it; got {kc:.3e} vs {ki:.3e}')


def _conditional_null_setup(n):
    """S and Z share only H, with an ORACLE conditional. Population blocks vanish."""
    h = RNG.uniform(-1, 1, n)
    p = 1 / (1 + np.exp(-2 * h))
    s = (RNG.uniform(size=n) < p).astype(int)
    z = h + RNG.normal(scale=0.5, size=n)              # Z independent of S given H
    v = whiten(np.column_stack((z, RNG.normal(size=(n, 2)))))
    w = np.linalg.lstsq(v, z, rcond=None)[0].reshape(-1, 1)
    w /= np.linalg.norm(w)
    onehot = np.eye(2)[s]
    oracle = np.column_stack((1 - p, p))               # ORACLE, not a fitted nuisance
    role = make_role('A/SEX', np.column_stack((np.ones(n), h, h ** 2)), onehot - oracle,
                     v.shape[1], classes=2)
    return v, w, role


def test_conditional_null_quadratic_block_concentrates_in_the_POOL_size():
    """The quadratic block averages over all valid rows, so it concentrates in n.

    Declared in advance: finite-sample blocks are NOT exactly zero, and the larger
    pool must give the smaller value. A nonzero value is a finite-sample artefact,
    not disclosure -- which is why this penalty is not a certificate.
    """
    values = []
    for n in (2000, 32000):
        v, w, role = _conditional_null_setup(n)
        quadratic = QuadraticFeatures.build(1)
        bundle = build_features(quadratic, v, w)
        values.append(quadratic_value(bundle, role.weights(), None, role.n_valid))
    assert values[1] < values[0], values
    assert all(x > 0.0 for x in values), 'finite-sample moments are not exactly zero'


def test_conditional_null_kernel_block_concentrates_in_the_SUBSET_size_not_the_pool():
    """The kernel block's null floor is governed by m, NOT by the representation pool.

    This is a property of the bounded-subset design and is recorded as a limitation in
    METHOD.md section 3.7: the kernel block averages over at most 512 frozen rows, so
    enlarging the pool does not shrink its conditional-null value. Measured decay is
    roughly 1/m. Consequence for interpretation: the fitted kernel block cannot be
    driven below its own m-dependent floor, and a small nonzero value is therefore not
    evidence of residual disclosure.
    """
    import experiments.pcrl_invariant_baselines_v1.invariant_moment as module

    v, w, role = _conditional_null_setup(64000)
    quadratic = QuadraticFeatures.build(1)
    bundle = build_features(quadratic, v, w)
    sigma = reference_bandwidth(bundle.z, 3, 1)['sigma_reference']

    # Enlarging the POOL alone does not move the kernel block: the subset size is what matters.
    original = module.KERNEL_SUBSET_MAX
    try:
        means = {}
        for m in (64, 512, 2048):
            module.KERNEL_SUBSET_MAX = m
            draws = [KernelBlock.build('A/SEX', role, np.arange(len(v)), sigma, seed=s, r=1,
                                       role_index=0).value(bundle.z) for s in range(5)]
            means[m] = float(np.mean(draws))
    finally:
        module.KERNEL_SUBSET_MAX = original

    assert means[2048] < means[512] < means[64], means
    assert all(value > 0.0 for value in means.values())
    # roughly 1/m: a 32x increase in m must buy at least an 8x reduction
    assert means[64] / means[2048] > 8.0, means


def test_misspecified_nuisance_manufactures_a_penalty_without_incremental_leakage():
    """A wrong m(H) makes the repaired penalty fire although Z adds nothing beyond H.

    Declared in advance: the misspecified role must exceed the oracle role by a large
    factor. Therefore a nonzero penalty is NOT evidence of incremental disclosure --
    the repair does not change this, and the limitation is inherited in full.
    """
    n = 20000
    h = RNG.uniform(-1, 1, n)
    p = 1 / (1 + np.exp(-3 * h))
    s = (RNG.uniform(size=n) < p).astype(int)
    z = h + RNG.normal(scale=0.5, size=n)              # Z independent of S given H
    v = whiten(np.column_stack((z, RNG.normal(size=(n, 2)))))
    w = np.linalg.lstsq(v, z, rcond=None)[0].reshape(-1, 1)
    w /= np.linalg.norm(w)
    onehot = np.eye(2)[s]
    basis = np.column_stack((np.ones(n), h, h ** 2))
    oracle = make_role('A/S', basis, onehot - np.column_stack((1 - p, p)), v.shape[1], classes=2)
    wrong = make_role('A/S', basis, onehot - onehot.mean(0), v.shape[1], classes=2)
    quadratic = QuadraticFeatures.build(1)
    bundle = build_features(quadratic, v, w)
    sigma = reference_bandwidth(bundle.z, 5, 1)['sigma_reference']
    oq = quadratic_value(bundle, oracle.weights(), None, oracle.n_valid)
    wq = quadratic_value(bundle, wrong.weights(), None, wrong.n_valid)
    ok = make_kernel('A/S', oracle, n, sigma, seed=5, r=1).value(bundle.z)
    wk = make_kernel('A/S', wrong, n, sigma, seed=5, r=1).value(bundle.z)
    assert wq > 10 * oq, f'quadratic: {wq:.3e} vs oracle {oq:.3e}'
    assert wk > 5 * ok, f'kernel: {wk:.3e} vs oracle {ok:.3e}'


# =================================================================== solver behaviour
def test_nonlinear_term_actually_depends_on_W_and_equals_one_at_the_reference():
    """Guards against a penalty that is secretly constant in W (a silent no-op)."""
    objective, v, w_ref = build_objective(r=4, n=400)
    at_reference = objective.bundle_for(w_ref)
    moved = objective.bundle_for(qr_retract(w_ref + 0.6 * RNG.normal(size=w_ref.shape)))
    for name, penalty in {**objective.local, **objective.coalition}.items():
        a = penalty.nonlinear_value(at_reference)[0]
        b = penalty.nonlinear_value(moved)[0]
        assert a == pytest.approx(1.0, abs=1e-9), name    # 0.5 + 0.5 by construction
        assert abs(a - b) > 1e-6, f'{name} nonlinear term looks constant in W'


def test_descent_decreases_the_objective_and_stays_feasible():
    from experiments.pcrl_invariant_baselines_v1.objective import stiefel_descent
    objective, v, w_ref = build_objective(r=4, n=400)
    result = stiefel_descent(objective, w_ref, max_updates=25)
    assert result['loss'] <= result['initial_loss'] + 1e-15
    assert result['feasibility_max_abs'] < 1e-10
    accepted = [h for h in result['history'][1:] if h['accepted']]
    for a, b in zip(accepted, accepted[1:]):
        assert b['loss'] <= a['loss'] + 1e-15


def test_utility_term_is_covariance_normalised_so_collapse_cannot_pay():
    v = whiten(RNG.normal(size=(800, 6)))
    for r in (1, 3, 6):
        w = qr_retract(RNG.normal(size=(v.shape[1], r)))
        z = v @ w
        assert np.max(abs(z.T @ z / len(z) - np.eye(r))) < 1e-10
