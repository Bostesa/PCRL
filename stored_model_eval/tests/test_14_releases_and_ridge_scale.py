"""Owner-added regression checks (2026-10-02).

1. Noise releases: documented convention, determinism, untreated identity, fresh-vs-persistent queries.
2. Fixed-penalty ridge R2 (PCRL's native convention, lambda=1e-6 on the unnormalised centred Gram) is
   scale-dependent when the attribute signal sits in a direction whose Gram eigenvalue is comparable to the
   penalty; unpenalised least squares is scale-invariant. test_07 only covers the benign case.
"""
import numpy as np

from stored_model_eval.releases import DOCUMENTED_ADULT_SIGMAS, gaussian_release, repeated_releases


def test_untreated_release_is_identity_and_noise_is_deterministic():
    H = np.random.default_rng(1).normal(size=(50, 8))
    assert np.array_equal(gaussian_release(H, 0.0, seed=0), H)
    a = gaussian_release(H, 1.0, seed=3)
    b = gaussian_release(H, 1.0, seed=3)
    c = gaussian_release(H, 1.0, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    # historical convention: one default_rng(seed) draw over the whole matrix
    expected = H + np.random.default_rng(3).normal(0.0, 1.0, size=H.shape)
    assert np.allclose(a, expected)
    assert DOCUMENTED_ADULT_SIGMAS == (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def test_repeated_releases_fresh_vs_persistent():
    H = np.zeros((200, 4))
    fresh = repeated_releases(H, 2.0, seed=0, n_queries=16, fresh_per_query=True)
    persist = repeated_releases(H, 2.0, seed=0, n_queries=16, fresh_per_query=False)
    # averaging fresh draws shrinks noise ~ 1/sqrt(16); a persistent token does not
    assert fresh.mean(0).std() < 0.75 and abs(persist.mean(0).std() - 2.0) < 0.2
    assert all(np.array_equal(persist[0], persist[q]) for q in range(16))


def _ridge_r2(H, Y, lam):
    Hc = H - H.mean(0)
    Yc = Y - Y.mean(0)
    W = np.linalg.solve(Hc.T @ Hc + lam * np.eye(H.shape[1]), Hc.T @ Yc)
    return 1 - ((Yc - Hc @ W) ** 2).sum() / (Yc ** 2).sum()


def test_fixed_penalty_ridge_is_scale_dependent_near_penalty_scale():
    rng = np.random.default_rng(0)
    n, d = 4000, 64
    s = rng.integers(0, 3, n)
    Y = np.eye(3)[s]
    Z = rng.normal(size=(n, d))
    Z[:, -1] = 1e-5 * (rng.normal(size=n) + (s - 1))  # signal direction, Gram eigenvalue ~ lambda
    H = Z @ np.linalg.qr(rng.normal(size=(d, d)))[0].T
    ridge = [_ridge_r2(c * H, Y, 1e-6) for c in (0.5, 1.0, 2.0)]
    ols = [_ridge_r2(c * H, Y, 0.0) for c in (0.5, 1.0, 2.0)]
    assert max(ols) - min(ols) < 1e-6           # unrestricted linear information unchanged
    assert ridge[2] - ridge[0] > 0.05            # native fixed-penalty score moves materially
    assert ridge[0] < 0.10 < ols[0]              # a rescale alone can turn a 'fail' at tau=0.10 into a 'pass'
