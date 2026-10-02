"""Test 10: the repeated-release averaging attacker beats the single-release attacker when the contract issues
fresh noise per query, and is identical to it under a persistent token (and then flagged invalid)."""
import numpy as np

from stored_model_eval.access import ReleaseContract
from stored_model_eval.attackers import NoiseLRTAttacker, ReleaseChannel, RepeatedReleaseAttacker
from stored_model_eval.fixtures import make_synthetic
from stored_model_eval.metrics import auc_binary


def _run(noise, N, auth):
    fx = make_synthetic("direct", n_units=1000, d=4, seed=21, signal=0.6)
    H, S = fx["H"], fx["S"]
    pop, tgt = np.arange(600), np.arange(600, 1000)
    contract = ReleaseContract(noise=noise, sigma=3.0)
    lrt = NoiseLRTAttacker(contract).fit(H[pop], S[pop], auth=auth, synthetic=True)
    chan = ReleaseChannel(contract, seed=7)
    tokens = fx["row_ids"][tgt]
    single = lrt.predict_proba(chan.release(H[tgt], tokens, 0), n_eff=1)
    rep = RepeatedReleaseAttacker(lrt, chan, N)
    multi = rep.predict_proba(H[tgt], tokens)
    return S[tgt], single, multi, rep.access_record()


def test_fresh_noise_repeated_release_gains(auth, record_property):
    y, single, multi, rec = _run("fresh_per_query", 16, auth)
    a1, a16 = auc_binary(y == 1, single[:, 1]), auc_binary(y == 1, multi[:, 1])
    record_property("auc_single", a1); record_property("auc_N16", a16)
    assert a16 > a1 + 0.05, (a1, a16)
    assert rec.valid and rec.effective_queries == 16 and rec.tag == "A3(16)"


def test_persistent_token_repeated_release_equals_single(auth, record_property):
    y, single, multi, rec = _run("persistent_token", 16, auth)
    record_property("auc_single_eq_N16", auc_binary(y == 1, single[:, 1]))
    assert np.array_equal(single, multi)
    assert auc_binary(y == 1, single[:, 1]) == auc_binary(y == 1, multi[:, 1])
    assert not rec.valid and rec.effective_queries == 1 and "persistent_token" in rec.invalid_reason
