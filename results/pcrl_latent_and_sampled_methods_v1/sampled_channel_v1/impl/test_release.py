"""Data-free one-draw public wire and durable paired-cache checks."""

from pathlib import Path
import gc
import hashlib
import hmac
import json
import sqlite3
import tempfile
import unittest
import warnings

import numpy as np

from bank import FixedBank
from channel import GateWeights, JointPolicy
from release import CachedJointRelease, draw_pair, offline_uniform


def toy_policy():
    source1 = np.array([[.9, .1], [.6, .4]])
    source2 = np.array([[.8, .2], [.7, .3]])
    b1 = FixedBank.from_source(source1, np.arange(2), np.zeros(2, dtype=np.int64))
    b2 = FixedBank.from_source(source2, np.arange(2), np.zeros(2, dtype=np.int64))
    return JointPolicy(b1, b2, (tuple(b1.old_to_global),),
                       (tuple(b2.old_to_global),), GateWeights.zeros(input_width=8))


class CachedReleaseTest(unittest.TestCase):
    def test_mechanism_hash_separates_hard_and_stochastic_laws(self):
        class HardPolicy(JointPolicy):
            law_kind = "deterministic_argmax_v1"

        with tempfile.TemporaryDirectory() as directory:
            base = toy_policy()
            hard = HardPolicy(base.bank1, base.bank2, base.actions1,
                              base.actions2, base.gate)
            basic = dict(cache_path=Path(directory) / "cache.sqlite3",
                         release_id="arm", source_seed=0, role="INNER_SELECTION",
                         realization=0, source_binding_sha256="a" * 64,
                         private_root=b"a" * 32, offline=True)
            stochastic = CachedJointRelease(base, **basic)
            with self.assertRaises(ValueError):
                CachedJointRelease(hard, **basic)
            self.assertNotEqual(stochastic.mechanism_sha256,
                                CachedJointRelease(hard,
                                    **{**basic, "cache_path": Path(directory) / "hard.sqlite3"}
                                ).mechanism_sha256)

    def test_mechanism_hash_binds_control_factor_arrays_and_descriptor(self):
        class FactorPolicy(JointPolicy):
            law_kind = "central_product_v1"

            def mechanism_descriptor(self):
                return {"law_kind": self.law_kind, "factor_order": "r1_then_r2"}

            def mechanism_arrays(self):
                return (*super().mechanism_arrays(), self.factor)

        with tempfile.TemporaryDirectory() as directory:
            base = toy_policy()
            kwargs = dict(cache_path=Path(directory) / "one.sqlite3",
                          release_id="arm", source_seed=0, role="INNER_SELECTION",
                          realization=0, source_binding_sha256="a" * 64,
                          private_root=b"a" * 32, offline=True)
            first = FactorPolicy(base.bank1, base.bank2, base.actions1,
                                 base.actions2, base.gate)
            first.factor = np.array([1.0], np.float64)
            second = FactorPolicy(base.bank1, base.bank2, base.actions1,
                                  base.actions2, base.gate)
            second.factor = np.array([2.0], np.float64)
            first_service = CachedJointRelease(first, **kwargs)
            with self.assertRaises(ValueError):
                CachedJointRelease(second, **kwargs)
            self.assertNotEqual(first_service.mechanism_sha256,
                                CachedJointRelease(second,
                                    **{**kwargs, "cache_path": Path(directory) / "two.sqlite3"}
                                ).mechanism_sha256)

    def test_cached_service_invokes_public_pair_sampler_hook(self):
        class SequentialPolicy(JointPolicy):
            law_kind = "sequential_first_then_conditional_v1"

            def sample_public_pair(self, score1, score2, cell1, cell2, uniform):
                self.calls += 1
                self.first_code = self.actions1[cell1][0]
                return self.first_code, self.actions2[cell2][0]

        with tempfile.TemporaryDirectory() as directory:
            base = toy_policy()
            policy = SequentialPolicy(base.bank1, base.bank2, base.actions1,
                                      base.actions2, base.gate)
            policy.calls = 0
            service = CachedJointRelease(
                policy, cache_path=Path(directory) / "cache.sqlite3",
                release_id="seq", source_seed=0, role="INNER_SELECTION",
                realization=0, source_binding_sha256="a" * 64,
                private_root=b"x" * 32, offline=True)
            request = dict(unit=9, score1=np.array([.9, .1]),
                           score2=np.array([.8, .2]), cell1=0, cell2=0)
            left = service.recipient(recipient=1, **request)
            right = service.recipient(recipient=2, **request)
            self.assertEqual((left["code_id"], right["code_id"]),
                             (policy.actions1[0][0], policy.actions2[0][0]))
            self.assertEqual(policy.calls, 1)

    def test_one_joint_draw_is_cached_and_projected_to_both_recipients(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = toy_policy()
            service = CachedJointRelease(
                policy, cache_path=Path(directory) / "private-cache.sqlite3",
                release_id="paired-arm-s0", source_seed=0, role="INNER_SELECTION",
                realization=0, source_binding_sha256="a" * 64,
                private_root=b"x" * 32, offline=True)
            kwargs = {"unit": 17, "score1": np.array([.9, .1]),
                      "score2": np.array([.8, .2]), "cell1": 0, "cell2": 0}
            first = service.recipient(recipient=1, **kwargs)
            second = service.recipient(recipient=2, **kwargs)
            again = service.recipient(recipient=1, **kwargs)
            self.assertEqual(first, again)
            self.assertEqual(set(first), {"schema", "recipient", "code_id",
                                          "probability", "decision", "release_id",
                                          "public_metadata"})
            self.assertEqual(first["schema"], "SCV1-score-release-v1")
            self.assertEqual(first["probability"],
                             policy.bank1.probability[first["code_id"]].tolist())
            self.assertEqual(second["probability"],
                             policy.bank2.probability[second["code_id"]].tolist())
            self.assertEqual(first["release_id"], second["release_id"])
            self.assertEqual(set(first["public_metadata"]), {
                "model_id", "source_binding_sha256", "canonical_bank_sha256",
                "mechanism_sha256", "code_length_bits", "fallback",
                "sampling_contract"})
            self.assertEqual(first["public_metadata"]["sampling_contract"],
                             "cached_one_joint_draw_v1")
            self.assertFalse(first["public_metadata"]["fallback"])
            self.assertFalse(any(k in first for k in (
                "unit", "fine_cell", "action", "uniform", "seed", "source_score")))
            with self.assertRaises(ValueError):
                service.recipient(recipient=1, **{**kwargs,
                    "score1": np.array([.85, .15])})

    def test_cache_requires_same_private_root_and_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            basic = dict(policy=toy_policy(), cache_path=path, release_id="arm",
                         source_seed=0, role="INNER_SELECTION", realization=0,
                         source_binding_sha256="a" * 64, private_root=b"a" * 32,
                         offline=True)
            CachedJointRelease(**basic)
            with self.assertRaises(ValueError):
                CachedJointRelease(**{**basic, "private_root": b"b" * 32})
            with self.assertRaises(ValueError):
                CachedJointRelease(**{**basic, "source_binding_sha256": "b" * 64})

    def test_restart_returns_identical_joint_pair_without_new_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = dict(policy=toy_policy(), cache_path=Path(directory) / "cache.sqlite3",
                             release_id="arm", source_seed=0, role="INNER_SELECTION",
                             realization=1, source_binding_sha256="a" * 64,
                             private_root=b"a" * 32, offline=True)
            request = dict(unit=17, score1=np.array([.9, .1]),
                           score2=np.array([.8, .2]), cell1=0, cell2=0)
            first = CachedJointRelease(**arguments)
            left = first.recipient(recipient=1, **request)
            del first
            reopened = CachedJointRelease(**arguments)
            right = reopened.recipient(recipient=2, **request)
            self.assertEqual(left, reopened.recipient(recipient=1, **request))
            self.assertEqual(right, reopened.recipient(recipient=2, **request))

    def test_tampered_cached_pair_outside_positive_support_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            service = CachedJointRelease(
                toy_policy(), cache_path=path, release_id="arm", source_seed=0,
                role="INNER_SELECTION", realization=0,
                source_binding_sha256="a" * 64, private_root=b"a" * 32,
                offline=True)
            request = dict(unit=17, score1=np.array([.9, .1]),
                           score2=np.array([.8, .2]), cell1=0, cell2=0)
            service.recipient(recipient=1, **request)
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE releases SET z1=999 WHERE key=?",
                                   (service._cache_key(17),))
            with self.assertRaises(ValueError):
                service.recipient(recipient=2, **request)

    def test_cache_path_must_not_be_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real.sqlite3"
            alias = Path(directory) / "alias.sqlite3"
            CachedJointRelease(toy_policy(), cache_path=real, release_id="arm",
                               source_seed=0, role="INNER_SELECTION", realization=0,
                               source_binding_sha256="a" * 64,
                               private_root=b"a" * 32, offline=True)
            alias.symlink_to(real)
            with self.assertRaises(ValueError):
                CachedJointRelease(toy_policy(), cache_path=alias, release_id="arm",
                                   source_seed=0, role="INNER_SELECTION", realization=0,
                                   source_binding_sha256="a" * 64,
                                   private_root=b"a" * 32, offline=True)

    def test_inverse_cdf_is_right_sided_and_tail_stays_on_positive_support(self):
        pairs = ((0, 1, .4), (0, 2, .6), (9, 9, 0.0))
        self.assertEqual(draw_pair(pairs, .4), (0, 2))
        self.assertEqual(draw_pair(((0, 1, .9999999999999996),
                                    (9, 9, 0.0)), .9999999999999998), (0, 1))

    def test_offline_hmac_is_stable_and_domain_separated(self):
        root = bytes(range(32))
        first = offline_uniform(root, source_seed=0, role="INNER_SELECTION",
                                realization=0, unit=7)
        self.assertEqual(first, offline_uniform(root, source_seed=0,
                                                role="INNER_SELECTION",
                                                realization=0, unit=7))
        self.assertNotEqual(first, offline_uniform(root, source_seed=0,
                                                   role="INNER_SELECTION",
                                                   realization=1, unit=7))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1)
        message = json.dumps(["SCV1-uniform-v1", 0, "INNER_SELECTION", 0, 7],
                             separators=(",", ":"), ensure_ascii=True).encode("ascii")
        value = int.from_bytes(hmac.new(root, message, hashlib.sha256).digest()[:8],
                               "big")
        expected = min((value + .5) / (1 << 64), np.nextafter(1.0, 0.0))
        self.assertEqual(first, expected)

    def test_source_digest_matches_frozen_binary_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            service = CachedJointRelease(
                toy_policy(), cache_path=Path(directory) / "cache.sqlite3",
                release_id="arm", source_seed=0, role="INNER_SELECTION",
                realization=0, source_binding_sha256="ab" * 32,
                private_root=b"x" * 32, offline=True)
            s1 = np.array([.9, .1], np.float64)
            s2 = np.array([.8, .2], np.float64)
            expected = hashlib.sha256(
                b"SCV1-input-v1\0" + s1.astype("<f8").tobytes() +
                s2.astype("<f8").tobytes() + (0).to_bytes(8, "little", signed=True) +
                (0).to_bytes(8, "little", signed=True) + bytes.fromhex("ab" * 32)
            ).hexdigest()
            self.assertEqual(service._input_digest(s1, s2, 0, 0), expected)

    def test_cache_connections_close_after_repeat_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            service = CachedJointRelease(
                toy_policy(), cache_path=Path(directory) / "cache.sqlite3",
                release_id="arm", source_seed=0, role="INNER_SELECTION",
                realization=0, source_binding_sha256="a" * 64,
                private_root=b"x" * 32, offline=True)
            kwargs = dict(unit=17, score1=np.array([.9, .1]),
                          score2=np.array([.8, .2]), cell1=0, cell2=0)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always", ResourceWarning)
                for _ in range(3):
                    service.recipient(recipient=1, **kwargs)
                gc.collect()
            self.assertFalse([item for item in observed
                              if issubclass(item.category, ResourceWarning)])


if __name__ == "__main__":
    unittest.main()
