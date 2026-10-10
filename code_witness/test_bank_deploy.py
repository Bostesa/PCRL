"""Data-free complete-wire and refusal tests for recipient-local bank deployment."""

import hashlib
import json

import joblib
import numpy as np
import pytest
import torch

from dpc.deploy import Refused
from jcv.train import Model
from qpc import release as RL
from qpc import stagea as SA
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder

from code_witness import bank_deploy as BD


class _Head:
    def __init__(self, classes):
        self.classes_ = np.arange(classes)
        self.weights = np.arange(16 * classes, dtype=np.float64).reshape(16, classes) / 40

    def predict_proba(self, latent):
        value = np.asarray(latent, np.float64) @ self.weights
        value -= value.max(axis=1, keepdims=True)
        exp = np.exp(value)
        return exp / exp.sum(axis=1, keepdims=True)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_bank(tmp_path, monkeypatch):
    names = tuple(f"f{i}" for i in range(83))
    schema_hash = hashlib.sha256("\n".join(names).encode()).hexdigest()
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(names))
    x = np.random.default_rng(42).normal(size=(96, 83)).astype(np.float32)
    raw = tmp_path / "raw.npz"
    np.savez(raw, X=x, feature_names=np.asarray(names))
    hcal_root = tmp_path / "hcal"
    source = hcal_root / "admitted" / "rel__s0__U"
    source.mkdir(parents=True)
    torch.save(Model(83, [2, 6], 0).state_dict(), source / "model.pt")
    source_sha = _sha(source / "model.pt")
    for i, k in enumerate((2, 6)):
        joblib.dump(_Head(k), source / f"head_{i}.joblib")
    (source / "record.json").write_text("{}\n")
    files = {name: _sha(source / name) for name in
             ("model.pt", "head_0.joblib", "head_1.joblib", "record.json")}
    (source / "COMPLETE.json").write_text(json.dumps({"id": "rel__s0__U", "files": files}))
    encoder = FrozenUEncoder.load(source / "model.pt", source_sha, names, schema_hash, 0)
    p = []
    for i, k in enumerate((2, 6)):
        h = encoder.encode_one(i, x, names, "DEPLOYMENT")
        score = _Head(k).predict_proba(h.astype(np.float64))
        p.append(score)
    pol = [SA.recipient_fit(p[i], p[i].argmax(axis=1), (2, 6)[i], 2, i + 1,
                            starts=("source",), rounds=4)[0] for i in (0, 1)]
    qpc_root = tmp_path / "qpc"
    families = {}
    for bank, family in (("W_Q", "DIRECT-TASK"), ("W_TSTAR", "FINE-TASK")):
        pair = SA.assemble_pair(pol[0], pol[1], family, 2, 2,
                                meta={"teacher": "U", "config": f"U|{family}|i8o64",
                                      "teacher_model_sha256": source_sha,
                                      "feature_names_sha256": schema_hash})
        policy = qpc_root / "run" / "units" / f"pol__s0__U_{family}_i8o64" / "policy.json"
        policy.parent.mkdir(parents=True)
        RL.save_policy(pair, policy)
        families[bank] = {
            "policy_sha256": _sha(policy), "policy_fingerprint": pair.fingerprint(),
            "decoder_sha256": None if bank == "W_Q" else "d" * 64,
            **{f"table{i}": (pair.p1, pair.p2)[i].token_proto.tolist() for i in (0, 1)},
            **{f"token_class{i}": (pair.p1, pair.p2)[i].token_class.tolist() for i in (0, 1)},
        }
    table = tmp_path / "tables.json"
    table.write_text(json.dumps({"schema": "code-utility-public-bank-tables-v1",
                                 "seed": 0, "bit_count": 10,
                                 "bit_order": "c0_least_significant",
                                 "source_model_sha256": source_sha,
                                 "feature_schema_sha256": schema_hash,
                                 "families": families}))
    monkeypatch.setattr(BD, "load_decoder", lambda path, pair, digest:
                        ("H-GLOBAL-TEMP", (pair.p1.token_proto, pair.p2.token_proto),
                         digest, None))
    return dict(names=names, schema=schema, schema_hash=schema_hash,
                raw=raw, x=x, source_root=hcal_root, source_sha=source_sha,
                qpc_root=qpc_root, table=table, table_sha=_sha(table))


@pytest.mark.parametrize("bank", ["W_Q", "W_TSTAR"])
@pytest.mark.parametrize("purpose", [0, 1])
def test_bank_deployment_emits_only_code_derived_recipient_wire(synthetic_bank,
                                                                  tmp_path, bank, purpose):
    s = synthetic_bank
    out = tmp_path / f"{bank}_{purpose}.npz"
    args = ["--bank", bank, "--purpose", str(purpose), "--seed", "0",
            "--source-root", str(s["source_root"]), "--source-sha256", s["source_sha"],
            "--qpc-root", str(s["qpc_root"]), "--public-table", str(s["table"]),
            "--public-table-sha256", s["table_sha"], "--schema", str(s["schema"]),
            "--schema-sha256", s["schema_hash"], "--input", str(s["raw"]),
            "--output", str(out)]
    BD.main(args)
    with np.load(out, allow_pickle=False) as archive:
        assert set(archive.files) == {"code", "prob", "decision"}
        pair = RL.load_policy(
            s["qpc_root"] / "run" / "units" /
            f"pol__s0__U_{'DIRECT-TASK' if bank == 'W_Q' else 'FINE-TASK'}_i8o64" /
            "policy.json")
        score = _Head((2, 6)[purpose]).predict_proba(
            FrozenUEncoder.load(s["source_root"] / "admitted/rel__s0__U/model.pt",
                                s["source_sha"], s["names"], s["schema_hash"], 0)
            .encode_one(purpose, s["x"], s["names"], "DEPLOYMENT"))
        token, _, _ = RL.encode((pair.p1, pair.p2)[purpose], score)
        expected_code = ((token[:, None] >> np.arange(10)) & 1).astype(np.uint8)
        assert np.array_equal(archive["code"], expected_code)
        table = np.asarray(json.loads(s["table"].read_text())["families"][bank][f"table{purpose}"])
        assert np.array_equal(archive["prob"], table[token])
        assert np.array_equal(archive["decision"], table[token].argmax(axis=1))


def test_bank_deployment_refuses_wrong_input_source_and_hidden_exports(synthetic_bank,
                                                                        tmp_path):
    s = synthetic_bank
    args = ["--bank", "W_Q", "--purpose", "0", "--seed", "0",
            "--source-root", str(s["source_root"]), "--source-sha256", s["source_sha"],
            "--qpc-root", str(s["qpc_root"]), "--public-table", str(s["table"]),
            "--public-table-sha256", s["table_sha"], "--schema", str(s["schema"]),
            "--schema-sha256", s["schema_hash"], "--input", str(s["raw"]),
            "--output", str(tmp_path / "out.npz")]
    wrong_source = args.copy()
    wrong_source[wrong_source.index("--source-sha256") + 1] = "f" * 64
    with pytest.raises(ValueError):
        BD.main(wrong_source)
    reordered = tmp_path / "reordered.npz"
    np.savez(reordered, X=s["x"][:, ::-1],
             feature_names=np.asarray(s["names"][::-1]))
    wrong_input = args.copy()
    wrong_input[wrong_input.index("--input") + 1] = str(reordered)
    with pytest.raises(Refused):
        BD.main(wrong_input)
    excluded = tmp_path / "excluded.npz"
    np.savez(excluded, X=s["x"], feature_names=np.asarray(s["names"]),
             SEX=np.zeros(len(s["x"])))
    wrong_input[wrong_input.index("--input") + 1] = str(excluded)
    with pytest.raises(Refused):
        BD.main(wrong_input)
    with pytest.raises(SystemExit):
        BD.main(args + ["--export-teacher"])
    assert not (tmp_path / "out.npz").exists()
