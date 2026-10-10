"""Data-free raw-schema to public hard-code deployment replay."""

import hashlib
import json

import numpy as np
import pytest
import torch

from jcv.train import Model
from code_witness.deploy import main
from code_witness.model import FixedScaler, HardCodeModel, HardCodeRelease
from code_witness.service import save_service


def test_deploy_uses_only_pinned_raw_schema_and_code_head(tmp_path):
    names = [f"f{i}" for i in range(83)]
    schema_hash = hashlib.sha256("\n".join(names).encode()).hexdigest()
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(names))
    x = np.random.default_rng(2).normal(size=(3, 83)).astype(np.float32)
    raw = tmp_path / "raw.npz"
    np.savez(raw, X=x, feature_names=np.asarray(names))
    source = tmp_path / "rel__s0__U"
    source.mkdir()
    model = Model(83, [2, 6], 0)
    torch.save(model.state_dict(), source / "model.pt")
    source_sha = hashlib.sha256((source / "model.pt").read_bytes()).hexdigest()
    code_model = HardCodeModel(16, (2, 6), 10)
    scaler = FixedScaler(np.zeros((2, 16), np.float32), np.ones((2, 16), np.float32))
    cal = tuple({"schema": "pcrl-portfolio-global-temp-v1", "task": task,
                 "alpha": 1.0} for task in ("income", "occupation"))
    release = HardCodeRelease(code_model, scaler, "synthetic", source_sha, cal)
    saved = save_service(tmp_path / "service", release, source_seed=0,
                         arm_id="T0_B10", role_manifest_sha256="b" * 64,
                         source_schema_sha256=schema_hash,
                         fit_receipt={"selected_epoch": 1})
    out = tmp_path / "release.npz"
    main(["--service-manifest", str(tmp_path / "service/manifest.json"),
          "--service-manifest-sha256", saved["manifest_sha256"],
          "--source-model", str(source / "model.pt"),
          "--source-model-sha256", source_sha,
          "--schema", str(schema), "--schema-sha256", schema_hash,
          "--input", str(raw), "--output", str(out), "--bits", "10"])
    with np.load(out, allow_pickle=False) as archive:
        assert set(archive.files) == {"code0", "prob0", "decision0", "code1", "prob1", "decision1"}
        for purpose in (0, 1):
            assert np.array_equal(archive[f"prob{purpose}"],
                                  release.probabilities_from_code(purpose,
                                                                   archive[f"code{purpose}"]))
    with pytest.raises(SystemExit):
        main(["--export-latent"])
