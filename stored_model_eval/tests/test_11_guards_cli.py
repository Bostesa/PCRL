"""Guards: no scientific fit on non-synthetic data without --execute-scientific-fits; no network; CLI
default mode performs no fit; metric sanity vs sklearn."""
import json
import socket

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from stored_model_eval import cli
from stored_model_eval.attackers import LinearAttacker
from stored_model_eval.fixtures import make_synthetic, write_manifest
from stored_model_eval.guards import FitAuthorization, NetworkRefused, ScientificFitRefused
from stored_model_eval.metrics import auc_binary


def test_fit_refused_without_flag():
    fx = make_synthetic("direct", n_units=200, seed=22)
    with pytest.raises(ScientificFitRefused):
        LinearAttacker().fit(fx["H"], fx["S"], fx["H"], fx["S"], auth=FitAuthorization(), synthetic=False)
    with pytest.raises(ScientificFitRefused):
        LinearAttacker().fit(fx["H"], fx["S"], fx["H"], fx["S"], auth=FitAuthorization(synthetic=True),
                             synthetic=False)


def test_cli_refuses_real_manifest_fit_and_blocks_network(tmp_path, monkeypatch):
    fx = make_synthetic("direct", n_units=200, seed=23)
    mp = write_manifest(fx, tmp_path)
    man = json.loads(mp.read_text())
    man["synthetic"] = False  # pretend these are real arrays
    from stored_model_eval.admission import sha256_file  # noqa: F401
    mp.write_text(json.dumps(man))
    orig, orig_cc = socket.socket, socket.create_connection
    try:
        with pytest.raises(SystemExit, match="REFUSED"):
            cli.main(["fit-attackers", "--manifest", str(mp), "--out-dir", str(tmp_path / "o")])
        assert not (tmp_path / "o").exists()
        with pytest.raises(NetworkRefused):
            socket.socket()
        assert cli.main(["--dry-run", "fit-attackers", "--manifest", str(mp), "--out-dir", str(tmp_path / "o2"),
                         "--execute-scientific-fits"]) == 0
        assert not (tmp_path / "o2").exists()
    finally:
        socket.socket, socket.create_connection = orig, orig_cc


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)  # with ties
    assert abs(auc_binary(y == 1, s) - roc_auc_score(y, s)) < 1e-12
