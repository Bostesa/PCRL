"""Data-free regression for the predecessor portfolio path correction."""

import json
import shutil
from pathlib import Path

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r03.a_fit_specs_v2 import fit_spec
from results.pcrl_latent_and_sampled_methods_v1.r03.test_a_specs import _fixture, _sha


def test_fit_spec_requires_predecessor_portfolio_and_pins_renderer(tmp_path):
    campaign, old_portfolio, seal, amendment, repair, state, parent = _fixture(tmp_path)
    predecessor = (tmp_path.parent / "pcrl-method-portfolio-v1" /
                   "results/pcrl_method_portfolio_v1")
    predecessor.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_portfolio), str(predecessor))
    arguments = dict(unit_id="A-FIT-X-000", amendment_path=amendment,
                     amendment_sha256=_sha(amendment), seal_path=seal,
                     seal_sha256=_sha(seal), repair_manifest=repair,
                     queue_state=state, private_parent=parent)
    with pytest.raises(FileNotFoundError):
        fit_spec(portfolio_package=old_portfolio, **arguments)
    spec = fit_spec(portfolio_package=predecessor, **arguments)
    assert spec["argv"][spec["argv"].index("--portfolio-package") + 1] == str(predecessor)
    assert str(Path(__file__).with_name("a_fit_specs_v2.py")) in {
        row["path"] for row in spec["sources"]
    }
    assert any(row["path"].startswith(str(predecessor)) and
               row["path"].endswith("BANK_ADMISSION_RECEIPT.json")
               for row in spec["sources"])
    original_state = json.loads(state.read_text())
    wrong_digest = json.loads(state.read_text())
    wrong_digest["seal"]["sha256"] = "f" * 64
    state.write_text(json.dumps(wrong_digest))
    with pytest.raises(PermissionError, match="not latest"):
        fit_spec(portfolio_package=predecessor, **arguments)
    wrong_path = json.loads(json.dumps(original_state))
    wrong_path["seal"]["path"] = str(seal.parent / "inactive-lock.json")
    state.write_text(json.dumps(wrong_path))
    with pytest.raises(PermissionError, match="not latest"):
        fit_spec(portfolio_package=predecessor, **arguments)
