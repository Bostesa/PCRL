"""The CLI loads the methodology role's protocol_config.json (translated) or defaults; invalid configs refuse."""
import json
from pathlib import Path

import pytest

from stored_model_eval.config import load_protocol

METH = Path(__file__).resolve().parents[2] / "results/combined_evaluation_preparation_v1/notes/methodology/protocol_config.json"


def test_defaults_and_override(tmp_path):
    c = load_protocol(None)
    assert c["bars"] == [0.52, 0.55, 0.60] and c["_meta"]["source"] == "defaults"
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"bootstrap": {"n_boot": 10}, "new_key": 1}))
    c = load_protocol(p)
    assert c["bootstrap"]["n_boot"] == 10 and c["bootstrap"]["alpha"] == 0.05
    assert "new_key" in c["_meta"]["unknown_keys"]
    p.write_text(json.dumps({"support": {"min_class_support": 1}}))
    with pytest.raises(ValueError):
        load_protocol(p)


@pytest.mark.skipif(not METH.exists(), reason="methodology protocol_config.json not present")
def test_methodology_config_translated():
    c = load_protocol(METH)
    assert c["_meta"]["format"].startswith("methodology")
    assert c["roles"]["score"] == "assessment"
    assert c["support"]["min_class_support"] >= 2
    assert 0 < c["bootstrap"]["alpha"] < 0.5
