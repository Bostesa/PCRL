"""Small independent laws for later frozen ATT/INNER privacy replay."""

import numpy as np

from results.pcrl_utility_reference_repair_v1.r3.privacy_replay import (
    reader_book, check_ignore_aliases, control_book,
)


def test_reader_book_freezes_tune_orientation_and_envelopes_inner():
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    tune = {"LR_C1": np.array([.9, .8, .2, .1]),
            "RAW_LR_C1": np.array([.1, .2, .8, .9])}
    inner = {"LR_C1": np.array([.1, .2, .8, .9]),
             "RAW_LR_C1": np.array([.9, .8, .2, .1])}
    result = reader_book(y, tune, y, inner, tuple(tune))
    assert result["validation_winner"] == "LR_C1"  # first wins the tie
    assert result["validation_winner_orientation"] == "reverse"
    assert result["validation_winner_inner_auc"] == 0
    assert result["primary_envelope_auc"] == 1
    assert result["validation"]["RAW_LR_C1"]["selected_orientation"] == "positive"


def test_ignore_aliases_are_exact_local_predictions():
    local = {"income": {"LR_C1": np.array([.1, .2])},
             "occupation_group": {"LR_C1": np.array([.7, .8])}}
    coalition = {"IGNORE_R2__LR_C1": np.array([.1, .2]),
                 "IGNORE_R1__LR_C1": np.array([.7, .8])}
    check_ignore_aliases(coalition, local)
    coalition["IGNORE_R2__LR_C1"][0] = .9
    try:
        check_ignore_aliases(coalition, local)
    except PermissionError:
        pass
    else:
        raise AssertionError("changed coalition alias was accepted")


def test_control_book_checks_null_and_planted_engineering_copy():
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    unit = np.arange(4, dtype=np.int64)
    book = control_book(y, unit, np.full(4, .5),
                        np.array([.01, .02, .98, .99]))
    assert book["null_auc"] == .5
    assert book["planted_auc"] == 1
    assert book["gates"]["planted_at_least_0.99"]
    assert book["gates"]["null_exact"]
