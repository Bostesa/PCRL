"""Synthetic capacity and exact-output compilation checks; no Adult arrays."""

import numpy as np
import pytest

from code_witness.bank import bits_to_tokens, tokens_to_bits, compile_release, decoder_capacity, decode_from_code


@pytest.mark.parametrize("width", [10, 16])
def test_token_bits_roundtrip_and_public_class_is_in_id(width):
    tokens = np.array([0, 15, 16, 320, (1 << width) - 1], np.int64)
    code = tokens_to_bits(tokens, width)
    assert code.dtype == np.uint8
    assert code.shape == (len(tokens), width)
    assert np.array_equal(bits_to_tokens(code), tokens)
    with pytest.raises(ValueError):
        tokens_to_bits(np.array([1 << width]), width)


def test_compiled_bank_reconstructs_exact_probabilities_and_decisions():
    table = np.array([[0.8, 0.2], [0.3, 0.7], [0.8, 0.2]], np.float64)
    token = np.array([0, 2, 1], np.int64)
    out = compile_release(token, table, 10)
    assert np.array_equal(out["code"], tokens_to_bits(token, 10))
    assert np.array_equal(out["probabilities"], table[token])
    assert np.array_equal(out["decision"], np.array([0, 0, 1]))
    decoded = decode_from_code(out["code"], table)
    assert np.array_equal(decoded["probabilities"], out["probabilities"])
    assert np.array_equal(decoded["decision"], out["decision"])
    # Tokens 0 and 2 disclose distinct IDs despite identical probability rows.
    assert not np.array_equal(out["code"][0], out["code"][1])
    with pytest.raises(ValueError):
        compile_release(np.array([3]), table, 10)


def test_decoder_capacity_counts_token_even_if_probabilities_coincide():
    table = np.array([[0.8, 0.2], [0.3, 0.7], [0.8, 0.2]], np.float64)
    result = decoder_capacity(table, np.array([0, 1, 0]), 10)
    assert result["global_states"] == 3
    assert result["capacity_bits_required"] == 2
    assert result["class_state_counts"] == [2, 1]
    with pytest.raises(ValueError):
        decoder_capacity(table, np.array([1, 1, 0]), 10)
