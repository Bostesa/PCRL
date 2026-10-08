"""Independent arithmetic for S4's inherited Team A score-bank admission.

No candidate, inherited endpoint, S2 adapter, or shared scoring implementation is
imported. This module has no command-line entry point and never opens a role file
on import. S4 must place any Adult invocation behind its separately sealed S3
unit. Only role-only U0 reference and S2 bank/token artifacts are valid inputs.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


SCHEMA = "pcrl-a-s4-score-bank-parity-v1"
S2_REFERENCE_RUNTIME = {"python": "3.13.7", "numpy": "2.4.3"}
ROLES = ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION")
ROLE_COUNTS = {"OSF_DEFENSE_FIT": (15434, 15428),
               "ATTACK_FIT_NEW": (4064, 4061),
               "INNER_SELECTION": (2235, 2234)}
BANK_MEMBERS = {"probability", "token_class", "token_id", "input_class",
                "fine_to_hcal_token"}
TOKEN_MEMBERS = {"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"}
REFERENCE_MEMBERS = {"U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2",
                     "row_id", "unit"}
ROLE_ROW_SHA256 = {
    "OSF_DEFENSE_FIT": "2fa6d03e663ee2816fe5616cc1a2b92e7b9c101d19efa908383f238f92b40329",
    "ATTACK_FIT_NEW": "22a36fc44e0932eb497644533345215bdaf0be0cc206bbcff38fcc028495acb8",
    "INNER_SELECTION": "f533da6a43f773d1d749da536bf70b387623c2b0f7a8aac6c98b96404dcb9598",
}
ROLE_PAIR_SHA256 = {
    "OSF_DEFENSE_FIT": "8dc63c5c35cfaebe8f348262d7f55be8c8c04412ee3f55798de549e3276aeb7f",
    "ATTACK_FIT_NEW": "e2de0109a93a6240062ff29eb1af4b73785e2ca04c2149f822eb2c8ec13a83e9",
    "INNER_SELECTION": "097f5f867be8a9b04ae7cd847d4d1a308aaa5f38693c5a4b6d6316f73c3030c3",
}
HCAL_MANIFEST_SHA256 = "52451ae3d99fa9c1182b820bd534b48feb7a926057b3b4a2a0175a9c9cd76c5a"
HCAL_ADMISSION_SHA256 = "1e03b084499a175edf64a4cd2527cd56059a3088ac0aaa80c94550664d2e435b"
LRA_MANIFEST_SHA256 = "67f6865e308fdf743cccb04c8fa77bb1604d9a5d8cd20a522b59f75a61fd47fc"
LRA_ADMISSION_SHA256 = "de87920621664f814ad4e01239f131224b0eb66b4c96bcdc2f589e7446a1bf9d"
ORDERED_SCHEMA_SHA256 = "2ce68c8a73e98f548b361ac964d718d9cc38155f68ac44dc80a9d27983495dff"
TEACHER_SHA256 = {
    0: "5061a0074af74d21b53aca80b96d8b83b85a3e43cdb9e6758b2551be9045b9a7",
    1: "f52bef54ef77faf51cb6f55273e54f75ea094c4379bbf73b57a3941ce2f7a74c",
}
FINE_SHA256 = {
    0: "f92bcd9bf313a937ff9caa79d5bd0779b44d583d0f732a7df8cad50a8f7cfa27",
    1: "6ab9ef7dd31375bd4c7733134b24b5192523ebbfd32011d47690a3ebe0d1432a",
}
# Per row: release ID, r1/r2 complete token counts, policy file, original frozen
# bank NPZ file, decoder internal hash, decoder file hash. None denotes D0.
OUTPUT_PINS = {
    (0, "tstar"): ("U|FINE-TASK|i8o64|H-GLOBAL-TEMP", (16, 321),
        "535725e96687083e7a5d345762ea52dd1ded7f5ad12fcb8ee7f4fcc3d6206236",
        "378888571b86af31334312ec0ceb2be03ebe49256c0ab198d81f05e44351672c",
        "2d978e76881af4fc6a59432537885f663f1327aaf34ae293a56eb347b6c4a77b",
        "83b88643ba3b81207558bd2936b15a0ea9a59ea3346dcf34b756a325cfcb6d9d"),
    (1, "tstar"): ("U|FINE-TASK|i8o64|H-GLOBAL-TEMP", (16, 321),
        "7e7bbd694379b21ceee773edd79147b6e4dbc1ea04957cb3daa849747603e0f2",
        "790977f0405fd8079d05b8c97647c4d6c301ba7481439d81a6cd7480af80d06a",
        "e8be75312624cf9a76ffb7b518029b1c9594be910c484ea500833b10326de625",
        "0d534c671ff3a62e2e08dd5f04eb6704627fb784c9212e58bdaec864f9f224ae"),
    (0, "pstar"): ("U|JOINT|i8o64|l0.1|H-CLASS-TEMP", (14, 270),
        "acd58d68d89f3a53a8ed111625ae0a0b636c334e9a1f2d66bbdc518d33faf47c",
        "748127a2ef58b809fbbd485c795c1298d6794b526ad75d3d2a3562515a317ea0",
        "30f53a7ef4c27631d221f0d28e455060fe8efa54dbd1ad578599280d2d93f717",
        "a1a5f7fb45a9f61f89f0809d86ef072461cf40efde5c5660a1c6a4b642de2841"),
    (1, "pstar"): ("U|JOINT|i8o64|l0.1|H-CLASS-TEMP", (13, 299),
        "4a72d860ef968199f6d566acf1319e50a61bcf9ab19b4addf39b5a5188737c20",
        "9436ebfcd3df2e79e6a84578460bf12525f5c4d3ca6c3494f4ddc61059a80fca",
        "ac28800e9611bd8864a8664693df88adae844801d6b8870225717033f74a18dd",
        "89d2c1e1df8bb0f4c52cb32e336792ecd9fbaa2574d2add7b8123c117f5cc4b8"),
    (0, "hcal_local_d0"): ("U|LOCAL|i8o64|l0.1", (16, 321),
        "e0673ec6dc430b5d036f2c617fe336293d405ae2616e9839ecdae3a2c845e6ba",
        "1881ea8d9832cfd37bebe2ea70ffdc1abf62e3ea680f0d0c8779f4469bee7ab2", None, None),
    (1, "hcal_local_d0"): ("U|LOCAL|i8o64|l0.1", (16, 321),
        "2ed3843cd04d81230572ccf8623307b275c853d8c20b3bd4a6e17458b215eb6c",
        "28c1e85d60cc4220e7509fd9a328f8ddbebe1b264190c4881553b9203e3f9506", None, None),
    (0, "hcal_seq12_d0"): ("U|SEQ-12|i8o64|l0.1", (16, 295),
        "dc06383267942182fee5047eb9ac7f4f11f84ed8a418598ffe298eae7c52cee2",
        "bc8f218fa05bf42fbe2a93ef1894d916cfcbe8afe24fc7921d958603b86c3fac", None, None),
    (1, "hcal_seq12_d0"): ("U|SEQ-12|i8o64|l0.1", (16, 302),
        "7f1dcdfb2728057e97776a9263624b2279eec791c6c53df4f5654b56eeba9ae5",
        "9aeff70976322cf03b07db04c20b8547be1178348f332a7d2551986bf0e46a44", None, None),
    (0, "lra_seq21"): ("U|SEQ-21|i8o64|l0.1", (14, 318),
        "67a6d0f71efade1b43579a88fac0d5515619737dcb803a54510a1c9bfff4270c",
        "12be2287a6b539a310c0c64744f47e65064089e62c84da954e9a9c7d563bd9f3", None, None),
    (1, "lra_seq21"): ("U|SEQ-21|i8o64|l0.1", (12, 320),
        "da02bde30d7c458eb39382064f359422780c63ad1c6b9fa346c0df5cf48ed9dd",
        "7aad7366ac4b77a2b748c4523fdf05286980e2f9619aa4477ef4895a4f71a3d5", None, None),
}


class ParityError(ValueError):
    """A source, custody, or discrete release identity fails closed."""


class NumericalParityError(ParityError):
    """Pinned source table and independent float arithmetic differ bitwise."""

    def __init__(self, stage: str, comparison: dict):
        super().__init__(f"{stage}: independent floating table does not reproduce bitwise")
        self.stage = stage
        self.max_abs_difference = comparison["max_abs_difference"]
        self.max_ulp_difference = comparison["max_ulp_difference"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(observed: str, expected: str, label: str) -> None:
    if observed != expected or not isinstance(expected, str) or len(expected) != 64:
        raise ParityError(f"{label} SHA-256 mismatch")


def require_members(observed: set[str], expected: set[str]) -> None:
    if observed != expected:
        raise ParityError("NPZ member set differs from role-only contract")


def array_digest(a: np.ndarray) -> str:
    """Independent reproduction of S2's dtype+shape+C-bytes custody digest."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _bytes(a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a)
    return str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()


def _i64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _f64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def fine_fingerprint(fine: dict) -> str:
    k = int(fine["K"])
    parts = (np.int64(k), _i64(fine["cell_class"]).astype("<i8"),
             _f64(fine["centroid"]).astype("<f8"),
             np.asarray(fine["fallback"], dtype=np.uint8),
             _i64(fine["n"]).astype("<i8"),
             _f64(fine["S"]).astype("<f8"),
             _f64(fine["A"]).astype("<f8"))
    h = hashlib.sha256()
    for part in parts:
        h.update(np.ascontiguousarray(part).tobytes())
    return h.hexdigest()


def check_fine(fine: dict) -> None:
    if fine.get("kind") != "qpc.FinePartition" or float(fine.get("eps")) != 1e-12:
        raise ParityError("fine partition kind or smoothing rule differs")
    k = int(fine["K"])
    classes = _i64(fine["cell_class"])
    centers = _f64(fine["centroid"])
    n = _i64(fine["n"])
    sums = _f64(fine["S"])
    fallback = np.asarray(fine["fallback"], dtype=bool)
    if (classes.ndim != 1 or centers.shape != (len(classes), k) or
            sums.shape != centers.shape or n.shape != classes.shape or
            fallback.shape != classes.shape or np.any(np.diff(classes) < 0) or
            np.any(classes < 0) or np.any(classes >= k)):
        raise ParityError("fine partition dimensions/order invalid")
    for c in range(k):
        idx = np.flatnonzero(classes == c)
        if len(idx) == 0 or (np.any(fallback[idx]) and
                            (len(idx) != 1 or n[idx[0]] != 0)) or \
                np.any(n[idx[~fallback[idx]]] <= 0):
            raise ParityError("fine class or reserved fallback invalid")
    if np.any(centers <= 0) or np.any(~np.isfinite(centers)) or \
            np.any(np.abs(centers.sum(axis=1) - 1.0) > 1e-12):
        raise ParityError("fine centroids invalid")
    strict_argmax(centers, classes)
    if "fingerprint" in fine:
        require_sha(fine_fingerprint(fine), fine["fingerprint"], "fine content")


def smooth(mean: np.ndarray, predicted_class: int) -> np.ndarray:
    m = _f64(mean)
    k = m.shape[-1]
    extra = np.zeros_like(m)
    extra[int(predicted_class)] = 1e-12
    return (m + 1e-12 * np.ones_like(m) + extra) / (1.0 + (k + 1) * 1e-12)


def token_prototypes(fine: dict, cell_token) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate fine sufficient statistics in increasing cell order."""
    check_fine(fine)
    cell_token = _i64(cell_token)
    classes = _i64(fine["cell_class"])
    counts = _i64(fine["n"])
    sums = _f64(fine["S"])
    if cell_token.shape != classes.shape or np.any(cell_token < 0):
        raise ParityError("cell-token map invalid")
    seen = {}
    canonical = np.empty_like(cell_token)
    for f, t in enumerate(cell_token):
        key = int(t)
        if key not in seen:
            seen[key] = len(seen)
        canonical[f] = seen[key]
    if not np.array_equal(canonical, cell_token):
        raise ParityError("token IDs are not canonical first-appearance IDs")
    tcount = len(seen)
    token_class = np.full(tcount, -1, dtype=np.int64)
    token_n = np.zeros(tcount, dtype=np.int64)
    token_s = np.zeros((tcount, int(fine["K"])), dtype=np.float64)
    token_fallback = np.zeros(tcount, dtype=bool)
    for f, t in enumerate(cell_token):
        t = int(t)
        c = int(classes[f])
        if token_class[t] not in (-1, c):
            raise ParityError("token mixes predicted classes")
        token_class[t] = c
        token_n[t] += counts[f]
        token_s[t] = token_s[t] + sums[f]
        token_fallback[t] = token_fallback[t] or bool(fine["fallback"][f])
    if np.any(token_fallback & (token_n > 0)):
        raise ParityError("reserved fallback token includes fitting rows")
    bank = np.empty_like(token_s)
    for t in range(tcount):
        mean = token_s[t] / token_n[t] if token_n[t] > 0 else \
            np.full(int(fine["K"]), 1.0 / int(fine["K"]))
        bank[t] = smooth(mean, int(token_class[t]))
    return token_class, bank, token_n


def policy_fingerprint(policy: dict, bank: np.ndarray, token_class: np.ndarray) -> str:
    f = policy["fine"]
    arrays = (np.int64(f["K"]), _i64(f["cell_class"]).astype("<i8"),
              _f64(f["centroid"]).astype("<f8"),
              _i64(policy["cell_token"]).astype("<i8"),
              token_class.astype("<i8"), bank.astype("<f8"))
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def rebuild_policy_bank(policy: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if policy.get("kind") != "qpc.Policy":
        raise ParityError("policy kind differs")
    classes, bank, counts = token_prototypes(policy["fine"], policy["cell_token"])
    if (not np.array_equal(classes, _i64(policy["token_class"])) or
            not np.array_equal(counts, _i64(policy["token_n"]))):
        raise ParityError("stored policy token class/count differs")
    proto_parity = compare_arrays(bank, _f64(policy["token_proto"]))
    if proto_parity["status"] == "FAIL":
        raise ParityError("stored policy token prototype shape differs")
    if proto_parity["status"] != "PASS":
        raise NumericalParityError("policy token prototype", proto_parity)
    if "fingerprint" in policy:
        require_sha(policy_fingerprint(policy, bank, classes), policy["fingerprint"],
                    "policy content")
    strict_argmax(bank, classes)
    return bank, classes, np.arange(len(classes), dtype=np.int64)


def pair_fingerprint(policy_pair: dict) -> str:
    pieces = []
    for r in (1, 2):
        p = policy_pair[f"p{r}"]
        bank, classes, _ = rebuild_policy_bank(p)
        pieces.append(policy_fingerprint(p, bank, classes))
    return hashlib.sha256("".join(pieces).encode()).hexdigest()


def strict_argmax(bank: np.ndarray, classes: np.ndarray) -> None:
    bank = _f64(bank)
    classes = _i64(classes)
    if bank.ndim != 2 or classes.shape != (len(bank),) or \
            np.any(~np.isfinite(bank)) or np.any(bank < 0) or \
            np.any(np.abs(bank.sum(axis=1) - 1.0) > 1e-12):
        raise ParityError("bank probability shape/range invalid")
    if np.any(classes < 0) or np.any(classes >= bank.shape[1]):
        raise ParityError("bank token class invalid")
    for row, c in zip(bank, classes):
        if row[c] <= np.max(np.delete(row, c)):
            raise ParityError("bank decision is not strict")


def assign_fine_cells(probability: np.ndarray, decision: np.ndarray, fine: dict) -> np.ndarray:
    """Masked KL in increasing class order; NumPy argmin chooses lower cell ID."""
    check_fine(fine)
    p = _f64(probability)
    d = _i64(decision)
    classes = _i64(fine["cell_class"])
    q = _f64(fine["centroid"])
    k = int(fine["K"])
    if p.ndim != 2 or p.shape[1] != k or d.shape != (len(p),) or \
            np.any(~np.isfinite(p)) or np.any(p < 0) or \
            np.any(np.abs(p.sum(axis=1) - 1.0) > 1e-12) or \
            not np.array_equal(np.argmax(p, axis=1), d):
        raise ParityError("U0 probability/decision contract differs")
    assigned = np.full(len(p), -1, dtype=np.int64)
    for c in range(k):
        rows = np.flatnonzero(d == c)
        cells = np.flatnonzero(classes == c)
        if len(rows) == 0:
            continue
        if len(cells) == 1:
            assigned[rows] = cells[0]
            continue
        divergence = np.zeros((len(rows), len(cells)), dtype=np.float64)
        for j in range(k):
            pj = p[rows, j:j + 1]
            log_p = np.log(np.where(pj > 0, pj, 1.0))
            divergence = divergence + np.where(
                pj > 0, pj * (log_p - np.log(q[cells, j])[None, :]), 0.0)
        assigned[rows] = cells[np.argmin(divergence, axis=1)]
    if np.any(assigned < 0):
        raise ParityError("fine assignment omitted a row")
    return assigned


def temperature_bank(q0: np.ndarray, token_class: np.ndarray,
                     alpha: float | list[float]) -> np.ndarray:
    q0 = _f64(q0)
    classes = _i64(token_class)
    strict_argmax(q0, classes)
    alphas = ([float(alpha)] * q0.shape[1] if np.isscalar(alpha) else
              [float(a) for a in alpha])
    if len(alphas) != q0.shape[1] or any(a < .25 or a > 4.0 for a in alphas):
        raise ParityError("decoder temperature outside registered family")
    table = q0.copy()
    for c, a in enumerate(alphas):
        if a == 1.0:
            continue
        mask = classes == c
        if not np.any(mask):
            continue
        z = a * np.log(q0[mask])
        exp = np.exp(z - z.max(axis=1)[:, None])
        denominator = np.zeros(len(exp), dtype=np.float64)
        for j in range(exp.shape[1]):
            denominator = denominator + exp[:, j]
        table[mask] = exp / denominator[:, None]
    strict_argmax(table, classes)
    return table


def decoder_internal_hash(decoder: dict) -> str:
    body = {k: v for k, v in decoder.items() if k != "decoder_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def decoded_bank(policy_pair: dict, decoder: dict | None,
                 expected_release: str, expected_internal_sha: str | None) -> dict[int, np.ndarray]:
    if policy_pair.get("kind") != "qpc.PolicyPair":
        raise ParityError("policy pair kind differs")
    pair_hash = pair_fingerprint(policy_pair)
    if policy_pair.get("fingerprint") != pair_hash:
        raise ParityError("policy pair fingerprint differs")
    banks = {}
    if decoder is not None:
        if (decoder.get("kind") != "hcal.CalibratedDecoderPair" or
                decoder.get("schema") != "hcal-decoder-v1" or
                decoder.get("release_id") != expected_release or
                decoder.get("policy_pair_fingerprint") != pair_hash or
                decoder.get("binding") != {k: policy_pair["config"].get(k) for k in
                                            ("teacher_model_sha256", "feature_names_sha256")} or
                decoder.get("decoder_sha256") != expected_internal_sha):
            raise ParityError("decoder identity or binding differs")
        require_sha(decoder_internal_hash(decoder), expected_internal_sha,
                    "decoder internal content")
    elif expected_internal_sha is not None:
        raise ParityError("expected decoder is absent")
    for r in (1, 2):
        pol = policy_pair[f"p{r}"]
        if int(pol.get("recipient")) != r:
            raise ParityError("recipient order differs")
        q0, classes, ids = rebuild_policy_bank(pol)
        if decoder is None:
            banks[r] = q0
            continue
        table_record = decoder[f"r{r}"]
        if (int(table_record.get("recipient")) != r or
                int(table_record.get("K")) != q0.shape[1] or
                int(table_record.get("T")) != len(ids) or
                not np.array_equal(_i64(table_record.get("token_class")), classes)):
            raise ParityError("decoder recipient table identity differs")
        family = decoder["family"]
        if table_record.get("kind") != family:
            raise ParityError("decoder family differs")
        alpha = table_record["alpha"] if family == "H-GLOBAL-TEMP" else \
                table_record["alphas"] if family == "H-CLASS-TEMP" else None
        if alpha is None:
            raise ParityError("decoder family is outside parity scope")
        reproduced = temperature_bank(q0, classes, alpha)
        stored = _f64(table_record["q"])
        table_parity = compare_arrays(reproduced, stored)
        if table_parity["status"] == "FAIL":
            raise ParityError("stored decoder table shape differs")
        if table_parity["status"] != "PASS":
            raise NumericalParityError("calibrated decoder table", table_parity)
        banks[r] = reproduced
    return banks


def _ordered_float_bits(a: np.ndarray) -> np.ndarray:
    bits = np.asarray(a, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1 << 63)
    return np.where(bits & sign, ~bits, bits | sign)


def compare_arrays(independent: np.ndarray, produced: np.ndarray) -> dict:
    a = np.ascontiguousarray(independent)
    b = np.ascontiguousarray(produced)
    result = {"independent_sha256": array_digest(a), "s2_sha256": array_digest(b),
              "mismatch_count": 0, "max_abs_difference": None,
              "max_ulp_difference": None, "status": "PASS"}
    if a.shape != b.shape or a.dtype != b.dtype:
        result.update(status="FAIL", mismatch_count=max(a.size, b.size))
        return result
    if a.tobytes() == b.tobytes():
        if np.issubdtype(a.dtype, np.floating):
            result.update(max_abs_difference=0.0, max_ulp_difference=0)
        return result
    different = np.not_equal(a, b)
    # Includes signed zero and NaN payload differences even if == says equal.
    byte_rows = np.frombuffer(a.tobytes(), dtype=np.uint8) != \
                np.frombuffer(b.tobytes(), dtype=np.uint8)
    result["mismatch_count"] = int(np.count_nonzero(different)) or int(np.count_nonzero(byte_rows))
    if np.issubdtype(a.dtype, np.floating):
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            result["max_abs_difference"] = float(np.max(np.abs(a - b)))
            if a.dtype == np.float64:
                x, y = _ordered_float_bits(a), _ordered_float_bits(b)
                result["max_ulp_difference"] = int(np.max(np.maximum(x, y) - np.minimum(x, y)))
        result["status"] = "UNRESOLVED_NUMERICAL_PARITY"
    else:
        result["status"] = "FAIL"
    return result


def compare_release(independent: dict, produced: dict) -> dict:
    cell = compare_arrays(_i64(independent["cell"]), _i64(produced["cell"]))
    token = compare_arrays(_i64(independent["token"]), _i64(produced["token"]))
    bank = compare_arrays(_f64(independent["bank"]), _f64(produced["bank"]))
    a_token, b_token = _i64(independent["token"]), _i64(produced["token"])
    a_bank, b_bank = _f64(independent["bank"]), _f64(produced["bank"])
    if np.any(a_token < 0) or np.any(a_token >= len(a_bank)) or \
            np.any(b_token < 0) or np.any(b_token >= len(b_bank)):
        raise ParityError("released token outside its complete bank")
    probability = compare_arrays(a_bank[a_token], b_bank[b_token])
    decision = compare_arrays(_i64(independent["decision"]),
                              _i64(produced["decision"]))
    fields = {"cell": cell, "token": token, "bank": bank,
              "probability": probability, "decision": decision}
    statuses = {x["status"] for x in fields.values()}
    return {**fields, "status": "FAIL" if "FAIL" in statuses else
            "UNRESOLVED_NUMERICAL_PARITY" if "UNRESOLVED_NUMERICAL_PARITY" in statuses else "PASS"}


def load_npz_exact(path: Path, expected_sha256: str, members: set[str]) -> dict[str, np.ndarray]:
    require_sha(file_sha256(path), expected_sha256, "role-only NPZ file")
    with np.load(path, allow_pickle=False) as archive:
        require_members(set(archive.files), members)
        arrays = {key: archive[key] for key in sorted(members)}
    if members == REFERENCE_MEMBERS:
        float_fields = {"U0_1", "U0_2", "Ucal_1", "Ucal_2"}
    elif members == BANK_MEMBERS:
        float_fields = {"probability"}
    elif members == TOKEN_MEMBERS:
        float_fields = set()
    else:
        raise ParityError("unregistered NPZ member contract")
    for key, array in arrays.items():
        expected = np.dtype("float64" if key in float_fields else "int64")
        if array.dtype != expected or not array.flags.c_contiguous:
            raise ParityError("role-only NPZ dtype or memory order differs")
    return arrays


def check_role_membership(role: str, reference: dict, tokens: dict,
                          receipt: dict) -> dict:
    if role not in ROLES:
        raise ParityError("role outside admitted score-bank scope")
    require_members(set(reference), REFERENCE_MEMBERS)
    require_members(set(tokens), TOKEN_MEMBERS)
    row, group = _i64(reference["row_id"]), _i64(reference["unit"])
    if (len(row) != ROLE_COUNTS[role][0] or len(np.unique(group)) != ROLE_COUNTS[role][1] or
            np.any(np.diff(row) <= 0) or not np.array_equal(row, _i64(tokens["row_id"])) or
            not np.array_equal(group, _i64(tokens["unit"]))):
        raise ParityError("role rows, groups or token alignment differs")
    pair_sha = array_digest(np.column_stack((row, group)))
    require_sha(array_digest(row), ROLE_ROW_SHA256[role], "role row order")
    require_sha(pair_sha, ROLE_PAIR_SHA256[role], "role pair order")
    require_sha(pair_sha, receipt["row_unit_pair_sha256"], "S2 role pair order")
    if int(receipt["rows"]) != len(row):
        raise ParityError("S2 role row count differs")
    return {"rows": len(row), "groups": len(np.unique(group)),
            "row_id_sha256": array_digest(row), "row_unit_pair_sha256": pair_sha}


def replay_role_recipient(role: str, recipient: int, reference: dict,
                          tokens: dict, policy: dict, decoded: np.ndarray,
                          produced_bank: dict, receipt: dict) -> tuple[dict, dict]:
    key = recipient
    p = _f64(reference[f"U0_{key}"])
    decision = _i64(reference[f"d_{key}"])
    cell = assign_fine_cells(p, decision, policy["fine"])
    mapping = _i64(policy["cell_token"])
    token = mapping[cell]
    q0, classes, token_ids = rebuild_policy_bank(policy)
    strict_argmax(decoded, classes)
    if (np.any(classes[token] != decision) or
            np.any(_i64(policy["fine"]["cell_class"])[cell] != decision)):
        raise ParityError("historical output changed a predicted class")
    s2_token = _i64(tokens[f"token_{key}"])
    if np.any(s2_token < 0) or np.any(s2_token >= len(classes)):
        raise ParityError("S2 token outside complete registered bank")
    observed = {"cell": _i64(tokens[f"cell_{key}"]),
                "token": s2_token,
                "bank": _f64(produced_bank["probability"]),
                "decision": classes[s2_token]}
    if (not np.array_equal(_i64(produced_bank["token_class"]), classes) or
            not np.array_equal(_i64(produced_bank["token_id"]), token_ids) or
            not np.array_equal(_i64(produced_bank["input_class"]),
                               _i64(policy["fine"]["cell_class"])) or
            not np.array_equal(_i64(produced_bank["fine_to_hcal_token"]), mapping)):
        raise ParityError("S2 complete bank metadata/token IDs differ")
    independent = {"cell": cell, "token": token, "bank": decoded,
                   "decision": decision}
    comparison = compare_release(independent, observed)
    for field in ("cell", "token"):
        require_sha(array_digest(observed[field]), receipt[f"{field}_{key}_sha256"],
                    f"S2 {role} {field} receipt")
    public = {"role": role, "recipient": key, "rows": len(cell),
              "strict_argmax": True, **comparison}
    private = {"role": role, "recipient": key,
               "first_cell_mismatch": _first_mismatch(cell, observed["cell"]),
               "first_token_mismatch": _first_mismatch(token, observed["token"]),
               "first_probability_mismatch": _first_mismatch(decoded[token],
                                                              observed["bank"][observed["token"]])}
    return public, private


def _first_mismatch(a: np.ndarray, b: np.ndarray) -> int | None:
    if a.shape != b.shape or a.dtype != b.dtype:
        return 0
    av = np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
    bv = np.ascontiguousarray(b).view(np.uint8).reshape(len(b), -1)
    difference = np.flatnonzero(np.any(av != bv, axis=1))
    return int(difference[0]) if len(difference) else None


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "byte_order": sys_byte_order(),
            "array_digest_serialization": "sha256(str(dtype).encode + str(shape).encode + C_contiguous_bytes)"}


def sys_byte_order() -> str:
    import sys
    return sys.byteorder


def _json_file(path: Path, expected_sha256: str | None = None) -> dict:
    if expected_sha256 is not None:
        require_sha(file_sha256(path), expected_sha256, "registered JSON file")
    return json.loads(Path(path).read_text())


def preflight_public_sources(paths: dict, expected_campaign_lock_sha256: str,
                             expected_data_manifest_sha256: str) -> dict:
    """Check public metadata only; call before any role-only NPZ load.

    `paths` has hcal_model, hcal_admission, lra_model, lra_admission,
    ordered_schema, hcal_role_manifest, campaign_lock, data_manifest. S4's
    separately sealed queue unit must still authorize the Adult read.
    """
    pinned = {"hcal_model": HCAL_MANIFEST_SHA256,
              "hcal_admission": HCAL_ADMISSION_SHA256,
              "lra_model": LRA_MANIFEST_SHA256,
              "lra_admission": LRA_ADMISSION_SHA256,
              "ordered_schema": ORDERED_SCHEMA_SHA256,
              "campaign_lock": expected_campaign_lock_sha256,
              "data_manifest": expected_data_manifest_sha256}
    if set(paths) != set(pinned) | {"hcal_role_manifest"}:
        raise ParityError("public source preflight path set differs")
    for name, expected in pinned.items():
        require_sha(file_sha256(Path(paths[name])), expected, f"public {name}")
    lock = _json_file(Path(paths["campaign_lock"]))
    if (lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or
            lock.get("adult_fit_permission") is not True or
            lock.get("families", {}).get("A", {}).get("preprocessing_admission") != "ALLOWED"):
        raise ParityError("effective A preprocessing seal absent")
    data = _json_file(Path(paths["data_manifest"]))
    if data.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise ParityError("S4 role-only manifest admission absent")
    roles = _json_file(Path(paths["hcal_role_manifest"]))
    if roles.get("ok") is not True:
        raise ParityError("hcal role manifest does not declare valid roles")
    for role in ROLES:
        entry = roles["roles" if role == "ATTACK_FIT_NEW" else "inherited"][role]
        if (int(entry["rows"]) != ROLE_COUNTS[role][0] or
                int(entry["groups"]) != ROLE_COUNTS[role][1] or
                entry["row_id_sha256"] != ROLE_ROW_SHA256[role]):
            raise ParityError("hcal role manifest membership differs")
        for seed in (0, 1):
            spec = data["real_file_specs"][f"{role}:references_s{seed}"]
            if (set(spec["members"]) != REFERENCE_MEMBERS or
                    spec["row_unit_pair_sha256"] != ROLE_PAIR_SHA256[role]):
                raise ParityError("S2 role-only reference membership differs")
    hcal_model = _json_file(Path(paths["hcal_model"]))
    hcal_admission = _json_file(Path(paths["hcal_admission"]))
    lra_model = _json_file(Path(paths["lra_model"]))
    _json_file(Path(paths["lra_admission"]))
    u0 = next((x for x in hcal_model["packages"] if x.get("release") == "SRC|U"), None)
    if u0 is None or any(u0["model_pt_sha256"][str(seed)] != TEACHER_SHA256[seed]
                         for seed in (0, 1)):
        raise ParityError("hcal teacher manifest differs")
    for (seed, key), (release, counts, policy_sha, source_bank_sha,
                     internal_sha, decoder_file_sha) in OUTPUT_PINS.items():
        if hcal_admission["units"][f"fine__s{seed}"]["files_sha256"]["fine.json"] != FINE_SHA256[seed]:
            raise ParityError("hcal fine source admission differs")
        base_release = release.split("|H-")[0]
        if key == "lra_seq21":
            pkg = next((x for x in lra_model["packages"] if x.get("config") == release), None)
            if pkg is None or pkg["seeds"][str(seed)]["policy_json_sha256"] != policy_sha or \
                    pkg["seeds"][str(seed)]["decoder_json_sha256"] is not None:
                raise ParityError("lra sequential source admission differs")
        elif key in ("tstar", "pstar"):
            pkg = next((x for x in hcal_model["packages"] if x.get("release") == release), None)
            if pkg is None or pkg["seeds"][str(seed)]["policy_json_sha256"] != policy_sha or \
                    pkg["seeds"][str(seed)]["decoder_sha256"] != internal_sha:
                raise ParityError("hcal calibrated source admission differs")
        else:
            stem = base_release.replace("|", "_").replace("l0.1", "l0.1")
            unit = f"pol__s{seed}__{stem}"
            if hcal_admission["units"][unit]["files_sha256"]["policy.json"] != policy_sha:
                raise ParityError("hcal D0 policy source admission differs")
        frozen_key = f"{seed}|{base_release}"
        # Hcal's historical bank key for l0.1 is its explicit l0.1 spelling.
        if hcal_admission["frozen_bank"][frozen_key]["bank_sha256"] != source_bank_sha:
            raise ParityError("hcal frozen-bank source admission differs")
    return data


def _source_identity(descriptor: dict, seed: int, key: str,
                     data_manifest: dict) -> tuple[dict, dict, dict | None, dict]:
    """Hash public source objects and checkpoint before any role-value read."""
    release, counts, policy_sha, source_bank_sha, internal_sha, decoder_file_sha = \
        OUTPUT_PINS[(seed, key)]
    paths = {name: Path(descriptor[name]) for name in
             ("fine_json", "policy_json", "teacher_checkpoint", "source_bank_file")}
    require_sha(file_sha256(paths["fine_json"]), FINE_SHA256[seed], "fine file")
    require_sha(file_sha256(paths["policy_json"]), policy_sha, "policy file")
    require_sha(file_sha256(paths["teacher_checkpoint"]), TEACHER_SHA256[seed],
                "teacher checkpoint")
    # This hashes the historical frozen-bank NPZ as bytes; it never deserializes
    # that file or any all-row predictions.
    require_sha(file_sha256(paths["source_bank_file"]), source_bank_sha,
                "original frozen bank file")
    fine = _json_file(paths["fine_json"])
    pair = _json_file(paths["policy_json"])
    decoder = None
    if decoder_file_sha is not None:
        if not descriptor.get("decoder_json"):
            raise ParityError("registered decoder file absent")
        decoder = _json_file(Path(descriptor["decoder_json"]), decoder_file_sha)
    elif descriptor.get("decoder_json") is not None:
        raise ParityError("D0 control unexpectedly carries decoder")
    if (pair.get("kind") != "qpc.PolicyPair" or
            pair.get("config", {}).get("teacher_model_sha256") != TEACHER_SHA256[seed] or
            pair.get("config", {}).get("feature_names_sha256") !=
            data_manifest["source"]["feature_names_sha256_newline_join"]):
        raise ParityError("policy teacher/schema binding differs")
    for r in (1, 2):
        check_fine(fine[f"fine{r}"])
        if fine_fingerprint(fine[f"fine{r}"]) != \
                fine_fingerprint(pair[f"p{r}"]["fine"]):
            raise ParityError("policy embedded fine partition differs from pinned fine file")
    expected_fingerprint = pair_fingerprint(pair)
    if pair.get("fingerprint") != expected_fingerprint:
        raise ParityError("registered policy fingerprint differs")
    tables = decoded_bank(pair, decoder, release, internal_sha)
    if (len(tables[1]) != counts[0] or len(tables[2]) != counts[1] or
            pair["p1"]["fine"]["K"] != 2 or pair["p2"]["fine"]["K"] != 6):
        raise ParityError("complete token counts or task class order differ")
    return fine, pair, decoder, tables


def _receipt_and_queue(descriptor: dict, seed: int, key: str,
                       campaign_lock_sha256: str, data_manifest_sha256: str) -> dict:
    receipt_path = Path(descriptor["bank_receipt_json"])
    receipt_sha = file_sha256(receipt_path)
    receipt = _json_file(receipt_path)
    release = OUTPUT_PINS[(seed, key)][0]
    unit_id = f"S2-A-SCORE-BANK-{key.upper().replace('_', '-')}-S{seed}"
    if (receipt.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1" or
            receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED" or
            receipt.get("seed") != seed or receipt.get("release_id") != release or
            receipt.get("s3_preprocessing_unit_id") != unit_id or
            receipt.get("science_lock_sha256") != campaign_lock_sha256 or
            receipt.get("data_access_manifest_sha256") != data_manifest_sha256):
        raise ParityError("S2 bank receipt identity differs")
    if (receipt.get("python_version") != S2_REFERENCE_RUNTIME["python"] or
            receipt.get("numpy_version") != S2_REFERENCE_RUNTIME["numpy"]):
        raise ParityError("S2 bank producer runtime differs from ten-unit receipt inventory")
    queue_receipt_path = Path(descriptor["queue_receipt_json"])
    queue_receipt_sha = file_sha256(queue_receipt_path)
    qreceipt = _json_file(queue_receipt_path)
    if (qreceipt.get("unit_id") != unit_id or qreceipt.get("status") != "done" or
            qreceipt.get("returncode") != 0 or
            qreceipt.get("output_sha256") != {str(receipt_path): receipt_sha}):
        raise ParityError("S3 preprocessing queue receipt differs")
    queue = _json_file(Path(descriptor["queue_state_json"]))
    unit = queue.get("units", {}).get(unit_id, {})
    spec = unit.get("spec", {})
    if (unit.get("status") != "done" or spec.get("unit_id") != unit_id or
            spec.get("seed") != seed or spec.get("release_key") != key or
            spec.get("data_access") != "A_ROLE_ONLY_REFERENCES_PREPROCESS" or
            spec.get("outputs") != [str(receipt_path)]):
        raise ParityError("S3 queued bank unit identity differs")
    if set(receipt.get("private_output_sha256", {})) != \
            {"bank_r1.npz", "bank_r2.npz"} | {f"tokens_{role}.npz" for role in ROLES}:
        raise ParityError("S2 score-bank output set differs")
    binding = receipt.get("source_binding", {})
    binding_bytes = json.dumps(binding, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode("utf-8")
    require_sha(hashlib.sha256(binding_bytes).hexdigest(),
                receipt.get("source_binding_sha256"), "S2 source binding")
    return {"receipt": receipt, "receipt_sha256": receipt_sha,
            "queue_receipt_sha256": queue_receipt_sha,
            "queue_unit_id": unit_id,
            "queue_spec_sha256": hashlib.sha256(json.dumps(
                spec, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()).hexdigest()}


def _check_source_binding(seed: int, key: str, fine: dict, policy: dict,
                          decoder: dict | None, banks: dict, identity: dict,
                          data_manifest: dict) -> None:
    binding = identity["receipt"]["source_binding"]
    release, _, policy_sha, _, internal_sha, decoder_file_sha = OUTPUT_PINS[(seed, key)]
    expected = {
        "schema": "pcrl-portfolio-a-source-binding-v1",
        "release_id": release, "model_seed": seed,
        "teacher_model_sha256": TEACHER_SHA256[seed],
        "feature_names_sha256_newline_join":
            data_manifest["source"]["feature_names_sha256_newline_join"],
        "fine_json_sha256": FINE_SHA256[seed],
        "policy_json_sha256": policy_sha,
        "decoder_json_sha256": decoder_file_sha,
        "decoder_internal_sha256": internal_sha,
        "policy_fingerprint": pair_fingerprint(policy),
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise ParityError(f"S2 source binding {field} differs")
    if set(binding.get("recipients", {})) != {"r1", "r2"}:
        raise ParityError("S2 source binding recipient set differs")
    source_json = identity["receipt"].get("source_json_sha256", {})
    if source_json != {"fine_sha256": FINE_SHA256[seed],
                       "policy_sha256": policy_sha,
                       "decoder_sha256": internal_sha,
                       "decoder_file_sha256": decoder_file_sha,
                       "teacher_model_sha256": TEACHER_SHA256[seed]}:
        raise ParityError("S2 source JSON/checkpoint hash map differs")
    if (identity["receipt"].get("policy_fingerprint") != expected["policy_fingerprint"] or
            identity["receipt"].get("decoder_sha256") != internal_sha):
        raise ParityError("S2 policy/decoder receipt differs")
    for r in (1, 2):
        pol = policy[f"p{r}"]
        _, classes, _ = rebuild_policy_bank(pol)
        expected_recipient = {
            "class_order": list(range(int(pol["fine"]["K"]))),
            "fine_fingerprint": fine_fingerprint(fine[f"fine{r}"]),
            "input_class_sha256": array_digest(_i64(pol["fine"]["cell_class"])),
            "fine_to_hcal_token_sha256": array_digest(_i64(pol["cell_token"])),
            "bank_probability_sha256": array_digest(banks[r]),
            "token_class_sha256": array_digest(classes),
        }
        if binding["recipients"][f"r{r}"] != expected_recipient:
            raise ParityError("S2 recipient source binding differs")


def verify_admitted_output(descriptor: dict, seed: int, key: str,
                           data_manifest: dict, campaign_lock_sha256: str,
                           data_manifest_sha256: str) -> tuple[dict, dict]:
    """Replay one paired historical output after caller's S3/S1 Adult admission.

    Descriptor paths stay private. This function independently hashes each
    registered source and output, and reads only role-specific reference and
    S2 role-specific bank/token NPZ members. It never reads a label.
    """
    if (seed, key) not in OUTPUT_PINS:
        raise ParityError("output outside ten pinned historical controls")
    receipt_id = _receipt_and_queue(descriptor, seed, key,
                                    campaign_lock_sha256, data_manifest_sha256)
    receipt = receipt_id["receipt"]
    fine, policy, decoder, decoded = _source_identity(descriptor, seed, key,
                                                       data_manifest)
    _check_source_binding(seed, key, fine, policy, decoder, decoded,
                          receipt_id, data_manifest)
    root = Path(descriptor["bank_receipt_json"]).parent
    bank = {}
    for r in (1, 2):
        filename = f"bank_r{r}.npz"
        bank[r] = load_npz_exact(root / filename,
                                 receipt["private_output_sha256"][filename], BANK_MEMBERS)
        binding = receipt["source_binding"]["recipients"][f"r{r}"]
        for field, bfield in (("probability", "bank_probability_sha256"),
                              ("token_class", "token_class_sha256"),
                              ("input_class", "input_class_sha256"),
                              ("fine_to_hcal_token", "fine_to_hcal_token_sha256")):
            require_sha(array_digest(bank[r][field]), binding[bfield],
                        f"S2 bank r{r} {field}")
    public_roles, private_roles = {}, {}
    seen_rows = set()
    seen_groups = set()
    for role in ROLES:
        ref_spec = data_manifest["real_file_specs"][f"{role}:references_s{seed}"]
        if set(ref_spec["members"]) != REFERENCE_MEMBERS or \
                ref_spec["role"] != role:
            raise ParityError("admitted role reference specification differs")
        reference = load_npz_exact(Path(descriptor["reference_npz_by_role"][role]),
                                   ref_spec["sha256"], REFERENCE_MEMBERS)
        token_name = f"tokens_{role}.npz"
        tokens = load_npz_exact(root / token_name,
                                receipt["private_output_sha256"][token_name], TOKEN_MEMBERS)
        role_receipt = receipt["role_receipts"][role]
        require_sha(ref_spec["sha256"], role_receipt["source_reference_sha256"],
                    "S2 role reference file")
        membership = check_role_membership(role, reference, tokens, role_receipt)
        row_ids = set(_i64(reference["row_id"]).tolist())
        if seen_rows.intersection(row_ids):
            raise ParityError("admitted roles are not disjoint")
        seen_rows.update(row_ids)
        group_ids = set(_i64(reference["unit"]).tolist())
        if seen_groups.intersection(group_ids):
            raise ParityError("admitted role groups are not disjoint")
        seen_groups.update(group_ids)
        public_roles[role] = {"membership": membership, "recipients": {}}
        private_roles[role] = {}
        for r in (1, 2):
            public, private = replay_role_recipient(role, r, reference, tokens,
                                                     policy[f"p{r}"], decoded[r], bank[r],
                                                     role_receipt)
            public_roles[role]["recipients"][f"r{r}"] = public
            private_roles[role][f"r{r}"] = private
    release, counts, policy_sha, source_bank_sha, internal_sha, decoder_file_sha = \
        OUTPUT_PINS[(seed, key)]
    statuses = {public_roles[role]["recipients"][f"r{r}"]["status"]
                for role in ROLES for r in (1, 2)}
    runner_runtime_matches_producer = (
        platform.python_version() == S2_REFERENCE_RUNTIME["python"] and
        np.__version__ == S2_REFERENCE_RUNTIME["numpy"])
    result = {
        "seed": seed, "release_key": key, "release_id": release,
        "source_binding_sha256": receipt["source_binding_sha256"],
        "bank_unit_id": receipt_id["queue_unit_id"],
        "bank_queue_spec_sha256": receipt_id["queue_spec_sha256"],
        "bank_queue_receipt_sha256": receipt_id["queue_receipt_sha256"],
        "bank_admission_receipt_sha256": receipt_id["receipt_sha256"],
        "bank_output_file_sha256": dict(receipt["private_output_sha256"]),
        "fine_file_sha256": FINE_SHA256[seed],
        "policy_file_sha256": policy_sha,
        "decoder_file_sha256": decoder_file_sha,
        "decoder_internal_sha256": internal_sha,
        "teacher_checkpoint_sha256": TEACHER_SHA256[seed],
        "source_bank_file_sha256": source_bank_sha,
        "s2_producer_runtime": dict(S2_REFERENCE_RUNTIME),
        "runner_runtime_matches_producer": runner_runtime_matches_producer,
        "decoded_bank_sha256": {f"r{r}": array_digest(decoded[r]) for r in (1, 2)},
        "expected_token_counts": list(counts),
        "observed_token_counts": [len(bank[r]["token_id"]) for r in (1, 2)],
        "roles": public_roles,
        "status": "FAIL" if "FAIL" in statuses else
                  "UNRESOLVED_NUMERICAL_PARITY" if "UNRESOLVED_NUMERICAL_PARITY" in statuses
                  or not runner_runtime_matches_producer
                  else "PASS",
    }
    return result, {"seed": seed, "release_key": key, "roles": private_roles}


def assemble_public_report(outputs: list[dict], verifier_source_sha256: str,
                           campaign_lock_sha256: str,
                           data_manifest_sha256: str) -> dict:
    """Aggregate all ten outputs; never infer PASS from a selected subset."""
    identities = [(x.get("seed"), x.get("release_key")) for x in outputs]
    missing = sorted(set(OUTPUT_PINS) - set(identities))
    duplicated = len(identities) != len(set(identities))
    unknown = sorted(set(identities) - set(OUTPUT_PINS))
    if duplicated or unknown:
        raise ParityError("parity report has duplicated or unpinned outputs")
    status = ("INCOMPLETE" if missing else "PASS" if all(x.get("status") == "PASS"
               for x in outputs) else "FAIL")
    return {"schema": SCHEMA, "overall_status": status,
            "missing_units": [f"s{seed}:{key}" for seed, key in missing],
            "verifier_source_sha256": verifier_source_sha256,
            "campaign_lock_sha256": campaign_lock_sha256,
            "data_access_manifest_sha256": data_manifest_sha256,
            "source_manifest_sha256": {
                "hcal_model": HCAL_MANIFEST_SHA256,
                "hcal_admission": HCAL_ADMISSION_SHA256,
                "lra_model": LRA_MANIFEST_SHA256,
                "lra_admission": LRA_ADMISSION_SHA256,
                "ordered_schema": ORDERED_SCHEMA_SHA256,
            },
            "runtime": runtime_identity(),
            "role_order": list(ROLES),
            "role_expected_counts": {role: {"rows": ROLE_COUNTS[role][0],
                                            "groups": ROLE_COUNTS[role][1],
                                            "row_id_sha256": ROLE_ROW_SHA256[role],
                                            "row_unit_pair_sha256": ROLE_PAIR_SHA256[role]}
                                     for role in ROLES},
            "outputs": sorted(outputs, key=lambda x: (x["seed"], x["release_key"]))}


def verify_admitted_bundle(public_source_paths: dict,
                           descriptors: dict[tuple[int, str], dict],
                           expected_campaign_lock_sha256: str,
                           expected_data_manifest_sha256: str) -> tuple[dict, dict]:
    """S4 integration entry point, callable only inside its separately admitted unit.

    Public source checks happen before the first role-only load. `descriptors`
    must contain no unpinned output. Missing outputs are reported INCOMPLETE.
    Per-output parity failures are recorded without public paths or row indices.
    """
    if set(descriptors) - set(OUTPUT_PINS):
        raise ParityError("bundle contains unpinned historical output")
    data_manifest = preflight_public_sources(public_source_paths,
                                             expected_campaign_lock_sha256,
                                             expected_data_manifest_sha256)
    public_outputs = []
    private_outputs = []
    for seed, key in sorted(descriptors):
        try:
            public, private = verify_admitted_output(
                descriptors[(seed, key)], seed, key, data_manifest,
                expected_campaign_lock_sha256, expected_data_manifest_sha256)
        except NumericalParityError as exc:
            public = {"seed": seed, "release_key": key,
                      "release_id": OUTPUT_PINS[(seed, key)][0],
                      "status": "UNRESOLVED_NUMERICAL_PARITY",
                      "numerical_stage": exc.stage,
                      "max_abs_difference": exc.max_abs_difference,
                      "max_ulp_difference": exc.max_ulp_difference,
                      "runtime": runtime_identity()}
            private = {"seed": seed, "release_key": key,
                       "numerical_stage": exc.stage, "failure": str(exc)}
        except (ParityError, OSError, KeyError, IndexError, TypeError, ValueError) as exc:
            # A source or custody failure cannot be counted as numerical parity.
            public = {"seed": seed, "release_key": key,
                      "release_id": OUTPUT_PINS[(seed, key)][0],
                      "status": "FAIL", "failure_type": type(exc).__name__}
            private = {"seed": seed, "release_key": key,
                       "failure_type": type(exc).__name__, "failure": str(exc)}
        public_outputs.append(public)
        private_outputs.append(private)
    report = assemble_public_report(public_outputs, file_sha256(Path(__file__)),
                                    expected_campaign_lock_sha256,
                                    expected_data_manifest_sha256)
    report["source_manifest_sha256"]["hcal_role_manifest"] = file_sha256(
        Path(public_source_paths["hcal_role_manifest"]))
    return report, {"schema": "pcrl-a-s4-score-bank-parity-private-v1",
                    "outputs": private_outputs}


def standard_paths(campaign_root: Path, cache_root: Path,
                   queue_root: Path, data_manifest: dict) -> tuple[dict, dict]:
    """Construct only the ten registered path sets; no file is opened here."""
    campaign_root = Path(campaign_root).resolve()
    cache_root = Path(cache_root).resolve()
    queue_root = Path(queue_root).resolve()
    results = campaign_root.parent
    public = {
        "hcal_model": results / "pcrl_heldout_calibration_v1" / "MODEL_MANIFEST.json",
        "hcal_admission": results / "pcrl_heldout_calibration_v1" / "SOURCE_ADMISSION.json",
        "hcal_role_manifest": results / "pcrl_heldout_calibration_v1" / "ROLE_MANIFEST.json",
        "lra_model": results / "pcrl_adult_learned_decoder_release_v1" / "MODEL_MANIFEST.json",
        "lra_admission": results / "pcrl_adult_learned_decoder_release_v1" / "SOURCE_ADMISSION.json",
        "ordered_schema": cache_root / "hcal_v1" / "admitted" / "inputs" / "schema.json",
        "campaign_lock": campaign_root / "CAMPAIGN_LOCK.json",
        "data_manifest": campaign_root / "DATA_ACCESS_MANIFEST.json",
    }
    stem = {"tstar": "U_FINE-TASK_i8o64",
            "pstar": "U_JOINT_i8o64_l0.1",
            "hcal_local_d0": "U_LOCAL_i8o64_l0.1",
            "hcal_seq12_d0": "U_SEQ-12_i8o64_l0.1",
            "lra_seq21": "U_SEQ-21_i8o64_l0.1"}
    decoder_stem = {"tstar": "U_FINE-TASK_i8o64_H-GLOBAL-TEMP",
                    "pstar": "U_JOINT_i8o64_l0.1_H-CLASS-TEMP"}
    descriptors = {}
    for seed, key in OUTPUT_PINS:
        pair_stem = stem[key]
        unit_id = f"S2-A-SCORE-BANK-{key.upper().replace('_', '-')}-S{seed}"
        bank_root = (campaign_root / "shared" / ".private" /
                     f"{key}_score_bank_s{seed}_v1")
        policy_root = (cache_root / ("lra_v1/run/units" if key == "lra_seq21"
                                     else "hcal_v1/admitted/units") /
                       f"pol__s{seed}__{pair_stem}")
        descriptors[(seed, key)] = {
            "fine_json": cache_root / "hcal_v1/admitted/units" /
                         f"fine__s{seed}" / "fine.json",
            "policy_json": policy_root / "policy.json",
            "decoder_json": (cache_root / "hcal_v1/package" /
                             f"decoder__s{seed}__{decoder_stem[key]}.json")
                            if key in decoder_stem else None,
            "teacher_checkpoint": cache_root / "hcal_v1/admitted" /
                                  f"rel__s{seed}__U" / "model.pt",
            "source_bank_file": cache_root / "hcal_v1/admitted/bank" /
                                f"bank__s{seed}__{pair_stem}.npz",
            "bank_receipt_json": bank_root / "BANK_ADMISSION_RECEIPT.json",
            "queue_receipt_json": queue_root / "units" / unit_id / "RECEIPT.json",
            "queue_state_json": queue_root / "QUEUE.json",
            "reference_npz_by_role": {
                role: results / "pcrl_method_portfolio_v1" /
                      data_manifest["real_file_specs"][f"{role}:references_s{seed}"]["path"]
                for role in ROLES},
        }
    return public, descriptors


def write_once_reports(public_path: Path, private_path: Path,
                       public_report: dict, private_report: dict) -> None:
    """Install each write-once report file atomically, private receipt first.

    S4's queued wrapper chooses and authorizes the destinations. This helper
    refuses overwrite; it is not an Adult admission gate.
    """
    import os
    import tempfile

    if public_report.get("schema") != SCHEMA or \
            private_report.get("schema") != "pcrl-a-s4-score-bank-parity-private-v1":
        raise ParityError("parity report schemas differ")
    public_path, private_path = Path(public_path), Path(private_path)
    if public_path.exists() or private_path.exists():
        raise FileExistsError("score-bank parity output is write-once")
    for path in (public_path, private_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for target, body, mode in ((private_path, private_report, 0o600),
                                   (public_path, public_report, 0o644)):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=target.parent, prefix=".parity-",
                                             delete=False) as handle:
                json.dump(body, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.chmod(handle.name, mode)
                pending.append((Path(handle.name), target))
        for temporary, target in pending:
            if target.exists():
                raise FileExistsError("score-bank parity output appeared during write")
            os.link(temporary, target)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
