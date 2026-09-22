"""The universal R²-to-accuracy guarantee is false; its API must refuse to run."""

import ast
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[name]


def test_zero_affine_r2_with_ninety_percent_threshold_accuracy():
    rows = [(1, 1)] * 9 + [(1, -9)] + [(0, -1)] * 9 + [(0, 9)]
    n = Fraction(len(rows))
    mh = sum(Fraction(h) for _, h in rows) / n
    ma = sum(Fraction(a) for a, _ in rows) / n
    cov = sum((Fraction(h) - mh) * (Fraction(a) - ma) for a, h in rows) / n
    assert cov == 0  # so the affine least-squares R² is exactly zero
    accuracy = Fraction(sum((h > 0) == (a == 1) for a, h in rows), len(rows))
    assert accuracy == Fraction(9, 10)


def test_certified_accuracy_bound_raises():
    bound = _function("pcrl/purposes/verification.py", "certified_accuracy_bound")
    with pytest.raises(NotImplementedError):
        bound(0.0, 0.5, 2)


def test_nonlinear_certificate_check_raises():
    src = (ROOT / "pcrl/purposes/verification.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NonlinearComplianceCertificate")
    check = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check")
    raises = [n for n in ast.walk(check) if isinstance(n, ast.Raise)]
    first = check.body[1] if isinstance(check.body[0], ast.Expr) else check.body[0]
    assert isinstance(first, ast.Raise) and raises


def test_generate_report_no_longer_calls_retired_certificates():
    src = (ROOT / "pcrl/evaluation/certificates.py").read_text()
    assert "nonlinear_cert.check(" not in src
    assert "certified_accuracy_bound(" not in src
