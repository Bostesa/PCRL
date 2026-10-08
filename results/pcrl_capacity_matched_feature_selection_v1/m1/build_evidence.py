"""Write M1's public, aggregate-only prefit capacity and admission evidence.

Invocation uses only the already admitted portfolio role exports. The output
contains no person IDs, labels, features or private filesystem paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from admission import admit_existing_exports, sha_file
from capacity import capacity_geometry


PRIOR_C_PROTOTYPE_SHA256 = "92b43aecf2e68f4d005e01df0e64d7ff149808850846d1e87abc2ed252f0e78d"
PRIOR_C_FINDINGS_SHA256 = "8f5453995a2a72eb2f0ea26b2c6cbe02c23b4c32946504f03e971336721aacd9"


def _public_feature_names(portfolio_package: Path) -> tuple[str, ...]:
    source = portfolio_package / "teams/c/prototype.py"
    if sha_file(source) != PRIOR_C_PROTOTYPE_SHA256:
        raise ValueError("prior C public grouped-schema source changed")
    spec = importlib.util.spec_from_file_location("focused_prior_c_prototype", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return tuple(module.FEATURE_NAMES)


def make_evidence(portfolio_package: Path) -> tuple[dict, dict]:
    access = admit_existing_exports(portfolio_package)
    findings = portfolio_package / "C_R1_AGGREGATE_FINDINGS.md"
    if sha_file(findings) != PRIOR_C_FINDINGS_SHA256:
        raise ValueError("prior C aggregate task-utility evidence changed")
    geometry = capacity_geometry(_public_feature_names(portfolio_package))
    geometry.update({
        "status": "SYMBOLIC_PREFIT_NO_NEW_SCIENTIFIC_FIT",
        "capacity_rule_id": "CFS-PRIMARY-4NUM-4CAT-MAX77-V1",
        "default_exact_signature_audit": {
            "signature": "numeric group count, categorical group count, exact encoded-column count",
            "at_eight_groups": "each one-categorical deletion has a distinct width; each exact signature has one mask",
            "reason_for_cap": "exact widths are singleton constraints, so use a prefit maximum and report actual widths",
        },
        "profile_choice_evidence": {
            "source": "portfolio Team C task-only C-SIMPLE-TASK-SHARED1",
            "fit_selection_role": "OSF_DEFENSE_FIT",
            "fit_selection_sex_used": False,
            "utility_scoring_role": "INNER_SELECTION (reused development rows)",
            "observed_selected_deletion": "native-country",
            "observed_encoded_columns": 42,
            "task_utility": "point-feasible against both U0 and Ucal for income and occupation; simultaneous intervals unresolved",
            "source_findings_sha256": PRIOR_C_FINDINGS_SHA256,
            "decision_basis": "task-only utility evidence and symbolic alternatives; historical SEX AUC did not rank the profile",
        },
        "union_geometry_limit": (
            "Distinct one-categorical deletions jointly disclose all 83 encoded columns. "
            "A shared deletion has a unique reduced union width, so an exact union-width "
            "control can legitimately choose the identical mask pair and tie."
        ),
        "capacity_interpretation": "same maximum encoded budget; actual transmitted widths differ and are not equal capacity",
        "no_fallback_profile": True,
        "mask_alias_rule": "same selected groups alone is not a fitted-release alias; heads/calibrators/views/budget must also match",
        "union_identity_rule": "same source-order raw union may be information-equivalent even when complete serialized releases differ",
    })
    access.update({
        "prior_package_id": "pcrl_method_portfolio_v1",
        "old_assessment_access": "FORBIDDEN_FOR_SCIENTIFIC_WORKERS",
        "head_validation_label_access": "FORBIDDEN",
        "scientific_role_label_allowlist": {
            "OSF_DEFENSE_FIT": ["income", "occupation", "sex"],
            "CALIBRATION_HELDOUT": ["income", "occupation"],
            "ATTACK_FIT_NEW": ["sex"],
            "INNER_SELECTION": ["income", "occupation", "sex"],
        },
        "source_export_custody_note": (
            "Prior isolated exporter mechanically deserialized full mixed-source X and selected "
            "label members twice before permitted-row slicing. No excluded values were inspected, "
            "used scientifically, logged or exported. No mixed-source reopen occurred in M1."
        ),
        "next_gate": "M0 effective method lock, M5 prefit review and shared compute queue before any new fit",
    })
    return access, geometry


def _write_once(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError("existing public evidence differs; write a versioned amendment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio-package", type=Path, required=True)
    args = parser.parse_args()
    access, geometry = make_evidence(args.portfolio_package)
    here = Path(__file__).resolve().parent
    _write_once(here / "DATA_ACCESS_MANIFEST.json", access)
    _write_once(here / "CAPACITY_SPEC.json", geometry)
    print("M1_ADMISSION_AND_CAPACITY_PASS")


if __name__ == "__main__":
    main()
