"""Extract already published 2018 task decoder aggregates without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import NAMES, digest, write_json


def extract(source: Path, output: Path) -> dict:
    grid = json.loads(source.read_text())
    if grid["pool"] != "test" or not grid["selection_frozen"]:
        raise AssertionError("expected frozen historical development grid")
    result = {"source": str(source), "source_sha256": digest(source),
              "scope": "historical supervised 2018 development, dependent anchors, post hoc",
              "anchors": {}}
    for anchor in "012":
        rows = {}
        for short, name in NAMES.items():
            record = grid["records"][name][anchor]["test"]["utility:A/same_residence"]
            rows[short] = {
                "fixed_decoder": {w: record["fixed_decoder_ce"][w] for w in ("unweighted", "weighted")},
                "independent_probe": {w: record["independent_ce"][w] for w in ("unweighted", "weighted")},
                "selected_deployment": {w: record["ce"][w] for w in ("unweighted", "weighted")},
                "selection": record["selection"], "independent_selection": record["independent_selection"]}
        result["anchors"][anchor] = rows
    result["means_across_dependent_anchors"] = {
        short: {kind: {w: sum(result["anchors"][a][short][kind][w] for a in "012")/3
                       for w in ("unweighted", "weighted")}
                for kind in ("fixed_decoder", "independent_probe", "selected_deployment")}
        for short in NAMES}
    result["Q_minus_D17"] = {
        kind: {w: result["means_across_dependent_anchors"]["Q"][kind][w]
                  -result["means_across_dependent_anchors"]["D17"][kind][w]
               for w in ("unweighted", "weighted")}
        for kind in ("fixed_decoder", "independent_probe", "selected_deployment")}
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    extract(a.source, a.output)
