#!/bin/sh
# Re-run every verification fixture (CPU, deterministic). ~3 min total.
set -e
cd "$(dirname "$0")"
SY=/opt/homebrew/bin/python3
VENV=/Users/nathansamson/PCRL/.venv/bin/python
SCR=/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad
for f in F01_onehot_identity F02_multiclass_contrast F03_unsupported_classes F04_nested_slate \
         F05_defense_aware_access F06_constant_extension F07_slate_controls F09_accuracy_guarantee F10_coarse_conditioning; do
  $SY $f.py > outputs/${f%%_*}.json
done
$SY F08_recount_headlines.py $SCR/dg $SCR/verify/pcrl_results /Users/nathansamson/PCRL/results/rebuttal/erase_layer_pilot_aws > outputs/F08.json
# ORIGINAL implementations from read-only exports:
#   git -C /Users/nathansamson/PCRL archive <ref> pcrl tests | tar -x -C $SCR/verify/<label>
for r in origin_main research_pcrl-submission-finish-v1; do
  $VENV orig_compare.py --root $SCR/verify/$r --label $r > outputs/orig_$r.json
done
$SY build_checks.py
