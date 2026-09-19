# REPRODUCE_PAPER — v4
```
PY=/Users/nathansamson/PCRL/.venv/bin/python
EV=/Users/nathansamson/PCRL-terminal-1-adversarial/results/pcrl_direct_adversarial_v1   # at 106de9afa (evidence 69e790af)
$PY results/pcrl_manuscript_review_v4/checks/verify_study4.py $EV > results/pcrl_manuscript_review_v4/STUDY4_VERIFICATION.json
$PY results/pcrl_manuscript_review_v4/checks/math_fixtures.py > results/pcrl_manuscript_review_v4/MATH_FIXTURES.json
$PY results/pcrl_manuscript_review_v4/checks/selection_checks.py $EV   # SELECTION_RESCORING_CHECK.json, DISPLACEMENT_CHECK.json
$PY results/pcrl_manuscript_review_v4/checks/build_assets.py           # tables, figures, SELECTION_AUDIT.json, ASSET_HASHES.json
cd papers/pcrl_manuscript_v4 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Expected: all STUDY4_VERIFICATION summary statuses PASS except `residence_above_point001 = REFUTED` and
`erasure_rank = PASS_WITH_CORRECTION`; gamma=0 rescoring changes 50/114 selections; 63/63 step-0 trajectories moved.
No step fits a model or reads 2016.
