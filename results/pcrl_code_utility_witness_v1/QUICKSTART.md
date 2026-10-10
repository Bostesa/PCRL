# Quickstart

From the repository root, enter `.worktrees/pcrl-code-utility-witness-v1`. The branch is `research/pcrl-code-utility-witness-v1`. Public source and aggregate results are in this directory; private models and per-person releases are under ignored `.private` directories and require the local restore package.

Commands executed in this campaign:

```sh
../../.venv/bin/python -m pytest -q code_witness/test_model.py code_witness/test_bank.py code_witness/test_service.py code_witness/test_worker.py code_witness/test_deploy.py code_witness/test_bank_deploy.py
../../.venv/bin/python -m code_witness.bank_deploy --help
../../.venv/bin/python -m code_witness.deploy --help
python3 -m results.pcrl_code_utility_witness_v1.r0.compute_queue status
```

The six code witness tests in the first command passed as 19 test cases in R1's recorded run. The two CLI `--help` commands were also run successfully. The actual raw-input deployment commands, argument meanings, pinned hashes and recipient-only output fields are in [R1's deployment quickstart](r1/DEPLOYMENT_QUICKSTART.md) and [deployment manifest](r1/DEPLOYMENT_MANIFEST.json). R5 executed the raw commands on a private 32-person sample and recorded results in its public deployment receipts. The bank variants are inherited useful score services; the new `T0_B10` and `T0_B16` services are runnable research artifacts but failed the sealed utility prerequisite.

Scientific training and verification workers used owner-only config files and the global queue. The immutable command contract is in [R1's fitting spec](r1/PREFIT_SPEC.md), [R2's gate inputs](r2/T0_GATE_QUEUE_INPUTS.md), [R4's interval worker contract](r4/UTILITY_INTERVAL_WORKER_CONTRACT.md), and the [protocol](PROTOCOL.md). Do not infer permission to open embargoed roles from the presence of source code.
