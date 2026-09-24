# Inner audit positive controls

`experiments.pcrl_adaptive_release_v1.positive_control` is a diagnostic module. Its label-derived token law is **not deployable**, is not a candidate/control release, and must not enter a competitive result table. It accepts only `audit_fit`, `inner_selection`, and `inner_check` rows whose households have the registered global role assignment; it has no outer-assessment loader. Fitted objects and the aggregate receipt require a private output path. A partial directory is retained for review rather than silently overwritten.

The exact A fixture sets `S = H_A[0] XOR Z`; the AB fixture sets `S = H_B[0] XOR Z` while A's H is constant. In each balanced four-row law, S is independent of the public bit alone and of Z alone, so H-only and token-only binary log loss is `log 2`. A predictor using the legal full view and assigning 0.99 to the correct class has exact expected-token loss `-log(0.99)`; the fixture tests both values through the existing exact scorer. The AB fixture also shows that A cannot infer the interaction when only B has the needed service bit. This is a hand-specified predictor counterexample. Whether the **fitted** common slate learns such an interaction remains an empirical power check, not a conclusion from the exact fixture.

The empirical positive control maps the actual protected class label to its existing token number, using only tokens 0–1 for SEX and 0–8 for RAC1P. This intentionally violates deployment input restrictions to create an unmistakably leaky, noncompetitive diagnostic law. `run_inner_diagnostic(inner_roles, role="attack:AB/SEX", anchor=0, output_dir=private_path, slate="standard")` fits the same declared audit slate and legal H, A, and B ancestors on `audit_fit`, selects routes on `inner_selection`, then scores only `inner_check`. It records U and PWGTP H-minus-diagnostic attack-loss improvements and marks a descriptive detection check at +0.01 nats in both weightings. The threshold is an audit-power warning flag, not a privacy success margin or inferential test. The output records source and inner-row hashes, selected route IDs, scores, status, and artifact inventory; it does not store person-level token laws or labels in the aggregate JSON. Predictor artifacts remain private.

Example coordinator-owned call after hash-verified loading, without opening outer rows:

```python
from experiments.pcrl_adaptive_release_v1 import roles, positive_control
inner = {name: roles.pooled_role(prepared, name)
         for name in ("audit_fit", "inner_selection", "inner_check")}
positive_control.run_inner_diagnostic(
    inner, role="attack:AB/SEX", anchor=0,
    output_dir="/opt/pcrl/work/private/positive_control/anchor0_ab_sex")
```

The module does not launch ACS work by itself. A direct label token tests whether the fitting/selection path notices obvious disclosure; passing it does not prove that the slate detects weaker, conditional, or nonlinear leakage. Failing it limits privacy conclusions and triggers the protocol's predeclared common audit additions, not an endpoint change. All actual candidate audits still need equal slates and fresh model selection.
