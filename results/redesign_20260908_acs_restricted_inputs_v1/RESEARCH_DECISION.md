# Restricted inputs, source controls, and exact coordination

Restricting the student to the erased teacher lowered mean attribute recovery under independent attacks, but cost source performance and did not reliably lower recovery once its own trained observers were included.

The complete study contains 24 matched E/S × full/restricted × C/D final models, six source-only probability banks, and the separate exact coordination calculation. No selective-preservation model or teacher was refitted. These are DEVELOPMENT EVALUATION results on the previously used cohort; three-seed variability is descriptive. All losses and gains below are nats. Lower is better. PWGTP scores use the same predictions selected by unweighted validation.

| Release | Residence, unweighted / PWGTP | SEX gain, independent | Race gain, independent / pooled | All three source margins, unweighted seeds passing |
| --- | --- | --- | --- | --- |
| Direct E teacher | .508385 / .486163 | .008736 | .048108 / unavailable | 0/3 |
| E full C | .503849 / .481240 | .016629 | .056080 / .074529 | 2/3 |
| E restricted C | .505432 / .482860 | .010100 | .051189 / .077666 | 0/3 |
| E restricted D | .505551 / .483033 | .010377 | .051889 / .075270 | 0/3 |
| E source bank | .521901 / .497591 | .006357 | .025224 / unavailable | 0/3 |
| Direct S teacher | .513080 / .488508 | .016212 | .055063 / unavailable | 0/3 |
| S full C | .510308 / .486068 | .022422 | .062959 / .073049 | 3/3 |
| S restricted C | .510575 / .485461 | .020420 | .056403 / .064855 | 0/3 |
| S restricted D | .510701 / .485466 | .020019 | .055881 / .064744 | 0/3 |
| S source bank | .526645 / .500173 | .009918 | .019934 / unavailable | 0/3 |

Attack gains in this compact table are unweighted, at 360 epochs. Pooled selection includes the learned release's own observer catch-up and its inherited representation-fitting exposure; it is not a matched-budget comparison with static teachers or banks. [TABLE](TABLE.md) includes both weights, all five tasks, both attributes, every final arm, native source heads, and mean ± sample SD. [ANALYSIS](ANALYSIS.md) gives the fixed paired comparisons and counterexamples. Figures show [SEX tradeoffs](tradeoff_SEX.png), [race tradeoffs](tradeoff_RAC1P.png), [source/residence stages](utility_stages.png), [native source controls](native_source_controls.png), [retained/removed structure](retained_removed_structure.png), and [exact repeated access](PURPOSE_COORDINATION_EXACT.svg).

The strongest positive result is a useful capability beyond three source probabilities: E restricted C improves residence loss over its source-only bank by .016469 unweighted and .014731 PWGTP. Every seed improves under both weights. This does not extend to commute, and the representation's attribute gains are substantially larger. Teacher-only training also improves source prediction over the original small probes, but equally exposed source-only banks perform better on all three mean native source losses and all three mean downstream source losses. Thus representation preservation is not required for those source gains. This control does not isolate label count from objective, architecture, and optimization.

The strongest counterevidence is E restricted C: its pooled race gain is .077666 versus .074529 for matched full-input C, and .069562 versus .067691 under PWGTP. Every restricted condition fails the original joint source-utility margins in every seed under both weights. E improves mean residence over S, but restricted E has worse pooled race recovery than restricted S in every unweighted seed. D yields a modest E race benefit with a SEX cost; it does not establish a joint improvement. All nine race categories remain reported, and absent fitting/validation support for RAC1P code 4 leaves the full race assessment unassessable.

Every saved restricted/bank composed witness reproduces the corresponding release-then-attacker predictions exactly. Public fixed postprocessing makes the same attack available on the teacher with the extra learned function and training exposure. Improved prediction over a limited direct-teacher probe is not creation of information. Likewise, greater affine accessibility of the erased residual after teacher-only learning does not imply a hidden raw-input route. [Input checks](INPUT_BOUNDARY.md), [composed attacks](COMPOSED_ATTACKS.md), and [mechanism diagnostics](MECHANISM.md) document these distinctions.

The exact calculation establishes a separate, conditional coordination benefit and a repeated-access failure:

| Noise / access | Coalition S accuracy, one pair | Two pairs | Four pairs | Each purpose's accuracy at four pairs |
| --- | --- | --- | --- | --- |
| Independent, fresh | 5/8 | 5/8 | 377/512 | 27/32 |
| Coordinated, fresh | 1/2 | 5/8 | 107/128 | 27/32 |
| Independent, identical cached pair | 5/8 | 5/8 | 5/8 | 3/4 |
| Coordinated, identical cached pair | 1/2 | 1/2 | 1/2 | 3/4 |

Each single noisy purpose has accuracy 3/4. Exact U/V predictions reveal their product perfectly. At four fresh issuances, this coordinated law leaks more about the product than independent noise despite identical marginal purpose channels. Full rational joint distributions verify independence where claimed; chance accuracy alone is not the proof. The union-bound lower bound, max(1/2, 1−q1−q2), is attained by disjoint errors on the declared grid. This is an illustrative prediction-release calculation, not a representation advantage or a novel theorem. [Exact note and verifier](PURPOSE_COORDINATION_EXACT.md) explain the assumptions and limits.

Recommend exactly one next method experiment: the predeclared two-purpose, separate-versus-coalition-trained comparison in [PURPOSE_COORDINATION_DESIGN](PURPOSE_COORDINATION_DESIGN.md), with mandatory coordinated prediction-only and direct-teacher controls, source utility requirements, reserved residence/commute tasks, and an explicit fixed-version repeated-access policy. A representation must earn its added interface through reserved-task usefulness beyond those controls while improving coalition recovery at acceptable utility. Current ACS results show residence headroom for that question, but provide no comparable commute advantage and no qualifying restricted source model. The next study is proposed, not launched.

All authorized units and 360-epoch audits completed; no stronger restart or follow-up study was launched. Full scientific process time was 9.49 minutes, including the exact calculation and the separately frozen bank affine diagnostic. [Runtime](runtime.json), [executed matrix](EXECUTED_MATRIX.json), [validation](VALIDATION.md), [audit budget](AUDIT_BUDGET.md), and [reproduction](REPRODUCTION.md) record completion, measured total work, immutable identities, and local-only artifact hashes.
