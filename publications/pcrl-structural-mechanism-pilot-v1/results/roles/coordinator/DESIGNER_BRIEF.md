# Designer brief (identical for designer 1 and designer 2)

You are one of two independent mechanism designers in the PCRL structural-mechanism campaign. The other designer receives this same brief. Do not open the other designer's folder (`roles/designer_1/` or `roles/designer_2/`, whichever is not yours) and do not coordinate. Write only inside your own folder.

## The problem

The service is the ACS two-recipient benchmark. Read, in this order:
1. `results/pcrl_structural_mechanism_pilot_v1/SERVICE_AND_THREAT_MODEL.md` (draft): the service, the declared adversary, the motivation.
2. `results/pcrl_mechanism_design_v1/PROBLEM_SPECIFICATION.md`: variables, release contract, utility clauses, access by role, claims kept separate, mathematical facts. Binding.
3. `results/pcrl_mechanism_design_v1/EVIDENCE_PACK.md`: published recovery and utility of every fitted ACS service.
4. `results/pcrl_mechanism_design_v1/CLOSED_LANES.md`: what was tried and what bound. Closed: fixed tree or hybrid compositions, small adapters, critic schedules or refresh, frozen-code rescue (decoders, calibrators or mixtures on frozen outputs), and penalty-strength sweeps.
5. `results/pcrl_mechanism_design_v1/CORRECTIONS.md` sections 1, 4 and 7: corrected mathematical facts. Deterministic many-to-one probability releases can remove SEX information inside the contract; population injectivity of the fitted models is unknown; MMP (SEX-mixture marginalisation) is declined and RCCS is not validated, so do not propose either or a renamed version.
6. `results/pcrl_mechanism_design_v1/CLOSEST_PRIOR_WORK.md`.
7. Code you may read for what exists: `acs_replication/` (`schema.py`, `trees.py`, `leafcorr.py`, `neural.py`, `neural_raw.py`, `fare_worker.py`, `fagtb.py`, `selection.py`, `readers.py`), `acs_hybrids/readers.py`, `mechanism_design/pilot/` (a prepared but unrun pilot whose pipeline shape a new pilot can reuse).

**Do not open any real-person array** (anything under `.private/` or any `.npz`), and do not run fits. Synthetic toy computations of your own are allowed but not required.

**Corrected claims you must respect.** No competitive new mechanism has been established. The neural adversarial model (NEURAL_RAW rho1 E2, coalition AUC 0.640 / 0.638) is a mandatory strong comparator. The roughly 0.10 score-versus-decision recovery gap is one post-hoc Adult setting at equal accuracy; it is not ACS headroom. The 24 utility caps, the references U0 and Ucal, the roles and the endpoint rules are fixed.

## The hypothesis to investigate

Learn which input distinctions to retain or merge while fitting useful task probabilities, instead of only modifying continuous scores on frozen representations, frozen tree partitions or historical code banks. One possible family is an explicitly many-to-one input partition or representation with probability heads trained to preserve task quality. This is a hypothesis, not a known route to success. You may propose a mechanism outside this family if you argue it is a substantive structural change.

A new privacy weight, wider adapter, refreshed critic, calibration sweep, frozen-output mixture or renamed earlier recipe does not count. A joint (two-recipient) component is not required; a competitive local method is a useful outcome. Include joint optimisation only with a concrete computational or statistical justification (remember `PROBLEM_SPECIFICATION.md` section 9.1).

## What to deliver

One file, `PROPOSAL.md`, in your folder, at most about 4 pages, containing:

1. **Name and one-paragraph summary.**
2. **The exact mapping and optimisation problem**: what is fitted on which role, with which labels and SEX access (SEX may be used only on MODEL_FIT, and only as a training signal), the objective and constraints, the algorithm, and how the deployed map takes the six raw fields to each recipient's probability vector and decision.
3. **Which input distinctions it merges**, concretely for these six fields and two tasks.
4. **Why those distinctions should matter less for the permitted tasks than for SEX inference.**
5. **Why the neural comparator may retain them.**
6. **How utility is maintained**, using separate selection records (MODEL_SELECT under the frozen rule: all 24 point clauses first, then the frozen OUT_LR coalition proxy) and, where needed, CALIBRATION.
7. **What prevents apparent success from being only critic-specific masking.**
8. **The actual transmitted output**, including any identifiers, cell labels or state (the primary contract allows none beyond each recipient's probability vector and argmax).
9. **Its strongest falsifying result**, stated before any data are seen.
10. **The difference from the closed lanes** and from FARE (restricted tree encoders with a fairness criterion), FAGTB, the privacy funnel and adversarial representations, as far as you know them.
11. **The task-only twin**: the same mechanism with the privacy component disabled, matched in capacity, data exposure and training budget.
12. **Implementation sketch and compute**: environment is Python 3.13 with numpy, scipy, scikit-learn 1.8, torch 2.10 (CPU) and CatBoost 1.2.8, single-threaded. Fitting the candidate and its twin for two seeds must fit within about 2 CPU-hours in total. At most **three predefined candidate settings per seed**, fixed now, plus a checkpoint rule if training is iterative.
13. **A minimal synthetic or finite-law check** that would show the mechanism does what it claims: it merges distinct inputs when it says so, repeated inputs in a cell give identical outputs, a task-feasible nontrivial reference exists, protection does not come from label leakage, an omitted wire or a useless task, and a case where sensitive and necessary task information conflict.

Be honest about weaknesses. Say where your rationale could be wrong. Do not claim novelty you cannot support; say which parts are established technique.

When done, compute `shasum -a 256 PROPOSAL.md` and put the hash on the first line of your final message. Then stop. Budget: about 45 minutes of work.
