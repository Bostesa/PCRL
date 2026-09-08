# Application-screen interpretation

Decision: no real-data representation-method pilot. HAR has a contradictory absolute activity policy and no independent future-task labels in its existing six-label task family. Diabetes has genuine recorded outcomes but no admitted reserved endpoint family or appropriate prospective timing/interface specification. ACS supports custom research but its permitted task/query family and output/API insufficiency are not specified. These are missing prerequisites, not proofs that reusable representations are generally unnecessary.

The learned HAR task releases an accurate coarse activity label and thereby improves inference of fine activity. This is required output content under the coarse task, so absolute independence from fine activity is an incoherent restriction at exact utility. A proposed conditional/incremental policy would have to explicitly permit that disclosure and still report absolute leakage; it was not silently substituted here.

The probability attack is a small fixed20-bin family and finds approximately the same association as hard output. It does not establish that all extra information in probability scores is absent. Subject identity was not audited on unseen people as closed-set classification; that would have unsupported classes. No privacy-pass criterion is declared for these exploratory output checks.

Only official HAR training data and Diabetes processed training labels were read. The learned task uses disjoint task/attacker/development participants. Earlier oracle-label checks use retrospective random-row splits, unsuitable for a session-generalization claim. UCI-supplied normalized features are accepted as dataset input; original provider normalization was not reconstructed. The new scaler uses task-fit rows only. The analytical label/coarsening contradiction does not depend on feature preprocessing.

One learned diagnostic configuration, no three-seed representation-method comparison, no final test, no task holdout or F/G advantage claimed. No further method grid is authorized by these findings. Detailed candidate policies, primary sources and reconsideration criteria are in docs/PCRL_APPLICATION_SELECTION.md; the checked prior-work matrix is docs/PCRL_PRIOR_WORK.md.

Reproduction (writes a new directory and refuses overwrite):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/screen_pcrl_applications.py --out-dir results/redesign_20260907_application_screen_reproduction
```
