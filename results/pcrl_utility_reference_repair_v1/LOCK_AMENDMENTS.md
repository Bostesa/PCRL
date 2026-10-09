# Engineering lock amendments

## 2026-10-09: parity interpreter path, before Adult replay

The first parity lock at commit `09b02e924e90a6e7d7679eda29038c9daee3b102`
and SHA-256 `41118373e368928f3bfd14c82386d9997cf7674102d15bfe1dbab218862a7723`
listed 12 role-bound parity units, but its builder resolved the project virtual
environment's `python` symlink into a base framework Python. R3's data-free
import check found that the resulting executable could not import Torch.
No parity or scientific unit was admitted or run under this lock. The original
public lock remains in Git history; the original lock and private registry
also remain in `.private/lock_attempts/parity_v1/` for local replay.

The repair preserves the literal virtual environment launcher in each unit
argv. Role files, source checkpoints, endpoint definitions, `2e-7` parity
tolerance, and the old assessment embargo are unchanged. R3 must review the
new committed lock and registry before any Adult replay.

## 2026-10-09: FIT-only screen usefulness receipt, before any recipe fit

The first FIT-only screen lock at commit
`0a0dc77b4cbe7dc8014ec379fb0e377afbcbca71` and SHA-256
`5272150632ab3142b54d36a768364cf4eb6286ffae8c12d4dc47276c118ac52a`
was not reviewed, admitted or run. The worker would have saved A/B outer-fold
task scores without the predeclared accuracy gain over a FIT-fitted constant
predictor. That would leave the usefulness prerequisite unreplayable from the
screen receipt alone. The original lock and registry remain in Git history and
`.private/lock_attempts/screen_v1_missing_constant/`.

The corrected worker records each outer-training-fold majority class (lowest
class ID on ties), its accuracy on the corresponding outer validation fold,
and A/B accuracy gains. The selection rule, role access, utility caps and
conditional branch remain unchanged. R3 must review the corrected committed
lock before any recipe fit.

## 2026-10-09: independent lead replay package identity, before admission

The first lead replay lock at commit `5670bce20` and SHA-256
`8717fa0b879dcc487a60d3058108fb9e0a32f19e519c9e2ac371b2dff4a5ef66`
bound replay worker SHA `baca55bb3faa7fa366ae33e8302db8f0b74449f233cbecdf1d8967eb0e86029f`.
Before R3 prefit approval, its reviewer strengthened package identity checks
and committed worker SHA `5df4bfcf115592349b909b26d1aa2933e002635431cea5efd35cb654a3ec4907`.
The first lock therefore failed its source hash and was never sealed,
admitted or run. Its lock and private unit registry remain in Git history
and `.private/lock_attempts/lead_replay_stale_worker_before_review/`.

The corrected lock binds the final worker bytes. The two lead masks, rows,
task references, numeric gates, inputs and output inventory are unchanged.
R3 must review the corrected committed lock before independent replay.
