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
