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
