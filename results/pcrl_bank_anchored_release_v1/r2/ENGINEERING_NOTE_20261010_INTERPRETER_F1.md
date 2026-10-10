# R2 protection interpreter correction, 2026-10-10

The first E-amendment unit, `R2_B0_S0_FIT`, failed at import time before any
role-array read or FIT output. Its immutable queue stderr shows
`ModuleNotFoundError: scipy`. The spec builder called `Path.resolve()` on
the configured virtual-environment Python entry, turning it into the base
interpreter path without the virtual environment's packages. The original
E configs/specs and failure receipt remain preserved.

The F1 builder keeps the absolute configured entry path without following
its symlink, checks that it is executable, and uses that exact path for a
data-free import smoke of NumPy, SciPy, PyTorch and the R2 worker. A synthetic
regression verifies symlink preservation. The 16 successor unit IDs append
`_F1`; recipe seeds, source roles, method choices, FIT caps and 300-second
optimizer ceiling are unchanged. A dated R0 amendment must seal the new
source index and IDs before any F1 spec or FIT work.
