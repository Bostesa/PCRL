# R2 prefit correctness and CPU engineering amendment request, 2026-10-10

Queued V2 ran 12 data-free tests and failed one. The new independent
Torch-versus-NumPy O-J objective fixture exposed an additional two-dimensional
inverse-index defect in the **NumPy replay**: `np.bincount` received a 2D
array while its action weights were flat. The NumPy inverse is now flattened,
matching the already repaired Torch path. V2's failed receipt remains intact.
No Adult policy fit or role-scored endpoint ran.

The bounded 14-unit policy plan needs a CPU-safe implementation of the same
frozen-reader constraints. Previously, each of up to 600 source updates
recreated dozens of FIT action-loss matrices for every reader and fold. A
round cache now builds frozen Torch cost/reader tensors once and evaluates the
same cross-fitted CE objective and each separate local guard in vectorized
form. A new data-free fixture compares every constraint and the pair
objective to the unvectorized formula to 1e-13 on a fixed finite law.
This changes no policy family, reader, support, fold, objective, cap, safety
rule, update allocation, eta schedule or selection criterion. The old direct
formula remains in source as an independent engineering comparator.

R0 must register a dated fixture/queue amendment before the V3 test runs.
The prospective science lock must bind V3's passed exact sources, the guarded
worker and all transitive package/materialization code. Stop or fail any fit
that exceeds its predeclared CPU ceiling; preserve checkpoint arrays.

V2 failed source pins: `r2_space.py` SHA
`56b131c7b119b60db15beb0418279cf1b8ce87e297f8dbd39f280a098be0c26b`;
V2 receipt is `R2_CORRECTNESS_V2_RECEIPT.json`. V3 public spec index records
new exact hashes after R0 amendment B is registered.

The queue-created V2 traceback contained an absolute user home path. Its
byte-identical original is retained owner-only under `.private`, SHA-256
`cb85f48932a8536d9f7628ff038a6c4848d0f68ce88b46a2ce34b66056e1edda`.
The public receipt replaces only that path prefix with a repository-relative
path, retains the exact error and failed status, and has SHA-256
`76d4e8a9bbf7287d8995a65a3436d8906ad4fe5dac2b090bc45fc4b8e9254580`.
