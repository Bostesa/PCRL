# Secondary context-shared free kernels

NARROW uses all 4×4 original nearest-neighbor position templates, including
KEEP×KEEP, with one categorical weight vector per frozen supported context.
A template not executable for every fitting row in that context has bound
zero. The original nearest-neighbor construction and code bank are unchanged.
The simplex sums all 16 template weights to one. A supported deployment input
with a missing positively weighted neighbor receives exact P0.

WIDE uses each recipient's template list `[KEEP, every original global bank
ID with the context's predicted class]`. The joint template list is their
Cartesian product, with one categorical vector per frozen supported context.
KEEP maps to the input's own code. Explicit global IDs can coincide with
KEEP for some inputs. All such aliases combine into the same actual emitted
tuple in objective, utility, scoring and sampling. No action position is
transmitted. All declared columns are included, so there is no omitted-column
pricing claim and no favorable bank pruning.

The input contexts and their count-based hierarchy are inherited from the
primary comparison without outcome-dependent refinement. Unsupported contexts
are exact identity. Each supported context has one class pair; inconsistent
input/schema/context bindings are rejected. The parameter count is 16 times
supported contexts for NARROW, and the sum of `(1+k1)*(1+k2)` over supported
contexts for WIDE, where k is the original bank's class-specific state count.
This is greater capacity and changed support relative to frozen-donor G/C.

Before any dense row/action array allocation, the builder estimates
`rows*max_actions*84 + parameters*64 + rows*64` bytes, accounting conservatively
for emitted IDs, base, intermediate COO/CSR and temporary masks. If this
exceeds the registered input-memory ceiling, it raises MemoryError without
allocating those arrays. LP matrix storage is measured separately, and solver
workspace must have a separate scheduler reserve. A memory block is an
untested dependency, not evidence against a broader-support mechanism.
