# Cached deployment sampling contract

The provider routes a permitted raw83 input through the frozen P0 teacher,
router and decoder. The release mechanism sees only the resulting original
code pair and fixed decoded values. A provider person key enters only the
private cache. It is neither a learned routing feature nor a public field.
Duplicate records using one provider person key share a cached emitted tuple;
a reused key with a different float32 raw row is refused.

G, C and the source V-J reference use one joint inverse-CDF draw. Local and
product policies use two recipient-domain draws and are required to have an
actual product law. Freeze-first sequential policies and source V-S21 use
one draw for each recipient, sampling the first marginal then the declared
conditional law in the specified order. Deterministic policies must have one
actual emitted tuple. Duplicate action aliases are combined before marginal
or conditional sampling. Action positions are private and not transmitted.

The private sampling root is 256 bits. HMAC-SHA256 domains bind sampler kind,
policy receipt identity, source binding, source seed and canonical provider
key. The recipient domains are distinct. The sampler retains 53 bits for a
uniform on the binary grid. Independence across distinct people and recipient
domains is the usual ideal PRF assumption, not an information theoretic result
about HMAC. The root and cache keys remain private; publishing the root would
change this contract.

Exact integrated AUC refers to the ideal independent continuous-uniform
finite-law estimand conditional on the fitted kernel and readers. It removes
release-draw Monte Carlo error for that law. Floating inverse-CDF cutoffs on
the 53-bit grid can perturb each action mass by at most 2^-53 per cutoff.
A joint row with k actions therefore has a conservative total error of
2*k/2^53. Two-stage sampling admits the corresponding sum of both marginal
and conditional cutoff errors. These errors fall below the registered
numerical tolerance at this study's finite bank sizes. They are numerical
implementation limits; they do not imply population privacy or fresh data.
A numerical deployment replay with pinned private roots is separate from
the expected-law estimand.

Only one cached release is returned for repeated queries for one person.
This interface does not provide independently resampled repeated queries.
The cache-key identity and duplicate-group pairing must be maintained by the
provider. Duplicate-group SEX consistency is checked only in roles where SEX
is authorized. INNER SEX remains closed until the final reporting seal.
