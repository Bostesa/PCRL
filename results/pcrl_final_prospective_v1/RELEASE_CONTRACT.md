# Release contract (prospective replacement setting)

Recipients already hold the services H. J is a comparator here; there is no evidence it was ever deployed.

| Recipient | Receives |
|---|---|
| A | [H_A, Z]; H_A holds the frozen income_binary and civilian_at_work service probabilities (4 float64 columns) |
| B | H_B, unchanged; H_B holds the frozen public_coverage service probabilities (2 columns) |
| AB (coalition) | [H_A, H_B, Z] |

What Z is for each family:

- **Token families (Q, D17, D33, RR75, W75).** Z is one categorical token. It is drawn once per person
  from the public kernel row for that person's T0 code, using a private persistent coin. Repeated requests
  return the same token. Independent redraws would be a different disclosure problem and are not part of
  this contract.
- **J.** Z is the frozen 16-coordinate mapper output.
- **C.** Z is the frozen teacher probability p.
- **E and S.** Z is the 33-coordinate affine eraser output.
- **H.** Z is empty.

The encoder's only permitted inference inputs are frozen PCA32 and H_A. PCA32 comes from the ten raw
covariates AGEP, WKHP, SCHL, MAR, RELP, CIT, DIS, DEAR, DEYE and DREM, standardized with the saved J
mean and scale.

The following never enter the encoder or the wire: residence labels, protected labels, source labels,
evaluation outcomes, partition identity, H_B, B-only quantities, household indices, survey weights, the Q
row, the internal code, and the private coin. A person identifier may locate the persistent coin; it is
never a predictive feature.

H parity requirements:

- The H columns of every interface must be byte-identical on the same host and input rows.
- "Exact preservation" means the service outputs are unchanged. It does not mean service accuracy is
  unchanged after transport to 2016; accuracy is measured separately.
- Historical byte fingerprints (2018 parity) and cross-platform toleranced compatibility are reported
  separately.

Scope of what the release and its evaluation certify:

- The finite constraints fitted in 2018 concern coarse conditioning cells. They are not a guarantee
  conditional on continuous H, or against joint-attribute, arbitrary-side-information or population
  attacks.
- Full-H attacker results are empirical diagnostics under the stated slate. They do not bound information.
