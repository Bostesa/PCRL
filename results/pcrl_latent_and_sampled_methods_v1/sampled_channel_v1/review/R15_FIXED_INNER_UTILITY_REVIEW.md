# R15 review: fixed A INNER utility verifier

Status: **SCOPED STATIC GO** for corrected source. Numerical verification remains pending.

- Worker SHA-256: `7079eb7efbfaf773094898811afc0ae607d560b98fd002f9364e32e96e790282`.
- R1 proposal SHA-256: `e14ecdc1f944d56bef3ce227aea0afe855f340708aac7a65606e304b4e3660a1`.
- Correction commit: `aab65ab78`; the original proposal and worker bytes are preserved.

My first review found that the role loader materialized every NPZ member before checking for forbidden members. The corrected loader checks the exact task or reference member set before accessing any array. A synthetic negative plants a `forbidden_sex` member. The worker reads INNER task labels and U0/Ucal references only, never SEX or the old assessment. It derives canonical diagnostic codes, checks each saved code, probability, decision and full public wire hash, then computes point utility against both frozen references. Six saved INNER release units are explicit dependencies. The 120-second CPU/wall and 2-GiB limits are declared in the R03 proposal.

This approval requires R00 to seal the dated verification amendment and R03 to enforce its exact spec. It supports no interval, privacy, or numerical pass claim.
