# R04 queue-spec path cleanup

The first R04 public-source commit (`6e7f11b61`) included three queue-spec
builders with a literal operator-home interpreter path. The builders now use
the interpreter that runs them (`sys.executable`), so invoking a builder with
the original project environment still yields that interpreter in a new
private queue spec. The untracked synthetic spec containing local absolute
paths was moved into R04 owner-only `.private` custody, and future synthetic
specs are written there. The already executed, hash-bound D first-pair and A
source verification inputs and receipts were left unchanged. This records the
historical public-source exposure; repository history was not rewritten.
