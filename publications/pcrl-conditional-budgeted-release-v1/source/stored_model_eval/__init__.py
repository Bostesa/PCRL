"""stored_model_eval: evaluate stored model releases (representations, outputs) for attribute recovery.

Design rules (enforced in code, see guards.py):
  * the default mode performs no scientific fits and never acquires data (no network);
  * attacker fitting on non-synthetic inputs requires --execute-scientific-fits;
  * a quantity that cannot be estimated is a NotEstimable sentinel, never 0/1/pass/fail;
  * every decision is an interval decision vs a declared bar: ESTABLISHED_BELOW / ESTABLISHED_ABOVE /
    UNRESOLVED / NOT_ESTIMABLE (there is no "pass").

numpy/scipy/sklearn only at import time; torch is imported lazily inside forward.py.
"""
__version__ = "0.1.0"
SCHEMA_PROTOCOL = "stored_model_eval.protocol/v1"
SCHEMA_MANIFEST = "stored_model_eval.manifest/v1"
