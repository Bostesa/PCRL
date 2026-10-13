"""Fixed precision-only corrected LP backend; same program and certificate cap."""
from .memory_lp import protection_program,direct_bayes,LazyTaskCosts,expected_rows
from .lp import solve_program


def solve_protection(law,sensitive,constraints,*,time_limit=300.,tolerance=1e-8):
    # Caller tolerance remains the registered executable/checker tolerance;
    # only the solver feasibility request is more exact. The shared certificate
    # still uses its original max(1e-7,10*tolerance) arithmetic rule:1e-7 here.
    if tolerance!=1e-8:raise ValueError('registered checker tolerance changed')
    return solve_program(protection_program(law,sensitive,constraints),time_limit=time_limit,tolerance=1e-10)
