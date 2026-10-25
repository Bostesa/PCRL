"""Runnable entry point of the EDG1 recovery-estimator diagnostic (synthetic data only). From the repository root:

    python3 results/pcrl_estimator_diagnostic_v1/edg_run.py laws        # write LAWS.json (known laws and population values)
    python3 results/pcrl_estimator_diagnostic_v1/edg_run.py selftest    # tiny-law and source-equality checks
    python3 results/pcrl_estimator_diagnostic_v1/edg_run.py simulate --cells 0-48 --reps 500 --label all
    python3 results/pcrl_estimator_diagnostic_v1/edg_run.py bench       # reader-bank runtime microbenchmark (timing only)
    python3 results/pcrl_estimator_diagnostic_v1/edg_run.py bank --label all
    python3 results/pcrl_estimator_diagnostic_v1/edg_run.py analyze     # SUMMARY.json, TABLES.md, figures
One numerical thread per process (set in edg/__init__.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == 'laws':
        from edg import laws, RESULTS
        rec = laws.write_laws(RESULTS / 'LAWS.json')
        print('all_checks_pass', rec['all_checks_pass'])
    elif cmd == 'grid':
        from edg import grid
        grid.main(rest)
    elif cmd == 'selftest':
        from edg import selftest
        selftest.main(rest)
    elif cmd == 'simulate':
        from edg import simulate
        simulate.main(rest)
    elif cmd in ('bench', 'bank'):
        from edg import bank
        bank.main([cmd] + rest)
    elif cmd == 'analyze':
        from edg import analyze
        analyze.main(rest)
    elif cmd == 'posthoc':
        from edg import posthoc
        posthoc.main(rest)
    elif cmd == 'custody':
        from edg import custody
        custody.main(rest)
    elif cmd == 'publish':
        from edg import publish
        publish.main(rest)
    else:
        raise SystemExit(f'unknown command {cmd}')


if __name__ == '__main__':
    main()
