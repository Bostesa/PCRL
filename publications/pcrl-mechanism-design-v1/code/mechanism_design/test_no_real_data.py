"""Self-test of the no_real_data guard: refused paths raise before any byte is read; allowed paths open."""
import os
import tempfile

from mechanism_design import no_real_data

no_real_data.install()
home = os.path.expanduser('~')
refused = [
    '<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/exports/DEV_REPORT.npz',
    os.path.join(home, 'PCRL_eval_cache_private', 'acsh_v1', 'x.json'),
    '<external drive>,
    '/tmp/psam_p12.csv',
    '/tmp/some_predictions.npz',
]
for p in refused:
    try:
        open(p, 'rb')
    except PermissionError as e:
        assert 'no_real_data guard' in str(e), e
    except FileNotFoundError:
        raise AssertionError(f'guard did not refuse {p}')
    else:
        raise AssertionError(f'guard did not refuse {p}')
with tempfile.TemporaryDirectory() as d:
    ok = os.path.join(d, 'mechanism_design_synthetic.json')
    with open(ok, 'w') as f:
        f.write('{}')
    assert open(ok).read() == '{}'
print('no_real_data guard self-test: PASS (%d refusals, allowed write/read ok)' % len(refused))
