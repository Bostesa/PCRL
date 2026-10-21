"""V5c exact aliases among saved releases (label-free), own code.

Per version, every pair of saved configurations of this campaign (runs/FA/<cfg>/V<v>/RELEASE.npz, hash-checked) is compared on the
transmitted vectors of all six roles: an exact alias transmits bitwise-identical income and occupation vectors to every person of
every role (so every reader sees the same inputs). Reported beside: pairs identical on MODEL_SELECT only, and, per configuration,
whether it is an exact alias of its version's task-only twin (PROTOCOL 7: a nonzero setting that changes nothing is an exact
alias). The registered saving rule detects aliases by SELECTION NAME (a budget whose selected strength equals an earlier budget's);
this check detects aliases by the transmitted bytes.

    python3 .../v5c_exact_aliases.py --tag V1
    python3 .../v5c_exact_aliases.py --selftest
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402


def role_digests(npz_path):
    out = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in sorted(z.files):
            out[k] = V.arr_sha(z[k])
    return out


def aliases(digests):
    """digests: {cfg: {key: sha}} of one version -> (exact alias pairs, MODEL_SELECT-only identical pairs)."""
    exact, ms_only = [], []
    for a, b in itertools.combinations(sorted(digests), 2):
        da, db = digests[a], digests[b]
        if da == db:
            exact.append([a, b])
        elif all(da.get(k) == db.get(k) for k in da if k.startswith('MODEL_SELECT__')):
            ms_only.append([a, b])
    return exact, ms_only


def run(tag):
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'per': {}}
    for v in (0, 1):
        dig, bad = {}, []
        for d in sorted((V.RUNS / 'FA').iterdir()):
            md = d / f'V{v}'
            if not (md / 'RELEASE.npz').is_file():
                continue
            rec = json.loads((md / 'RELEASE.json').read_text())
            if V.file_sha(md / 'RELEASE.npz') != rec.get('release_sha256'):
                bad.append(d.name)
            dig[d.name] = role_digests(md / 'RELEASE.npz')
        exact, ms_only = aliases(dig)
        twin = 'TWB_C64' if 'TWB_C64' in dig else None
        twin_alias = sorted(c for pair in exact for c in pair if twin in pair and c != twin) if twin else []
        record['per'][f'V{v}'] = {'configurations': sorted(dig), 'exact_alias_pairs': exact, 'model_select_only_identical_pairs': ms_only,
                                  'exact_aliases_of_twin': twin_alias, 'hash_mismatches': bad}
        checks.append({'id': f'V5C.EXACT_ALIASES_V{v}__{tag}',
                       'what': f'version {v}: pairs of saved configurations transmitting bitwise-identical vectors on all six roles '
                               f'(exact aliases), MODEL_SELECT-only identical pairs, exact aliases of the twin; descriptive input to the '
                               f'alias handling of PROTOCOL 7 / RULES 7 (name-based) and to the grid and configuration accounting',
                       'inputs': {'configurations': sorted(dig)}, 'result': record['per'][f'V{v}'],
                       'numerical_status': 'PASS' if not bad else 'FAIL',
                       'procedural_status': 'ATTENTION' if (exact or ms_only) else 'PASS'})
    V.write_record(f'V5C_ALIASES__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'], c['result']['exact_alias_pairs'],
              c['result']['model_select_only_identical_pairs'], c['result']['exact_aliases_of_twin'])
    return 0


def selftest():
    d = {'A': {'MODEL_SELECT__income': '1', 'MODEL_FIT__income': '2'}, 'B': {'MODEL_SELECT__income': '1', 'MODEL_FIT__income': '2'},
         'C': {'MODEL_SELECT__income': '1', 'MODEL_FIT__income': '3'}, 'D': {'MODEL_SELECT__income': '9', 'MODEL_FIT__income': '2'}}
    exact, ms = aliases(d)
    ok = {'exact_pair_found': exact == [['A', 'B']], 'ms_only_pairs_found': sorted(ms) == [['A', 'C'], ['B', 'C']],
          'different_not_reported': all('D' not in p for p in exact + ms)}
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
