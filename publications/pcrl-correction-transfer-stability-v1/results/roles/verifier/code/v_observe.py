"""Verifier observations computed from saved campaign records only (label-free; no data read): whether implementations deliver what
the registered text says, where the numbers themselves already replay exactly.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
  comp     compression twins 'at matched GUARD utilization' (RULES.md section 7 rung 3; cts.compression): achieved / target utilization
           of every saved code and the monotonicity of the evaluated K path (the bisection assumes utilization falls with K).

    python3 .../v_observe.py comp --partition R0 --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

TRACKF = V.ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'


def comp(P, tag):
    rows, nonmono, loose = {}, [], []
    for f in sorted((TRACKF / P / 'COMP').glob('COMP__*/COMP.json')):
        d = json.loads(f.read_text())
        path = [(p['K'], p['utilization']) for p in d['ladder_path']]
        ks = sorted(path)
        mono = all(ks[i][1] >= ks[i + 1][1] for i in range(len(ks) - 1))
        tgt, ach = d['matched_to']['guard_utilization'], d['guard_utilization']
        smaller_ok = [k for k, u in ks if u <= tgt and k < d['K']]
        rows[d['name']] = {'flag': d['flag'], 'K': d['K'], 'target_utilization': tgt, 'achieved_utilization': ach,
                           'achieved_over_target': (ach / tgt) if tgt > 0 else None, 'evaluated_path_sorted_by_K': ks,
                           'path_monotone_decreasing': mono, 'evaluated_smaller_K_also_within_target': smaller_ok}
        if not mono:
            nonmono.append(d['name'])
        if d['flag'] == 'MATCHED' and tgt > 0 and ach / tgt < 0.8:
            loose.append(d['name'])
    chk = {'id': f'C6.2.COMP_MATCH_QUALITY.{P}__{tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': f'{P}: do the compression twins deliver the registered control "at matched GUARD utilization" (rung 3)?',
           'method': 'saved COMP.json records (numbers replayed exactly in C3.COMP): achieved / target utilization, monotonicity of the '
                     'evaluated K path that the bisection assumes',
           'status': 'ATTENTION' if (nonmono or loose) else 'PASS',
           'numbers': {'codes': rows, 'non_monotone_paths': nonmono, 'matched_codes_below_80pct_of_target': loose,
                       'reading': 'MATCHED means "utilization <= target at the K the bisection stopped at"; with a non-monotone path the '
                                  'achieved utilization can be far below the target (a cheaper, weaker control) and TARGET_NOT_REACHED with '
                                  'only K = 2048 evaluated does not show that no K matches'}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_OBSERVE_COMP__{P}__{tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'nonmono': nonmono, 'loose': loose}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('comp',)); ap.add_argument('--partition', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    comp(a.partition, a.tag)


if __name__ == '__main__':
    main()
