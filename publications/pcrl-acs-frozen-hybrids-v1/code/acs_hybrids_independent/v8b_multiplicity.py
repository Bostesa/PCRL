"""V8 supplement (no data access): source K = 40 labels and the settled-before-readers clause bounds.

1. For every endpoint with a source counterpart that carries a source sim_status, the campaign's source_K40_status must
   equal the published source status, its K38_status_at_source_margin must equal the verifier's own K = 38 status at that
   margin (from V8), and status_changed_by_multiplicity_only must equal (verifier status != source status).
2. ANALYSIS_LOCK.settled_before_readers clause bounds: compared with the verifier's exact-t U1 (V8) and with
   published point + 2.3950 x published SE, to identify which critical value the lock used.
"""
from __future__ import annotations

import argparse

from . import vcommon as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(V.OUT / 'V8b_multiplicity_labels.json'))
    a = ap.parse_args()
    L = V.analysis_lock()
    camp = V.jload(V.RES / 'DEV_REPORT_RESULTS.json')
    src = V.jload(V.SRC_RES / 'DEV_REPORT_RESULTS.json')
    v8 = V.jload(V.OUT / 'V8_endpoints_and_multiplicity.json')
    rows, bad = {}, []
    for fam, roster, mine in (('endpoints_primary', L['primary_roster'], v8['primary']), ('endpoints_secondary', L['secondary_roster'], v8['secondary'])):
        for sp in roster:
            cp, k = sp.get('source_counterpart'), sp['key']
            rec40 = None
            if cp and cp.get('type') == 'endpoint':
                rec40 = src['endpoints'][cp['key']]
            elif cp and cp.get('type') == 'service_utility':
                rec40 = src['per_seed'][str(sp['seed'])]['services'][cp['service']]['utility'][cp['key']]
            elif cp and cp.get('type') == 'service_subgroup':
                rec40 = src['per_seed'][str(sp['seed'])]['services'][cp['service']]['subgroup'][cp['group']]
            if rec40 is None or rec40.get('sim_status') is None:
                continue
            m40 = rec40.get('margin')
            st = mine[k]['status']
            my = st.get(str(m40)) if isinstance(st, dict) else (st if m40 == sp.get('margin') else None)
            c = camp[fam][k]
            r = {'source_margin': m40, 'source_status': rec40['sim_status'], 'verifier_K38_status_at_margin': my,
                 'campaign_source_K40_status': c.get('source_K40_status'), 'campaign_K38_status_at_source_margin': c.get('K38_status_at_source_margin'),
                 'campaign_changed_flag': c.get('status_changed_by_multiplicity_only')}
            r['ok'] = bool(r['campaign_source_K40_status'] == r['source_status'] and r['campaign_K38_status_at_source_margin'] == my
                           and r['campaign_changed_flag'] == (my != r['source_status']))
            if not r['ok']:
                bad.append(k)
            rows[k] = r
    settled = L['settled_before_readers']['per_key']
    sb = {}
    for k, v in settled.items():
        if 'clause_bound_U1' in v:
            mineU1 = v8['primary'][k]['U1']
            sb[k] = {'lock_U1': v['clause_bound_U1'], 'verifier_exact_t_U1': mineU1, 'abs_diff_vs_exact_t': abs(v['clause_bound_U1'] - mineU1),
                     'abs_diff_vs_rounded_2.3950': abs(v['clause_bound_U1'] - (v['published_point'] + 2.395 * v['published_se'])),
                     'clause_pass_lock': v['clause_pass'], 'clause_pass_verifier': v8['primary'][k]['clause_pass']}
    out = {'check': 'V8 supplement: multiplicity labels and settled clause bounds', 'scope': 'JSON comparison only (no data access)',
           'endpoints_with_source_status': len(rows), 'label_failures': bad, 'changed_by_multiplicity_only': [k for k, r in rows.items() if r['campaign_changed_flag']],
           'rows': rows,
           'settled_clause_bounds': {'n': len(sb), 'max_abs_diff_vs_exact_t': max(x['abs_diff_vs_exact_t'] for x in sb.values()),
                                     'max_abs_diff_vs_rounded_2.3950': max(x['abs_diff_vs_rounded_2.3950'] for x in sb.values()),
                                     'clause_decisions_identical': all(x['clause_pass_lock'] == x['clause_pass_verifier'] for x in sb.values()),
                                     'min_distance_of_any_U1_from_its_margin': min(abs(v8['primary'][k]['U1'] - next(sp['margin'] for sp in L['primary_roster'] if sp['key'] == k)) for k in sb)}}
    out['status'] = 'PASS' if (not bad and out['settled_clause_bounds']['clause_decisions_identical']) else 'FAIL'
    V.write_json(a.out, out)
    print({'status': out['status'], 'n': len(rows), 'bad': bad, 'settled': out['settled_clause_bounds']})


if __name__ == '__main__':
    main()
