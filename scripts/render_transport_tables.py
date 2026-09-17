"""Render TRANSPORT_RESULTS.md tables from the locked result files (no recomputation of scores)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from experiments import acs_spectral_transport_eval as ev

OUT = ev.OUT
ORD = ['H', 'E', 'A0', 'L025', 'L20', 'J']+['spectral_'+x for x in ('S0', 'M025', 'M1', 'L025', 'L1', 'C025', 'C1', 'L2')]
W = {'unweighted': 'unweighted', 'person_weighted': 'PWGTP'}


def f(v, n=4): return '—' if pd.isna(v) else f'{v:.{n}f}'


def headline(mode, scope, agg):
    lines = []
    for w in ('unweighted', 'person_weighted'):
        lines += [f'\n*{W[w]}* (three-seed mean, household-bootstrap 95% interval)\n',
                  '| Interface | residence gain vs H | A/SEX | A/race | AB/SEX | AB/race |', '|---|---:|---:|---:|---:|---:|']
        for c in ORD:
            cells = []
            for kind, e in (('utility_gain_vs_H', 'same_residence'), ('additional_recovery', 'A/SEX'), ('additional_recovery', 'A/RAC1P'),
                            ('additional_recovery', 'AB/SEX'), ('additional_recovery', 'AB/RAC1P')):
                q = agg[(agg['mode'] == mode) & (agg.scope == scope) & (agg.condition == c) & (agg.weight == w) & (agg.kind == kind) & (agg.endpoint == e)]
                if not len(q): cells.append('—'); continue
                r = q.iloc[0]
                cells.append(f(r['seed_mean'])+('' if c == 'H' or pd.isna(r['boot_low']) else f" [{f(r['boot_low'])}, {f(r['boot_high'])}]"))
            lines.append(f'| {c} | '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def family_table(fam, name):
    q = fam[fam.family == name]
    lines = ['| Contrast | Endpoint | Weight | Estimate | Seeds 0/1/2 | Adjusted 95% | Significant |', '|---|---|---|---:|---|---|---|']
    for _, r in q.iterrows():
        sig = 'better' if r['adjusted_high'] < 0 else ('worse' if r['adjusted_low'] > 0 else '')
        lines.append(f"| {r['left']} − {r['right']} | {r['endpoint']} | {W[r['weight']]} | {f(r['estimate'],5)} | "
                     f"{f(r['seed_0'],4)} / {f(r['seed_1'],4)} / {f(r['seed_2'],4)} | [{f(r['adjusted_low'],5)}, {f(r['adjusted_high'],5)}] | {sig} |")
    return '\n'.join(lines)


def criteria_table(crit):
    lines = ['| Interface | Mode | Weight | mean residence gain | min seed | .01 in all seeds | half-headroom seeds | source allowance seeds |', '|---|---|---|---:|---:|---|---:|---:|']
    for mode in ('B', 'A'):
        for c in ORD:
            for w in ('unweighted', 'person_weighted'):
                q = crit[(crit['mode'] == mode) & (crit.condition == c) & (crit.weight == w)]
                if not len(q): continue
                r = q.iloc[0]
                lines.append(f"| {c} | {mode} | {W[w]} | {f(r['residence_gain_mean'])} | {f(r['residence_gain_min'])} | "
                             f"{'yes' if r['residence_01_all_seeds_and_mean'] else 'no'} | {int(r['half_headroom_pass_seeds'])}/3 | {int(r['source_allowance_pass_seeds'])}/3 |")
    return '\n'.join(lines)


def scope_table(agg):
    lines = ['| Interface | Endpoint | Mode B common_fresh | Mode B transport_all | Mode A kernel_expanded_catchup |', '|---|---|---:|---:|---:|']
    for c in ('J', 'spectral_C1', 'spectral_L1', 'spectral_L2', 'spectral_S0'):
        for e in ('AB/SEX', 'A/RAC1P', 'AB/RAC1P'):
            vals = []
            for mode, scope in (('B', 'common_fresh'), ('B', 'transport_all'), ('A', 'kernel_expanded_catchup')):
                q = agg[(agg['mode'] == mode) & (agg.scope == scope) & (agg.condition == c) & (agg.weight == 'unweighted') & (agg.kind == 'additional_recovery') & (agg.endpoint == e)]
                vals.append(f(q['seed_mean'].iloc[0]) if len(q) else '—')
            lines.append(f'| {c} | {e} | '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def service_table():
    s = pd.read_csv(OUT/'evidence/SERVICE_QUALITY.csv').groupby('task')[['development_unweighted', 'transport_unweighted', 'development_pwgtp', 'transport_pwgtp', 'development_accuracy', 'transport_accuracy']].mean()
    lines = ['| Service task | 2018 log loss | 2017 log loss | 2018 PWGTP | 2017 PWGTP | 2018 accuracy | 2017 accuracy |', '|---|---:|---:|---:|---:|---:|---:|']
    for t, r in s.iterrows():
        lines.append(f"| {t} | {f(r['development_unweighted'])} | {f(r['transport_unweighted'])} | {f(r['development_pwgtp'])} | {f(r['transport_pwgtp'])} | {f(r['development_accuracy'])} | {f(r['transport_accuracy'])} |")
    return '\n'.join(lines)


def cost_table():
    c = pd.read_csv(OUT/'evidence/CANDIDATE_COSTS.csv')
    lines = ['| Contrast | Weight | Endpoint | Estimate | Unadjusted 95% |', '|---|---|---|---:|---|']
    for _, r in c.iterrows():
        lines.append(f"| {r['left']} − {r['right']} | {W[r['weight']]} | {r['endpoint']} | {f(r['estimate'],5)} | [{f(r['unadjusted_low'],5)}, {f(r['unadjusted_high'],5)}] |")
    return '\n'.join(lines)


def consistency_table():
    d = pd.read_csv(OUT/'evidence/DEVELOPMENT_CONSISTENCY.csv')
    g = d.groupby(['family', 'status']).size().reset_index(name='endpoints')
    lines = ['| Family | Status | Endpoints |', '|---|---|---:|']
    for _, r in g.iterrows(): lines.append(f"| {r['family']} | {r['status']} | {r['endpoints']} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    agg = pd.read_csv(OUT/'AGGREGATE.csv'); fam = pd.read_csv(OUT/'FAMILIES.csv'); crit = pd.read_csv(OUT/'evidence/CRITERIA_SUMMARY.csv')
    out = {'headline_B': headline('B', 'transport_all', agg), 'headline_A': headline('A', 'kernel_expanded_catchup', agg),
           'F1': family_table(fam, 'F1_primary'), 'F2': family_table(fam, 'F2_neural_replication'), 'F3': family_table(fam, 'F3_secondary'),
           'F4': family_table(fam, 'F4_frozen_transfer'), 'criteria': criteria_table(crit), 'scopes': scope_table(agg),
           'service': service_table(), 'costs': cost_table(), 'consistency': consistency_table()}
    Path(OUT/'evidence/tables.json').write_text(json.dumps(out, indent=1)+'\n')
    for k, v in out.items(): print('##', k); print(v); print()
