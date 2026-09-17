"""Development (2018) tables for the transport package, read from committed aggregate CSVs only."""
from pathlib import Path
import pandas as pd
DEV = Path(__file__).resolve().parents[1]/'results/redesign_20260910_acs_residual_spectral_v1/PUBLICATION_EVIDENCE'
ARMS = ['H', 'E', 'A0', 'L025', 'L20', 'J']+['spectral_'+a for a in ('S0', 'M025', 'M1', 'L025', 'L1', 'C025', 'C1', 'L2')]
ENDS = [('utility_gain_vs_H', 'same_residence', 'res. gain'), ('additional_recovery', 'A/SEX', 'A/SEX'), ('additional_recovery', 'A/RAC1P', 'A/race'),
        ('additional_recovery', 'AB/SEX', 'AB/SEX'), ('additional_recovery', 'AB/RAC1P', 'AB/race')]


def fmt(vals): return f"{vals.mean():.4f} [{', '.join(f'{v:.4f}' for v in vals)}]"


def tables(scope='kernel_expanded_catchup', budget=360):
    d = pd.read_csv(DEV/'PER_SEED.csv.gz'); d = d[(d.split == 'test') & (d.scope == scope) & (d.budget == budget)]
    lines = []
    for w in ('unweighted', 'person_weighted'):
        lines += [f'\n**{w}** (three-seed mean [seed 0, 1, 2]; development split; scope `{scope}`, budget {budget})\n',
                  '| Interface | '+' | '.join(e[2] for e in ENDS)+' |', '|---|'+'---:|'*len(ENDS)]
        for c in ARMS:
            row = []
            for k, e, _ in ENDS:
                q = d[(d.condition == c) & (d.weight == w) & (d.kind == k) & (d.endpoint == e)].sort_values('seed')['value']
                row.append(fmt(q.to_numpy()))
            lines.append(f'| {c} | '+' | '.join(row)+' |')
    return '\n'.join(lines)


def paired(pairs, scope='kernel_expanded_catchup', budget=360):
    d = pd.read_csv(DEV/'PAIRED.csv.gz'); d = d[(d.split == 'test') & (d.scope == scope) & (d.budget == budget)]
    lines = ['| Left − right | Weight | residence loss | A/SEX rec. | AB/SEX rec. | A/race rec. | AB/race rec. |', '|---|---|---:|---:|---:|---:|---:|']
    for l, r in pairs:
        for w in ('unweighted', 'person_weighted'):
            q = d[(d.left == l) & (d.right == r) & (d.weight == w)]
            cells = []
            for kind, e in (('utility', 'same_residence'), ('gains', 'A/SEX'), ('gains', 'AB/SEX'), ('gains', 'A/RAC1P'), ('gains', 'AB/RAC1P')):
                v = q[(q.kind == kind) & (q.endpoint == e)].sort_values('seed')['difference'].to_numpy()
                if len(v) != 3:  # pair absent from the paired file: difference of per-seed selected values
                    ps = pd.read_csv(DEV/'PER_SEED.csv.gz'); ps = ps[(ps.split == 'test') & (ps.scope == scope) & (ps.budget == budget) & (ps.weight == w)]
                    k = 'utility_loss' if kind == 'utility' else 'absolute_recovery'
                    get = lambda c: ps[(ps.condition == c) & (ps.kind == k) & (ps.endpoint == e)].sort_values('seed')['value'].to_numpy()
                    v = get(l)-get(r)
                cells.append(fmt(v))
            lines.append(f'| {l} − {r} | {w} | '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def scope_sensitivity(arms=('J', 'spectral_C1', 'spectral_L1', 'spectral_L2', 'spectral_S0')):
    d = pd.read_csv(DEV/'PER_SEED.csv.gz'); d = d[(d.split == 'test') & (d.budget == 360) & (d.weight == 'unweighted') & (d.kind == 'additional_recovery')]
    scopes = ['standard_independent', 'kernel_standard_independent', 'expanded_catchup', 'kernel_expanded_catchup']
    lines = ['| Interface | endpoint | '+' | '.join(scopes)+' |', '|---|---|'+'---:|'*len(scopes)]
    for c in arms:
        for e in ('AB/SEX', 'A/RAC1P', 'AB/RAC1P'):
            vals = [d[(d.condition == c) & (d.endpoint == e) & (d.scope == s)]['value'].mean() for s in scopes]
            lines.append(f'| {c} | {e} | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(tables()); print(); print(paired([('spectral_C1', 'spectral_L1'), ('spectral_C1', 'spectral_L2'), ('spectral_C025', 'spectral_L025'), ('J', 'L025'), ('J', 'L20'), ('spectral_C1', 'J'), ('spectral_S0', 'A0'), ('spectral_L1', 'spectral_M1'), ('spectral_M1', 'spectral_S0')]))
    print(); print(scope_sensitivity())
