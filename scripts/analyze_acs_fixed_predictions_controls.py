"""Post-fit fixed comparator and incremental-reference exports; no fitting/selection."""
from pathlib import Path
import argparse
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
LABELS={'H':'H','E':'H+E_A','A0':'H+A0','L025':'H+L025','L20':'H+L20','J':'H+J'}
def main(out):
    inc=pd.read_csv(out/'INCREMENTAL.csv')
    k=inc[(inc.split=='test')&(inc.budget==360)&(inc.scope=='expanded_catchup')].copy()
    k['within_point005_reference']=[(x<=.005+1e-12) if e.endswith(('/SEX','/RAC1P')) else None for x,e in zip(k.increment,k.endpoint)]
    k['race_full_category_assessable']=k.endpoint.apply(lambda x:False if x.endswith('/RAC1P') else None)
    # The .005 reference only has scientific meaning for sensitive targets.
    k.to_csv(out/'INCREMENT_REFERENCE_CHECKS.csv',index=False)
    p=pd.read_csv(out/'PAIRED.csv');p=p[(p.left=='J')&p.right.isin(['L025','L20','A0','E'])&(p.split=='test')&(p.budget==360)]
    lines=['# All fixed J control contrasts','', 'J minus comparator, in nats. Negative task loss is better utility; negative recovery gain is less measured prediction. All three seeds enter every mean. This is a component table, not a dominance claim. See [research decision](RESEARCH_DECISION.md) and [matching](MATCHING_ANALYSIS.md).','']
    for (w,s),g in p.groupby(['weight','scope'],sort=False):
        lines += [f'## {w}: {s}, 360 epochs','', '| Comparator | Endpoint | Mean ± sample SD | Seed 0 | Seed 1 | Seed 2 |','| --- | --- | --- | --- | --- | --- |']
        for (c,e),v in g.groupby(['right','endpoint'],sort=False):
            v=v.sort_values('seed').difference
            lines.append(f'| {LABELS[c]} | {e} | {v.mean():+.6f} ± {v.std():.6f} | '+' | '.join(f'{n:+.6f}' for n in v)+' |')
        lines+=['']
    (out/'ANALYSIS.md').write_text('\n'.join(lines).rstrip()+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1');a=p.parse_args();main(a.out)
