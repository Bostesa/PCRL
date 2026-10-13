"""Queue-only standalone figures from public aggregate CSV; observed points only."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path


def run(spec):
    from .evaluation_worker import queue_config
    active,cfg,verify,guard=queue_config(spec,'verification')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .source import write_json
    public=Path(__file__).resolve().parents[1]/'results/pcrl_compressed_income_safeguard_v1'
    figures=[]
    for chart in cfg['charts']:
        pin=chart['csv'];p=verify(pin['path'],pin['sha256'])
        if not p.resolve().is_relative_to(public):raise PermissionError('figures accept public aggregate CSV only')
        with p.open(newline='') as f:rows=list(csv.DictReader(f))
        fig,ax=plt.subplots(figsize=(7.2,4.7));groups={};points=[]
        for row in rows:
            if any(row.get(k)!=v for k,v in chart.get('filters',{}).items()):continue
            try:x=float(row[chart['x']]);y=float(row[chart['y']])
            except (ValueError,KeyError):continue
            import math
            if not math.isfinite(x) or not math.isfinite(y):continue
            group=' / '.join(row.get(k,'') for k in chart.get('group_by',[])) or 'observed release'
            groups.setdefault(group,[]).append((x,y,row));points.append((x,y))
        if not points:raise ValueError('no finite observed aggregate points; figure unavailable')
        for group,values in sorted(groups.items()):
            ax.scatter([v[0] for v in values],[v[1] for v in values],label=group,s=38,alpha=.85)
            if chart.get('interval_columns'):
                cols=chart['interval_columns']
                for x,y,row in values:
                    try:
                        xerr=None if 'x_lower' not in cols else [[x-float(row[cols['x_lower']])],[float(row[cols['x_upper']])-x]]
                        yerr=None if 'y_lower' not in cols else [[y-float(row[cols['y_lower']])],[float(row[cols['y_upper']])-y]]
                    except (ValueError,KeyError):continue
                    if any(value<0 for err in (xerr,yerr) if err is not None for side in err for value in side):raise ValueError('interval excludes reported point')
                    ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='none',ecolor='#555555',alpha=.6,linewidth=.8,capsize=2)
            if chart.get('annotate'):
                for x,y,row in values:ax.annotate(' / '.join(row.get(k,'') for k in chart['annotate']),(x,y),xytext=(4,4),textcoords='offset points',fontsize=7)
        # Report frontier membership through rings, never a fabricated curve.
        if chart.get('pareto_minimize_both'):
            frontier=[p for p in points if not any(q[0]<=p[0] and q[1]<=p[1] and q!=p for q in points)]
            if frontier:ax.scatter([p[0] for p in frontier],[p[1] for p in frontier],facecolors='none',edgecolors='black',s=95,linewidths=1,label='discrete observed Pareto member')
        for line in chart.get('lines',[]):
            if line['axis']=='x':ax.axvline(float(line['value']),color='#9b3030',linestyle='--',linewidth=1,label=line['label'])
            elif line['axis']=='y':ax.axhline(float(line['value']),color='#9b3030',linestyle='--',linewidth=1,label=line['label'])
            else:raise ValueError('unknown threshold axis')
        ax.set(xlabel=chart['xlabel'],ylabel=chart['ylabel'],title=chart['title']);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=7,loc='best');fig.text(.02,.012,'Observed checkpoints only; no interpolated deployable release. FIT selection values are diagnostic proxies.\nUtility caps: accuracy loss ≤0.010; NLL excess ≤0.010; summed Brier excess ≤0.005.',fontsize=7);fig.tight_layout(rect=(0,.055,1,1))
        for format in ('pdf','png'):
            dest=Path(cfg['output_dir'])/(chart['name']+'.'+format)
            if not dest.resolve().is_relative_to(public) or dest.exists():raise PermissionError('fresh public figure required')
            dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=240,bbox_inches='tight');figures.append({'alias':str(dest.relative_to(public)),'chart':chart['name'],'points':len(points),'format':format})
        plt.close(fig);guard()
    out={'schema':'OB1-public-aggregate-figures-v1','status':'PASS','unit_id':active['unit_id'],'figures':figures,'sources':[chart['csv']['sha256'] for chart in cfg['charts']],'only_observed_points':True,'utility_caps':{'accuracy_loss':.010,'NLL_excess':.010,'summed_Brier_excess':.005},'interpolated_deployment_claim':False};write_json(cfg['public_output'],out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(run(a.spec)))
