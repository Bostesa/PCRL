from scripts import summarize_acs_restricted as report

def row(release,role='transfer',target='same_residence',loss=.5):
    value=dict(seed=0,role=role,release=release,target=target,candidate_id='logistic',family='logistic',selected=True,independent_selected=True,selected_within_family=True,auc_selected=False)
    value.update({k:{'log_loss':loss} for k in report.SPLITS})
    return value



def test_fixed_comparisons_include_all_tasks_both_weights_and_scopes():
    pairs=report.comparisons();names={n for _,a,b in pairs for n in (a,b)}
    values={name:.1+i/1000 for i,name in enumerate(sorted(names))}
    raw=[]
    for name in names:
        for task in report.UTILITY_TASKS:raw.append(row(name,target=task,loss=values[name]))
        if name not in report.STAGES:
            for target in report.ATTRIBUTES:raw.append(row(name,'audit',target,loss=values[name]))
    index=report.shared.index_records([{'seed':0,'raw_metrics':raw}])
    result=report.paired([0],{120:index,360:index})
    assert len([r for r in result if r['role']=='transfer'])==len(pairs)*5*4
    assert {r['comparison'] for r in result}=={p[0] for p in pairs}
    for r in result:
        assert r['left_minus_right']==values[r['left']]-values[r['right']]
        if r['role']=='audit':assert r['signed_gain_left_minus_right']==-r['left_minus_right']
    assert sum(x[0]=='K_minus_F' for x in pairs)==4
    assert sum(x[0]=='E_minus_S' for x in pairs)==4
    assert not any(r['role']=='audit' and (r['left'] in report.STAGES or r['right'] in report.STAGES) for r in result)


def test_native_bank_alias_and_undefined_seed_mean():
    assert report.alias('E_bank','B')=='E_bank'
    assert report.alias('E_K','C')=='E_K_C'
    assert report.statistics([.2,None,.3])['mean'] is None
    assert report.table(['A'],[['B']]).startswith('| A |')
