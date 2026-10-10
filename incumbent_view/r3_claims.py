"""Explicit A/B/C accounting from fixed-family intervals, no endpoint fitting."""
from .r3_family import ARMS, TASKS, REFERENCES, METRICS, oriented_recovery_slot

CLAIM_C_CONTROLS=("V_P","V_D","V_L","V_S12","V_S21")


def oriented_interval(slots,candidate,control,view,benefit=True):
    slot,sign=oriented_recovery_slot(candidate,control,view)
    item=slots.get(slot)
    if item is None or item.get('point') is None or item.get('lower') is None or item.get('upper') is None:
        return {'slot':slot,'status':'INCOMPLETE','point':None,'lower':None,'upper':None}
    if not benefit:sign=-sign
    return {'slot':slot,'status':'AVAILABLE','point':sign*item['point'],
            'lower':item['lower'] if sign==1 else -item['upper'],
            'upper':item['upper'] if sign==1 else -item['lower']}


def utility_components(slots,arm):
    out=[]
    for seed in (0,1):
        for task in TASKS:
            for reference in REFERENCES:
                for metric in METRICS:
                    slot=f's{seed}:utility:{arm}:{task}:{reference}:{metric}'
                    item=slots.get(slot)
                    out.append({'slot':slot,'status':('INCOMPLETE' if item is None or item.get('point') is None else
                                                     'PASS' if item.get('status')=='PASS' else 'NOT_ESTABLISHED'),
                                'lower':None if item is None else item.get('lower'),
                                'upper':None if item is None else item.get('upper')})
    return out


def recovery_components(slots,candidate,control,threshold,strict=False):
    pair=oriented_interval(slots,candidate,control,'pair')
    pair['criterion']='pair_benefit';pair['required_lower']=threshold;pair['strict']=strict
    if pair['status']=='AVAILABLE':pair['status']='PASS' if (pair['lower']>threshold if strict else pair['lower']>=threshold) else 'NOT_ESTABLISHED'
    out=[pair]
    for reference in dict.fromkeys(('P0',control)):
        if reference==candidate:continue
        for view in ('r1','r2'):
            local=oriented_interval(slots,candidate,reference,view,benefit=False)
            local.update(criterion='local_worsening',reference=reference,view=view,required_upper=.010)
            if local['status']=='AVAILABLE':local['status']='PASS' if local['upper']<=.010 else 'NOT_ESTABLISHED'
            out.append(local)
    return out


def aggregate_status(components):
    if any(x['status']=='INCOMPLETE' for x in components):return 'INCOMPLETE'
    return 'PASSED' if all(x['status']=='PASS' for x in components) else 'NOT_ESTABLISHED'


def claim(candidate,controls,gates,slots,threshold,strict=False):
    if candidate is None or candidate not in ARMS:
        return {'status':'UNTESTED','reason':'no prospectively nominated nonidentity candidate','components':[]}
    required=tuple(dict.fromkeys((candidate,*controls)))
    missing=[a for a in required if gates.get(a,{}).get('all_two_seed_task_reference_point_caps_pass') is not True]
    if missing:return {'status':'UNTESTED','reason':'required original point utility gate did not pass or is absent','arms':missing,'components':[]}
    components=[x for arm in required for x in utility_components(slots,arm)]
    components += [x for control in controls for x in recovery_components(slots,candidate,control,threshold,strict)]
    return {'status':aggregate_status(components),'candidate':candidate,'controls':list(controls),'components':components}


def all_claims(nominee,gates,slots,*,nominee_exact_p0_alias=False):
    a=({'status':'UNTESTED','reason':'both-seed exact P0 alias supplies no new method gain','components':[]}
       if nominee_exact_p0_alias else claim(nominee,('P0',),gates,slots,.020))
    b=claim('V_J',('O_J',),gates,slots,0.,strict=True)
    missing_seq=[a for a in ('V_S12','V_S21') if gates.get(a,{}).get('all_two_seed_task_reference_point_caps_pass') is not True]
    eligible=tuple(a for a in CLAIM_C_CONTROLS if gates.get(a,{}).get('all_two_seed_task_reference_point_caps_pass') is True)
    c=({'status':'UNTESTED','reason':'required sequential comparison unavailable or point-ineligible','arms':missing_seq,'components':[]}
       if missing_seq else claim('V_J',eligible,gates,slots,.020))
    return {'schema':'pcrl-incumbent-r3-three-claims-v1','A':a,'B':b,'C':c,
            'interpretation':'Conditional exploratory inference with fixed K948; component passes do not imply independent confirmation.'}
