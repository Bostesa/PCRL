"""Pure prospective calibration of registered P1–P7 bets, never a science gate.

Inputs are explicitly supplied native validation records (or their grid
envelope), adjusted CLAIM_RESULTS, exact ZERO_PRIVACY_CERTIFICATES, and optional
frozen SELECTION metadata. No files, fitted models or people are discovered or
read. The caller supplies accepted artifact hashes as provenance. Refutation
requires complete pertinent coverage; a complete existential witness can
establish support while other attempts remain missing, which is reported.
"""
from __future__ import annotations

import copy
import json
import math
import re

from . import config

PROBABILITIES={'P1':.85,'P2':.60,'P3':.60,'P4':.40,'P5':.65,'P6':.25,'P7':.70}
ROUTES=('utility_first','protection_first')
TASK='utility:A/same_residence'
WEIGHTS={'unweighted':'unweighted','PWGTP':'weighted'}


def assessment_rules():
    """Outcome-independent literal definitions frozen with the assessor."""
    return {'schema':1,'registered_before_outcome_assessment':True,
        'source_predictions':'PREDICTIONS.md; original subjective probabilities retained',
        'scientific_gate':False,'inference_performed':False,
        'aggregation':'exactly three anchors, equal mean of native balanced CE; caps use unweighted and PWGTP separately',
        'comparison_arithmetic':'finite Python floats, literal <, <= or >=; no rounding, comparison tolerance or outcome-conditioned choice',
        'missingness':'supported if any complete existential witness passes; refuted only with complete pertinent inputs and every case false; otherwise unassessed',
        'P1':{'probability':.85,'formula':'mean_balanced(H.validation) - mean_balanced(continuous_task.fixed_decoder) >= 0.003',
              'role':TASK,'utility_gain':.003},
        'P2':{'probability':.60,'formula':'mean_balanced(T0_U_unconstrained_a17.fixed_decoder) - mean_balanced(continuous_task.fixed_decoder) <= 0.001',
              'role':TASK,'allowed_loss':.001},
        'P3':{'probability':.60,'formula':'exists policy in {L,C}, budget in {0.0005,0.002,0.01}: mean_balanced(J.validation) - mean_balanced(T0_policy_budget_a17.validation) >= 0.003',
              'role':TASK,'utility_gain':.003,'six_common_three_anchor_configurations':True},
        'P4':{'probability':.40,'formula':'exists one common policy in {L,C} at budget0.002,a17: Trisk selected balanced task CE is strictly smaller than both same-policy T0 and Ttask, and CE(comparator)-CE(Trisk) <=0.001 for each primary role and each weighting, separately against each comparator',
              'privacy_roles':list(config.PRIMARY),'privacy_weightings':list(WEIGHTS),'allowed_sensitive_increment':.001,
              'interpretation':'same-policy triplet, direct comparator-relative measured recovery increments; no policy mixing or J-only substitute'},
        'P5':{'probability':.65,'formula':'max_{AB/SEX,AB/RAC1P}[balanced CE(H)-balanced CE(Trisk_C_0.002_a17)] < corresponding maximum for Trisk_L_0.002_a17, and balanced task CE(C)-balanced task CE(L) <=0.001',
              'allowed_task_penalty':.001,
              'interpretation':'balance the two weightings within each AB role, then maximize over the two roles; H is the same three-anchor H deployment baseline'},
        'P6':{'probability':.25,'formula':'at least one frozen route has passed the registered adjusted competitive criterion',
              'routes':list(ROUTES),'selection_metadata_required_for_refutation':True,
              'refutation':'complete scheduled programme and mandatory family attempts, then both routes fail the validation screen, have a fully measured empty feasible-comparator union, or fail their evaluated adjusted conjunction',
              'blocked_reason_rule':'unknown blocked reasons and missing attempts remain unassessed; complete validation-screen failures and complete empty comparator unions refute this programme-level bet',
              'scope':'registered common-configuration programme, not every possible encoder or a population claim'},
        'P7':{'probability':.70,'formula':'any primary zero-budget accepted map has an exact observed-support nonconstant feasible kernel certificate',
              'required_proof':'supported-state rank/nullity plus actual action count; nonconstant rational simplex/equation witness and independent CMI check',
              'coverage':'all18 primary zero-budget map/anchor systems required for refutation; one valid exact witness suffices for support',
              'scope':'finite empirical feasibility; neither a nonconstant optimized solution nor useful held-out utility is required or implied'}}


def _number(value):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
        raise ValueError('Expected finite nonnegative native CE/certificate value')
    return float(value)


def _integer(value):
    if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('Expected nonnegative integer')
    return value


def _anchor(value):
    if isinstance(value,bool) or str(value) not in ('0','1','2'):raise ValueError('Unknown or duplicate anchor schema')
    return int(value)


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('Invalid provenance hash')
    return value


def _digest(value):
    return config.digest(json.loads(json.dumps(value,allow_nan=False)))


def _pointer(*parts):return '/'+ '/'.join(str(x).replace('~','~0').replace('/','~1') for x in parts)


class _Validation:
    def __init__(self,value):
        if not isinstance(value,dict):raise ValueError('Native validation mapping required')
        envelope='records' in value
        if envelope and (value.get('phase','validation')!='validation' or value.get('evaluation_opened') is not False):
            raise ValueError('Only validation records are admissible for P1–P5')
        raw=value['records'] if envelope else value
        if not isinstance(raw,dict):raise ValueError('Validation records must be a mapping')
        self.records={};self.prefix=('records',) if envelope else ();self.receipts={};self.cache={}
        for name,anchors in raw.items():
            if not isinstance(name,str) or not isinstance(anchors,dict):raise ValueError('Invalid validation configuration')
            self.records[name]={}
            for key,native in anchors.items():
                anchor=_anchor(key)
                if anchor in self.records[name] or not isinstance(native,dict):raise ValueError('Duplicate or invalid native anchor')
                self.records[name][anchor]=(key,native)
        for receipt in value.get('receipts',[]) if envelope else []:
            key=(receipt['configuration'],_anchor(receipt['anchor']))
            if key in self.receipts:raise ValueError('Duplicate validation receipt')
            self.receipts[key]={k:_hash(receipt[k]) for k in ('summary_sha256','receipt_sha256')}

    def read(self,name,role=TASK,metric='validation'):
        key=(name,role,metric)
        if key in self.cache:return self.cache[key]
        observed=[];missing=[]
        for anchor in (0,1,2):
            item=self.records.get(name,{}).get(anchor)
            raw_anchor,native=item if item is not None else (str(anchor),{})
            entry=native.get(role,{})
            if not isinstance(entry,dict):raise ValueError('Invalid native role summary')
            scores=entry.get(metric)
            if scores is None:
                missing.append({'configuration':name,'anchor':anchor,'role':role,'metric':metric});continue
            if not isinstance(scores,dict) or not {'unweighted','weighted','balanced'}<=set(scores):
                raise ValueError('Native CE weighting schema missing')
            values={w:_number(scores[w]) for w in ('unweighted','weighted','balanced')}
            if not math.isclose(values['balanced'],.5*(values['unweighted']+values['weighted']),rel_tol=1e-12,abs_tol=1e-14):
                raise ValueError('Native balanced CE disagrees with its weightings')
            observed.append({'anchor':anchor,'values':values,
                'pointer':_pointer(*self.prefix,name,raw_anchor,role,metric),**self.receipts.get((name,anchor),{})})
        result={'configuration':name,'role':role,'metric':metric,'complete':not missing,
                'mean':{w:math.fsum(r['values'][w] for r in observed)/3 for w in ('unweighted','weighted','balanced')} if not missing else None,
                'anchors':observed,'missing':missing}
        self.cache[key]=result;return result


def _coverage(reads):
    unique={(r['configuration'],r['role'],r['metric']):r for r in reads}
    missing=[m for r in unique.values() for m in r['missing']]
    return {'complete':not missing,'required_anchor_metrics':3*len(unique),
            'available_anchor_metrics':sum(len(r['anchors']) for r in unique.values()),'missing':missing}


def _bet(name,cases,coverage):
    decisions=[c.get('passed') for c in cases]
    value=True if any(v is True for v in decisions) else False if coverage['complete'] and decisions and all(v is False for v in decisions) else None
    return {'probability':PROBABILITIES[name],'status':'supported' if value is True else 'refuted' if value is False else 'unassessed',
            'value':value,'coverage':coverage,'evidence':cases,'scientific_gate':False}


def _point_bets(v):
    output={}
    h,p,u=(v.read('H'),v.read('continuous_task',metric='fixed_decoder'),v.read('T0_U_unconstrained_a17',metric='fixed_decoder'))
    for name,left,right,field,threshold,op in (('P1',h,p,'gain',.003,'>='),('P2',u,p,'loss',.001,'<=')):
        ready=left['complete'] and right['complete'];difference=left['mean']['balanced']-right['mean']['balanced'] if ready else None
        case={field:difference,'threshold':threshold,'operator':op,'inputs':[left,right],
              'passed':(difference>=threshold if op=='>=' else difference<=threshold) if ready else None}
        output[name]=_bet(name,[case],_coverage([left,right]))
    j=v.read('J');cases=[];reads=[j]
    for policy in ('L','C'):
        for budget in config.POSITIVE_BUDGETS:
            arm=v.read(config.mechanism_id('T0',policy,budget));reads.append(arm)
            ready=j['complete'] and arm['complete'];gain=j['mean']['balanced']-arm['mean']['balanced'] if ready else None
            cases.append({'configuration':arm['configuration'],'gain':gain,'threshold':.003,'inputs':[j,arm],
                          'passed':gain>=.003 if ready else None})
    output['P3']=_bet('P3',cases,_coverage(reads));cases=[];reads=[]
    for policy in ('L','C'):
        names={code:config.mechanism_id(code,policy,.002) for code in config.INPUTS}
        utility={code:v.read(name) for code,name in names.items()}
        sensitive={code:{role:v.read(name,'attack:'+role) for role in config.PRIMARY} for code,name in names.items()}
        inputs=list(utility.values())+[r for roles in sensitive.values() for r in roles.values()];reads+=inputs
        ready=all(r['complete'] for r in inputs);case={'policy':policy,'inputs':inputs,'passed':None}
        if ready:
            task={code:r['mean']['balanced'] for code,r in utility.items()}
            increments={code:{role:{w:sensitive[code][role]['mean'][field]-sensitive['Trisk'][role]['mean'][field]
                for w,field in WEIGHTS.items()} for role in config.PRIMARY} for code in ('T0','Ttask')}
            case.update(balanced_task_ce=task,sensitive_increments=increments,
                passed=bool(all(task['Trisk']<task[c] for c in ('T0','Ttask')) and
                            all(x<=.001 for roles in increments.values() for weights in roles.values() for x in weights.values())))
        cases.append(case)
    output['P4']=_bet('P4',cases,_coverage(reads))
    names={'L':config.mechanism_id('Trisk','L',.002),'C':config.mechanism_id('Trisk','C',.002),'H':'H'}
    utility={policy:v.read(names[policy]) for policy in ('L','C')}
    sensitive={policy:{role:v.read(name,'attack:'+role) for role in ('AB/SEX','AB/RAC1P')} for policy,name in names.items()}
    reads=list(utility.values())+[r for rows in sensitive.values() for r in rows.values()]
    ready=all(r['complete'] for r in reads);case={'inputs':reads,'passed':None}
    if ready:
        recovery={policy:{role:sensitive['H'][role]['mean']['balanced']-sensitive[policy][role]['mean']['balanced']
                    for role in ('AB/SEX','AB/RAC1P')} for policy in ('L','C')}
        local,coalition=(max(recovery[policy].values()) for policy in ('L','C'))
        penalty=utility['C']['mean']['balanced']-utility['L']['mean']['balanced']
        case.update(balanced_AB_recovery=recovery,local_max_recovery=local,coalition_max_recovery=coalition,
                    task_penalty=penalty,passed=bool(coalition<local and penalty<=.001))
    output['P5']=_bet('P5',[case],_coverage(reads));return output


def _formula(node,checks):
    if not isinstance(node,dict) or type(node.get('passed')) is not bool:raise ValueError('Malformed adjusted claim formula')
    if 'endpoint' in node:
        value=checks.get(node['endpoint'],{}).get(node.get('check'))
        if type(value) is not bool:raise ValueError('Adjusted formula check is missing')
    elif node.get('kind')=='blocked':value=False
    elif node.get('kind') in ('all','any') and isinstance(node.get('clauses'),list) and node['clauses']:
        children=[_formula(child,checks) for child in node['clauses']]
        value=all(children) if node['kind']=='all' else any(children)
    else:raise ValueError('Unknown adjusted claim formula')
    if node['passed']!=value:raise ValueError('Adjusted formula boolean disagrees with its checks')
    return value


def _programme_coverage(selection):
    if selection is None:return {'complete':False,'missing':[{'kind':'frozen_selection_metadata'}]}
    if selection.get('selection_frozen') is not True:raise ValueError('P6 requires frozen selection metadata')
    completion=selection.get('completion_at_freeze')
    if not isinstance(completion,dict) or not isinstance(completion.get('records'),list) or not isinstance(completion.get('missing'),list):
        return {'complete':False,'missing':[{'kind':'programme_completion_metadata'}]}
    records={}
    for r in completion['records']:
        key=(r['configuration'],_anchor(r['anchor']))
        if key in records or type(r.get('complete')) is not bool:raise ValueError('Invalid programme completion record')
        records[key]=r
    declared={(r['configuration'],_anchor(r['anchor'])) for r in completion['missing']}
    actual={key for key,r in records.items() if not r['complete']}
    if declared!=actual:raise ValueError('Programme missing-unit list disagrees with completion records')
    required={(r['configuration'],r['anchor']) for r in config.release_ledger()['records']}
    for name,spec in selection.get('registered_controls',{}).items():
        if spec.get('required') is True:required.update((name,a) for a in (0,1,2))
    missing=[{'configuration':name,'anchor':anchor,'kind':'incomplete_or_unrecorded_scheduled_unit'}
             for name,anchor in sorted(actual | (required-set(records)))]
    for route,r in selection.get('routes',{}).items():
        missing.extend({'route':route,'family':name,'kind':'incomplete_required_family'} for name in r.get('incomplete_required_families',[]))
    return {'complete':not missing,'required_programme_units':len(required|set(records)),
            'recorded_programme_units':len(records),'missing':missing}


def _competitive(claims,selection,provenance):
    coverage=_programme_coverage(selection);cases=[]
    if claims is not None:
        if not isinstance(claims,dict) or not isinstance(claims.get('claim_results'),dict):raise ValueError('Native CLAIM_RESULTS required')
        pins=claims.get('provenance',{})
        for field in ('selection_sha256','contrasts_sha256'):_hash(pins.get(field))
        for filename,field in (('SELECTION.json','selection_sha256'),('CONTRASTS.json','contrasts_sha256')):
            if filename in provenance and provenance[filename]!=pins[field]:raise ValueError('Claim provenance differs from supplied accepted artifacts')
        if selection is not None and selection.get('contrasts_sha256') not in (None,pins['contrasts_sha256']):
            raise ValueError('Claim provenance differs from frozen contrast hash')
        if selection is not None:
            for field in ('config_hash','source_hashes','inference_source_hashes'):
                if field in selection and pins.get(field)!=selection[field]:
                    raise ValueError('Claim source/configuration provenance differs from frozen selection')
        checks=claims.get('checks',{})
        if not isinstance(checks,dict) or _integer(claims.get('family_size'))!=len(checks):raise ValueError('Adjusted endpoint family is incomplete')
        if selection is not None and selection.get('contrast_family_size') not in (None,claims['family_size']):
            raise ValueError('Adjusted endpoint family differs from selection')
    else:checks={}
    for route in ROUTES:
        selected={} if selection is None else selection.get('routes',{}).get(route,{})
        node=None if claims is None else claims['claim_results'].get(route,{}).get('competitive')
        value=None;reason='adjusted competitive result or route metadata unavailable'
        if node is not None:
            passed=_formula(node,checks)
            if passed:
                if selected and (selected.get('screen_passed') is not True or selected.get('incomplete_required_families')):
                    raise ValueError('Passed competitive formula contradicts frozen route eligibility')
                value=True;reason='registered adjusted competitive criterion passed'
            elif node.get('kind')!='blocked':value=False;reason='registered adjusted competitive conjunction failed'
            elif node.get('reason')=='validation screen did not pass; descriptive nominee only':
                if selected.get('screen_passed') is False:value=False;reason='complete-programme validation screen failure'
            elif node.get('reason')=='mandatory comparison attempts incomplete or total feasible comparator set empty':
                if (selected and not selected.get('incomplete_required_families')
                        and selected.get('eligible_comparator_families')==[]
                        and node.get('incomplete_required_families')==[] and node.get('eligible_comparator_families')==[]):
                    value=False;reason='fully measured feasible comparator union is empty'
        elif selected.get('screen_passed') is False:
            value=False;reason='complete-programme validation screen failure; adjusted comparisons cannot make this route eligible'
        elif (selected and selected.get('competitive_validation_eligible') is False
              and selected.get('incomplete_required_families')==[] and selected.get('eligible_comparator_families')==[]):
            value=False;reason='fully measured feasible comparator union is empty'
        cases.append({'route':route,'passed':value,'reason':reason,
            'nominee':selected.get('nominee'),'screen_passed':selected.get('screen_passed'),
            'incomplete_required_families':copy.deepcopy(selected.get('incomplete_required_families')),
            'eligible_comparator_families':copy.deepcopy(selected.get('eligible_comparator_families')),
            'adjusted_formula':copy.deepcopy(node)})
    return _bet('P6',cases,coverage)


def _perfect_privacy(value):
    expected={(s['configuration'],s['anchor']) for s in config.configuration()['maps'] if s['budget']==0}
    records={} if value is None else value.get('certificates')
    if value is not None and not isinstance(records,list):raise ValueError('Exact certificate list required')
    found={};cases=[]
    for r in [] if value is None else records:
        key=(r['configuration'],_anchor(r['anchor']))
        if key not in expected or key in found:raise ValueError('Unregistered or duplicate exact certificate')
        n,rank,nullity,actions,zero=(_integer(r.get(k)) for k in ('supported_states','exact_rank','exact_nullity','action_count','zero_action'))
        if not n or not actions or zero>=actions or rank+nullity!=n or nullity<1:
            raise ValueError('Invalid exact certificate support/action dimensions')
        positive=nullity>1 and actions>1
        if (type(r.get('nonconstant_supported_kernel_exists')) is not bool or r['nonconstant_supported_kernel_exists']!=positive
                or r.get('constant_only_on_supported_states') is not (not positive)
                or r.get('arithmetic')!='integer elimination and rational witness verification'):
            raise ValueError('Exact certificate predicate disagrees with supported rank/action schema')
        _hash(r.get('map_receipt_sha256'))
        if positive:
            if (r.get('rational_equations_verified') is not True or r.get('rational_simplex_verified') is not True
                    or _number(r.get('supported_row_range'))<=0 or _number(r.get('maximum_independent_float_cmi'))>1e-10):
                raise ValueError('Nonconstant certificate lacks its exact/independent feasibility proof')
            _hash(r.get('private_witness_sha256'))
        evidence={k:copy.deepcopy(r[k]) for k in ('configuration','anchor','supported_states','unsupported_states',
            'exact_rank','exact_nullity','action_count','zero_action','map_receipt_sha256','private_witness_sha256',
            'maximum_independent_float_cmi','supported_row_range','rational_equations_verified','rational_simplex_verified') if k in r}
        cases.append({**evidence,'passed':positive});found[key]=r
    missing=[{'configuration':name,'anchor':anchor} for name,anchor in sorted(expected-set(found))]
    coverage={'complete':not missing,'required_zero_budget_systems':len(expected),'certified_systems':len(found),'missing':missing}
    return _bet('P7',cases,coverage)


def assess_predictions(validation, *,claim_results=None,zero_certificates=None,selection=None,provenance=None):
    """Assess bets from explicit accepted aggregates; never infer missing outcomes.

    ``validation`` accepts VALIDATION_GRID or its native ``records`` mapping.
    ``selection`` supplies completion and eligibility metadata needed for a P6
    refutation. ``provenance`` maps accepted input filenames to SHA-256 hashes;
    these are declarations, preserved and cross-checked without file access.
    Canonical content digests also bind every supplied input independently of
    a caller's original JSON serialization. No new inference test is run.
    """
    provenance={} if provenance is None else copy.deepcopy(provenance)
    if not isinstance(provenance,dict):raise ValueError('Provenance must be an artifact hash mapping')
    for name,value in provenance.items():
        if not isinstance(name,str):raise ValueError('Invalid provenance artifact name')
        _hash(value)
    v=_Validation(validation);predictions=_point_bets(v)
    predictions['P6']=_competitive(claim_results,selection,provenance)
    predictions['P7']=_perfect_privacy(zero_certificates)
    inputs={'validation':validation,'claim_results':claim_results,'zero_certificates':zero_certificates,'selection':selection}
    return {'schema':1,'scope':'calibration of prospective subjective bets in this registered empirical programme only',
            'scientific_gate':False,'inference_performed':False,'rules_digest':config.digest(assessment_rules()),
            'provenance':provenance,'input_content_digests':{k:_digest(x) for k,x in inputs.items() if x is not None},
            'provenance_scope':'caller-supplied accepted artifact hashes; this pure assessor opens no files',
            'predictions':predictions}
