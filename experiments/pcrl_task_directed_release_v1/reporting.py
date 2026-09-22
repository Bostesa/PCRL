"""Machine-recomputed aggregates and paired inference from frozen private losses."""
from __future__ import annotations
import argparse
import copy
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from .config import OUT,configuration,release_ledger,digest
from .run import atomic,sha,now
from . import run


def _safe(role):return role.replace(':','__').replace('/','__')


def _validate_rows(rows,n_classes=None):
    required=('ids','households','weights','y','loss')
    if not all(k in rows for k in required):raise ValueError('Incomplete expected-loss artifact')
    n=len(rows['ids'])
    if not n or any(np.asarray(rows[k]).shape!=(n,) for k in required):raise ValueError('Expected-loss rows must align')
    for field in ('ids','households'):
        if any(not isinstance(v,(str,np.str_)) or not v.strip() for v in rows[field]):
            raise ValueError('Missing or non-string identity in private loss artifact')
    if len(np.unique(rows['ids']))!=n:raise ValueError('Person IDs must be unique')
    if not np.isfinite(rows['loss']).all() or np.any(rows['loss']<0):raise ValueError('Invalid per-person log loss')
    if not np.isfinite(rows['weights']).all() or np.any(rows['weights']<=0):raise ValueError('Invalid survey weights')
    if not np.isfinite(rows['y']).all() or np.any(rows['y']<0) or np.any(rows['y']!=np.floor(rows['y'])):
        raise ValueError('Invalid labels in accepted expected-loss artifact')
    if n_classes is not None and np.any(rows['y']>=n_classes):
        raise ValueError('Labels exceed the registered target class schema')


def _target(role):
    cfg=configuration()['roles']
    permitted={'attack:'+r for r in (*cfg['primary'],*cfg['secondary'])}|{'utility:'+r for r in cfg['utility']}
    if role not in permitted:raise ValueError('Unknown registered target role')
    return role.split('/')[1]


def paired_endpoint(endpoint,loader):
    """Preserve person pairing and exact zero differences; never pool anchors."""
    if endpoint['anchors']!=[0,1,2]:raise ValueError('Exactly the three registered anchors are required')
    if endpoint['weighting'] not in ('unweighted','PWGTP'):raise ValueError('Unknown reporting weighting')
    terms=endpoint['terms']
    if not terms:raise ValueError('An endpoint needs explicit oriented loss terms')
    targets={_target(term['role']) for term in terms}
    if len(targets)!=1:raise ValueError('A paired endpoint cannot mix target/class definitions')
    n_classes=9 if next(iter(targets))=='RAC1P' else 2
    records=[]
    for anchor in endpoint['anchors']:
        reference=None;difference=None
        for term in terms:
            if term.get('predictor','frozen_validation_selection')!='frozen_validation_selection':
                raise ValueError('Primary contrasts must use frozen deployment selections')
            rows=loader(anchor,term['configuration'],term['role']);_validate_rows(rows,n_classes)
            if reference is None:
                reference=rows;difference=np.zeros(len(rows['ids']),dtype=np.float64)
            elif any(not np.array_equal(reference[k],rows[k]) for k in ('ids','households','weights','y')):
                raise ValueError('Paired endpoint identity/label/weight alignment mismatch')
            coefficient=float(term['coefficient'])
            if not math.isfinite(coefficient):raise ValueError('Nonfinite contrast coefficient')
            difference+=coefficient*np.asarray(rows['loss'],np.float64)
        records.append({'household':np.asarray(reference['households'],str),'difference':difference,
                        'weights':np.ones(len(difference)) if endpoint['weighting']=='unweighted' else np.asarray(reference['weights'],float)})
    return {'id':endpoint['id'],'anchors':records}


def check_bound(interval,check):
    bound=check['bound'];op=check['operator'];threshold=float(check['threshold'])
    if bound not in ('lower','upper'):raise ValueError('Check requires a declared interval bound')
    value=float(interval[bound])
    if not math.isfinite(value) or not math.isfinite(threshold):raise ValueError('Nonfinite prospective bound check')
    if op=='<':return value<threshold
    if op=='<=':return value<=threshold
    if op=='>':return value>threshold
    if op=='>=':return value>=threshold
    raise ValueError('Unknown prospective bound operator')


def evaluate_formula(formula,checks):
    if 'endpoint' in formula:
        key=(formula['endpoint'],formula['check'])
        if key not in checks:raise ValueError('Claim references a missing endpoint check')
        return {**formula,'passed':bool(checks[key])}
    kind=formula['kind']
    if kind=='blocked':return {**formula,'passed':False}
    if kind not in ('all','any') or not formula.get('clauses'):raise ValueError('A claim needs a nonempty explicit conjunction/disjunction')
    clauses=[evaluate_formula(c,checks) for c in formula['clauses']]
    passed=all(c['passed'] for c in clauses) if kind=='all' else any(c['passed'] for c in clauses)
    return {'kind':kind,'clauses':clauses,'passed':passed}


def evaluate_claims(contrasts,results):
    """Evaluate main, attribution and separately named stricter diagnostic trees."""
    endpoints=contrasts['endpoints'];ids=[e['id'] for e in endpoints]
    if (len(ids)!=len(set(ids)) or len(ids)!=contrasts['family_size']
            or results['family_size']!=len(ids) or set(results['bounds'])!=set(ids)):
        raise ValueError('Adjusted bounds do not match the explicit prospective endpoint family')
    flat={};records={}
    for endpoint in endpoints:
        names=[check['name'] for check in endpoint['checks']]
        if len(set(names))!=len(names):raise ValueError('Duplicate prospective endpoint check name')
        records[endpoint['id']]={}
        for check in endpoint['checks']:
            value=check_bound(results['bounds'][endpoint['id']],check)
            flat[(endpoint['id'],check['name'])]=value
            records[endpoint['id']][check['name']]=value
    def formulas(name):
        return {route:{claim:evaluate_formula(formula,flat) for claim,formula in group.items()}
                for route,group in contrasts.get(name,{}).items()}
    return {'family_size':len(ids),'checks':records,
            'claim_results':formulas('claim_formulas'),
            'attribution_claim_results':formulas('attribution_claim_formulas'),
            'diagnostic_claim_results':formulas('diagnostic_claim_formulas'),
            'scope':'frozen prospective formulas, simultaneous adjusted interval bounds; both routes retained'}


def inference_source_fingerprint():
    return {name:sha(Path(__file__).with_name(name+'.py')) for name in ('selection','uncertainty','reporting')}


def bind_frozen_provenance(selection):
    """Before freeze_selection: pin accepted audit weights/receipts and source closure.

    Reads validation summaries and hashes fitted artifacts, never deserializes a
    model or reads evaluation losses. Caller must close scientific writes first.
    """
    if not selection.get('validation_only') or selection.get('selection_frozen'):
        raise ValueError('Bind provenance to an unfrozen validation-only selection')
    cfg_hash=digest(configuration())
    if selection.get('configuration_digest')!=cfg_hash:raise ValueError('Selection configuration changed before freeze')
    result=copy.deepcopy(selection);pins={}
    for name in selection['descriptive_configurations']:
        if not re.fullmatch('[A-Za-z0-9_.-]+',name):raise ValueError('Invalid configuration path')
        pins[name]={};summaries=[]
        for anchor in (0,1,2):
            base=OUT/'private/run'/f'anchor_{anchor}'/'audits'/name
            marker=base/'COMPLETE.json'
            receipt=run._read_marker(marker,'audit',anchor=anchor,name=name)
            registry_hash=sha(base/'registry.joblib')
            if receipt.get('registry_sha256')!=registry_hash:raise ValueError('Accepted audit registry changed')
            summary=json.loads((base/'summary.json').read_text());summaries.append(summary)
            expected=selection['predictor_choices'][name][str(anchor)]
            if set(summary)!=set(expected):raise ValueError('Frozen predictor role set differs from accepted audit')
            for role,choice in expected.items():
                if any(choice.get(k)!=summary[role].get(k) for k in ('selection','independent_selection')):
                    raise ValueError('Chosen validation predictor differs from accepted audit')
            pins[name][str(anchor)]={'registry_sha256':registry_hash,'receipt_sha256':sha(marker)}
        for role,weights in selection['aggregates'][name]['roles'].items():
            for weighting,field in (('unweighted','unweighted'),('PWGTP','weighted')):
                actual=math.fsum(float(s[role]['validation'][field]) for s in summaries)/3
                if actual!=weights[weighting]:raise ValueError('Validation means changed before provenance freeze')
    result.update(frozen_audits=pins,config_hash=cfg_hash,source_hashes=run.source_fingerprint(),
                  inference_source_hashes=inference_source_fingerprint())
    return result


def _frozen_selection():
    path=OUT/'SELECTION.json';selection=json.loads(path.read_text())
    if selection.get('selection_frozen') is not True:raise RuntimeError('No frozen selection')
    cfg_hash=digest(configuration())
    if selection.get('configuration_digest')!=cfg_hash or selection.get('config_hash')!=cfg_hash:
        raise ValueError('Frozen selection configuration changed or missing')
    if selection.get('source_hashes')!=run.source_fingerprint():raise ValueError('Frozen scientific source provenance changed')
    if selection.get('inference_source_hashes')!=inference_source_fingerprint():raise ValueError('Frozen inference source provenance changed')
    if not isinstance(selection.get('frozen_audits'),dict):raise ValueError('Selection lacks frozen audit registry hashes')
    return selection,sha(path)


def collect_validation():
    schedule=json.loads((OUT/'RESOURCE_SCHEDULE.json').read_text())
    if not schedule.get('frozen_before_comparative_outcomes'):raise RuntimeError('Comparative scores stay closed until resource schedule is frozen')
    if schedule['config_hash']!=digest(configuration()):raise ValueError('Registered configuration changed')
    validation={};receipts=[]
    for anchor in (0,1,2):
        base=OUT/'private/run'/f'anchor_{anchor}'/'audits'
        for marker in sorted(base.glob('*/COMPLETE.json')):
            record=json.loads(marker.read_text());summary=marker.parent/'summary.json'
            if record['config_hash']!=schedule['config_hash']:raise ValueError('Audit uses a different scientific configuration')
            if sha(summary)!=record['artifact_hashes']['summary.json']:raise ValueError('Validation summary hash mismatch')
            name=record['configuration']
            validation.setdefault(name,{})[anchor]=json.loads(summary.read_text())
            receipts.append({'anchor':anchor,'configuration':name,'receipt_sha256':sha(marker),
                             'summary_sha256':sha(summary),'new_role_fits':record['new_role_fits'],
                             'reused_role_audits':record['reused_role_audits'],'seconds':record['seconds']})
    atomic(OUT/'VALIDATION_GRID.json',{'written_utc':now(),'scope':'validation only; historically used2018 development',
           'records':validation,'receipts':receipts,'evaluation_opened':False})
    return validation


def frozen_loss_loader():
    selection,selection_hash=_frozen_selection()
    cache={};receipts={}
    def load(anchor,name,role):
        if anchor not in (0,1,2) or isinstance(anchor,bool):raise ValueError('Unknown evaluation anchor')
        target=_target(role)
        key=(anchor,name,role)
        if key not in cache:
            if not re.fullmatch('[A-Za-z0-9_.-]+',name):raise ValueError('Invalid configuration path')
            if name not in selection['evaluation_configurations']:raise ValueError('Configuration was not frozen for evaluation')
            base=OUT/'private/run'/f'anchor_{anchor}'/'evaluation'/name
            pair=(anchor,name)
            if pair not in receipts:
                if sha(OUT/'SELECTION.json')!=selection_hash:raise ValueError('Selection changed during inference')
                pin=selection['frozen_audits'].get(name,{}).get(str(anchor))
                if not pin:raise ValueError('Evaluation lacks a selection-time audit registry pin')
                audit=OUT/'private/run'/f'anchor_{anchor}'/'audits'/name
                if sha(audit/'COMPLETE.json')!=pin['receipt_sha256'] or sha(audit/'registry.joblib')!=pin['registry_sha256']:
                    raise ValueError('Frozen audit receipt/registry hash changed')
                accepted=json.loads((audit/'COMPLETE.json').read_text())
                run._validate_provenance(accepted,'audit')
                if (accepted.get('anchor')!=anchor or accepted.get('configuration')!=name
                        or accepted.get('registry_sha256')!=pin['registry_sha256']):
                    raise ValueError('Frozen audit identity changed')
                marker=json.loads((base/'COMPLETE.json').read_text())
                run._validate_provenance(marker,'evaluation')
                if (marker.get('anchor')!=anchor or marker.get('configuration')!=name
                        or marker.get('selection_sha256')!=selection_hash
                        or marker.get('registry_sha256')!=pin['registry_sha256']):
                    raise ValueError('Evaluation receipt does not match frozen selection/registry')
                receipts[pair]=marker
            marker=receipts[pair]
            path=base/'test'/(_safe(role)+'.npz')
            relative=str(path.relative_to(base))
            if sha(path)!=marker['artifact_hashes'][relative]:raise ValueError('Evaluation loss artifact hash mismatch')
            with np.load(path,allow_pickle=False) as f:
                cache[key]={k:f[k].copy() for k in ('ids','households','weights','y','loss')}
            _validate_rows(cache[key],9 if target=='RAC1P' else 2)
        return cache[key]
    return load


def infer():
    from .uncertainty import paired_household_bounds
    selection,selection_hash=_frozen_selection()
    if sha(OUT/'CONTRASTS.json')!=selection['contrasts_sha256']:raise ValueError('Prospective contrast family changed')
    contrasts=json.loads((OUT/'CONTRASTS.json').read_text())
    if (len(contrasts['endpoints'])!=contrasts['family_size']
            or selection.get('contrast_family_size')!=contrasts['family_size']):raise ValueError('Endpoint count mismatch')
    loader=frozen_loss_loader();paired=[paired_endpoint(e,loader) for e in contrasts['endpoints']]
    cfg=configuration()['inference']
    results=paired_household_bounds(paired,n_boot=cfg['bootstrap_replicates'],seed=cfg['seed'],alpha=cfg['alpha'])
    if sha(OUT/'SELECTION.json')!=selection_hash:raise ValueError('Selection changed during inference')
    provenance={'selection_sha256':selection_hash,'contrasts_sha256':selection['contrasts_sha256'],
                'config_hash':selection['config_hash'],'source_hashes':selection['source_hashes'],
                'inference_source_hashes':selection['inference_source_hashes']}
    results['provenance']=provenance
    claims={**evaluate_claims(contrasts,results),'provenance':provenance}
    atomic(OUT/'PAIRED_BOUNDS.json',results)
    atomic(OUT/'CLAIM_RESULTS.json',claims)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('validation','inference'))
    args=parser.parse_args()
    if args.command=='validation':collect_validation()
    else:infer()
