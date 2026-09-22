"""Exact saved-channel equivalence; no fitting, rounding or row-cache access.

Literal groups require identical input family, anchor, frozen encoder identity,
ordered action offsets and Q entries. Reduced groups additionally permit an
input-independent row, or exact equal child pairs under the registered t//2
parent relation. Every row is checked, including states absent from fitting.
Different noncollapsed task/risk codes never merge merely because Q arrays or
observed predictions coincide. Equality is numeric (signed zeros are equal).

Counts concern stored mechanisms, not independent observations or evidence.
The collector opens JSON receipts/metadata, encoder.joblib, and Q.npz only;
solution/prepared hashes are receipt declarations, not deserialized objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np

from . import config, run
from .finite import ACCEPTANCE_TOLERANCES


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):
        raise ValueError('Invalid SHA-256 identity')
    return value


def _name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',value):
        raise ValueError('Invalid registered map name')
    return value


def _index(value):
    if isinstance(value,bool):raise ValueError('Integer schema required')
    try:return operator.index(value)
    except TypeError as exc:raise ValueError('Integer schema required') from exc


def _array(value,ndim):
    raw=np.asarray(value)
    if raw.ndim!=ndim or raw.dtype.kind not in 'biuf' or not np.isfinite(raw).all():
        raise ValueError('Finite numeric array required')
    if ((raw.dtype.kind=='f' and raw.dtype.itemsize>8)
            or (raw.dtype.kind in 'iu' and (np.any(raw>2**53) or np.any(raw<-(2**53))))):
        raise ValueError('Conversion could round stored values')
    a=np.asarray(raw,dtype='<f8')
    if not np.array_equal(raw,a):raise ValueError('Conversion would round stored values')
    a=a.copy();a[a==0]=0.  # Exact numeric equality treats both signed zeros alike.
    return a


def _array_hash(a):
    h=hashlib.sha256();h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()


def _record(record):
    name=_name(record['configuration']);anchor=_index(record['anchor'])
    if anchor not in (0,1,2):raise ValueError('Unknown anchor')
    family=record['input'];coarse=_index(record['coarse_states'])
    if family not in config.INPUTS or not 1<=coarse<=32:raise ValueError('Invalid registered input schema')
    q=_array(record['Q'],2);offsets=_array(record['offsets'],1)
    rows=coarse*(1 if family=='T0' else 2)
    if q.shape!=(rows,len(offsets)) or not len(offsets):raise ValueError('Channel/action/input shape mismatch')
    if np.any(q<0) or np.any(q>1) or np.max(np.abs(q.sum(1)-1))>ACCEPTANCE_TOLERANCES['simplex']:
        raise ValueError('Invalid accepted stochastic channel')
    encoder=_hash(record['encoder_sha256'])
    public={k:record[k] for k in ('solver_status','solver') if k in record}
    public.update(configuration=name,anchor=anchor,input=family,coarse_states=coarse,
                  action_count=len(offsets),encoder_sha256=encoder,
                  action_dictionary_sha256=_array_hash(offsets),stored_kernel_sha256=_array_hash(q))
    for k in ('receipt_sha256','Q_artifact_sha256','solution_sha256','prepared_receipt_sha256'):
        if k in record:public[k]=_hash(record[k])
    common={'anchor':anchor,'encoder_sha256':encoder,'action_dictionary_sha256':_array_hash(offsets)}
    literal={**common,'canonical_domain':family,'kernel_sha256':_array_hash(q),'shape':list(q.shape)}
    if np.array_equal(q,np.repeat(q[:1],len(q),axis=0)):
        domain,reduced,basis='constant',q[:1],'all stored rows exactly equal'
    elif family!='T0' and np.array_equal(q[0::2],q[1::2]):
        domain,reduced,basis='T0',q[0::2],'every child pair exactly equal under registered floor(t/2)'
    else:
        domain,reduced,basis=family,q,'literal registered input kernel'
    reduced_identity={**common,'canonical_domain':domain,'kernel_sha256':_array_hash(reduced),
                      'shape':list(reduced.shape)}
    return {'public':public,'source':record,'Q':q,'offsets':offsets,
            'literal':literal,'reduced':reduced_identity,'basis':basis}


def group_maps(records, *, nominal_specs=None):
    """Pure exact grouping of already verified map records; no source discovery.

    Each input record provides Q, ordered offsets, input, coarse_states, anchor,
    configuration and frozen encoder_sha256. Optional receipt/solver fields
    establish provenance and distinguish declared fallback from exact copying.
    No kernel arrays or offsets are returned in the public report.
    """
    normalized=[];by_unit={}
    for raw in records:
        r=_record(raw);key=(r['public']['configuration'],r['public']['anchor'])
        if key in by_unit:raise ValueError('Duplicate map/anchor record')
        by_unit[key]=r;normalized.append(r)
    groups={'literal':{},'reduced':{}}
    for r in normalized:
        for mode in groups:
            identity=r[mode];digest=config.digest(identity)
            group=groups[mode].setdefault(digest,{'group_sha256':digest,**identity,'members':[],
                                                 'exact_equality_verified':True})
            # Hash keys index candidates; compare exact arrays to the first member
            # as well, so a collision never establishes an equivalence assertion.
            peers=group['members']
            if peers:
                peer=by_unit[(peers[0]['configuration'],peers[0]['anchor'])]
                def canonical(item):
                    if mode=='literal':return item['Q']
                    domain=item['reduced']['canonical_domain']
                    if domain=='constant':return item['Q'][:1]
                    if domain=='T0' and item['public']['input']!='T0':return item['Q'][::2]
                    return item['Q']
                if not (np.array_equal(canonical(peer),canonical(r)) and np.array_equal(peer['offsets'],r['offsets'])):
                    raise ValueError('Identity hash collision without exact array equality')
            peers.append({'configuration':r['public']['configuration'],'anchor':r['public']['anchor'],
                          'reduction_basis':r['basis'] if mode=='reduced' else 'literal same-family equality'})
        r['public']['literal_group_sha256']=config.digest(r['literal'])
        r['public']['reduced_group_sha256']=config.digest(r['reduced'])
        r['public']['reduction_basis']=r['basis']
        source=r['source'];coarse_name=source.get('coarse_configuration')
        declared=source.get('solver_status')=='feasible_witness' and source.get('solver')=='embedded'
        witness={'declared_coarse_configuration':coarse_name,'solver_returned_embedded_witness':declared,
                 'exact_copy_of_declared_coarse':None,'declaration_disagrees_with_exact_copy':False}
        if coarse_name is None:witness['reference_status']='none_declared'
        else:
            other=by_unit.get((coarse_name,r['public']['anchor']))
            if other is None:witness['reference_status']='unavailable_in_verified_map_set'
            else:
                compatible=(other['public']['input']=='T0' and r['public']['input']!='T0'
                            and other['public']['encoder_sha256']==r['public']['encoder_sha256']
                            and np.array_equal(other['offsets'],r['offsets'])
                            and other['Q'].shape[0]*2==r['Q'].shape[0])
                pin=source.get('coarse_solution_sha256')
                if pin is not None and pin!=other['source'].get('solution_sha256'):
                    witness['reference_status']='coarse_solution_receipt_pin_mismatch'
                else:
                    witness['reference_status']='verified_reference_available'
                    exact=compatible and np.array_equal(r['Q'],np.repeat(other['Q'],2,axis=0))
                    witness['exact_copy_of_declared_coarse']=bool(exact)
                    witness['declaration_disagrees_with_exact_copy']=bool(declared and not exact)
        r['public']['witness']=witness
    def ordered(mode):
        result=[]
        for group in groups[mode].values():
            group['members'].sort(key=lambda x:(x['anchor'],x['configuration']));result.append(group)
        return sorted(result,key=lambda x:(x['anchor'],x['group_sha256']))
    n=len(normalized)
    nominal_records=None if nominal_specs is None else _specs(nominal_specs)
    if nominal_records is not None and not set(by_unit).issubset(nominal_records):
        raise ValueError('Verified records are outside the supplied nominal specifications')
    nominal=n if nominal_records is None else len(nominal_records)
    return {'schema':1,'method':'exact full-array equality; signed zeros compare equal; no rounding or prediction-based equivalence',
            'scope':'saved per-anchor empirical channels and their fixed action decoder; all stored objects retained',
            'independent_evidence_claimed':False,
            'construction_identity':'same verified frozen encoder artifact, including teachers, input codebook and ordered action dictionaries',
            'counts':{'nominal_map_units':nominal,'accepted_verified_map_units':n,
                      'unique_literal_kernel_groups':len(groups['literal']),
                      'unique_reduced_kernel_groups':len(groups['reduced']),
                      'literal_duplicate_map_units':n-len(groups['literal']),
                      'reduced_duplicate_map_units':n-len(groups['reduced']),
                      'solver_embedded_witness_returns':sum(r['public']['witness']['solver_returned_embedded_witness'] for r in normalized),
                      'exact_declared_coarse_copy_units':sum(r['public']['witness']['exact_copy_of_declared_coarse'] is True for r in normalized),
                      'witness_copy_disagreement_units':sum(r['public']['witness']['declaration_disagrees_with_exact_copy'] for r in normalized),
                      'witness_reference_unavailable_units':sum(r['public']['witness']['reference_status']=='unavailable_in_verified_map_set' for r in normalized)},
            'maps':sorted((r['public'] for r in normalized),key=lambda r:(r['anchor'],r['configuration'])),
            'literal_groups':ordered('literal'),'reduced_groups':ordered('reduced')}


def _specs(values):
    result={}
    for spec in values:
        key=(_name(spec['configuration']),_index(spec['anchor']))
        if key[1] not in (0,1,2) or spec.get('input') not in config.INPUTS or spec.get('kind')=='control':
            raise ValueError('Expected registered finite-map specifications')
        if key in result:raise ValueError('Duplicate nominal map/anchor specification')
        result[key]=spec
    return result


def _json(path):return json.loads(Path(path).read_text())


def _verified_json(path,expected):
    contents=Path(path).read_bytes()
    if hashlib.sha256(contents).hexdigest()!=_hash(expected):raise ValueError('JSON artifact hash mismatch')
    return json.loads(contents)


def _extra_provenance(root,spec,receipt):
    branch=spec.get('branch')
    if branch is None:return
    if branch=='A':
        registration,schedule,key,source='EXTRA_CONFIGS.json','EXTRA_RESOURCE_SCHEDULE.json','extra_registration_sha256','branches.py'
        source_key='branch_source_sha256'
        from .branches import action33_specs,action33_controls
        expected_maps,expected_controls=action33_specs(),action33_controls()
    elif branch=='C':
        registration,schedule,key,source='ROBUSTNESS_CONFIGS.json','ROBUSTNESS_RESOURCE_SCHEDULE.json','robustness_registration_sha256','robustness.py'
        source_key='robustness_source_sha256'
        from .robustness import robustness_specs
        expected_maps,expected_controls=robustness_specs(),[]
    else:raise ValueError('Unknown registered branch')
    reg=_verified_json(root/registration,spec[key]);planned=_json(root/schedule)
    expected={'schema':1,'branch':branch,'registered':True,
              'primary_config_hash':config.digest(config.configuration()),
              'scientific_source_hashes':run.source_fingerprint(),
              'maps':expected_maps,'controls':expected_controls}
    if (any(reg.get(k)!=v for k,v in expected.items()) or not reg.get('trigger',{}).get('triggered')
            or reg.get('registration_payload_hash')!=config.digest({k:v for k,v in reg.items() if k!='registration_payload_hash'})):
        raise ValueError('Frozen branch registration changed')
    if spec[source_key]!=run.sha(Path(__file__).with_name(source)) or reg.get(source_key)!=spec[source_key]:
        raise ValueError('Branch source identity changed')
    if (planned.get('schedule_payload_hash')!=config.digest({k:v for k,v in planned.items() if k!='schedule_payload_hash'})
            or planned.get('frozen_before_affected_outcomes') is not True
            or planned.get(key)!=spec[key] or spec['id'] not in planned.get('unit_ids',[])
            or planned.get('primary_resource_schedule_sha256')!=run.sha(root/'RESOURCE_SCHEDULE.json')
            or receipt.get('extra_resource_schedule_sha256')!=run.sha(root/schedule)):
        raise ValueError('Branch schedule identity changed')
    raw=next((r for r in reg['maps'] if r['configuration']==spec['configuration'] and r['anchor']==spec['anchor']),None)
    if raw is None or {**raw,key:spec[key],source_key:spec[source_key]}!=spec:
        raise ValueError('Map differs from frozen branch registration')


def collect_equivalence(*,out_root=config.OUT,specs=None,out_path=None):
    """Explicit read-only collection after resource freeze; no live call on import.

    Primary maps are nominal by default. Pass combined primary/registered extra
    map specs to include branches. Missing markers are pending or unfinished;
    malformed accepted artifacts are listed as invalid and never grouped.
    Optional public JSON output is atomic and refuses replacement.
    """
    root=Path(out_root)
    if out_path is not None and Path(out_path).exists():raise FileExistsError('Equivalence report already exists')
    schedule=_json(root/'RESOURCE_SCHEDULE.json')
    if (schedule.get('frozen_before_comparative_outcomes') is not True
            or schedule.get('config_hash')!=config.digest(config.configuration())):
        raise ValueError('Frozen resource schedule/configuration required')
    schedule_sha=run.sha(root/'RESOURCE_SCHEDULE.json')
    nominal=_specs(config.configuration()['maps'] if specs is None else specs)
    primary=_specs(config.configuration()['maps']);encoders={};records=[];states=[];marker_count=0

    def encoder(anchor):
        if anchor in encoders:
            if isinstance(encoders[anchor],Exception):raise encoders[anchor]
            return encoders[anchor]
        try:
            base=root/'private/run'/f'anchor_{anchor}';prep_path=base/'PREPARED.json';prep=_json(prep_path)
            run._validate_provenance(prep,'prepare')
            if prep.get('anchor')!=anchor:raise ValueError('Preparation anchor changed')
            expected=_hash(prep['artifact_hashes']['encoder/encoder.joblib']);path=base/'encoder/encoder.joblib'
            if run.sha(path)!=expected:raise ValueError('Frozen encoder hash mismatch')
            obj=joblib.load(path)
            if run.sha(path)!=expected:raise ValueError('Frozen encoder changed during read')
            from .encoding import Codebook
            if not isinstance(obj.code,Codebook):raise ValueError('Expected registered Codebook implementation')
            coarse=obj.code.n_states('T0')
            for family in config.INPUTS:
                size=coarse*(1 if family=='T0' else 2)
                expected_parents=np.arange(size) if family=='T0' else np.arange(size)//2
                if obj.code.n_states(family)!=size or not np.array_equal(obj.code.parents(family),expected_parents):
                    raise ValueError('Unregistered input parent relation')
            result=(obj,prep,expected,run.sha(prep_path));encoders[anchor]=result;return result
        except Exception as error:
            encoders[anchor]=error;raise

    for (name,anchor),spec in sorted(nominal.items(),key=lambda x:(x[0][1],x[0][0])):
        base=root/'private/run'/f'anchor_{anchor}'/'maps'/name;marker=base/'ACCEPTED.json'
        state={'configuration':name,'anchor':anchor}
        if not marker.exists():
            state['status']='unfinished' if base.exists() else 'pending_missing';states.append(state);continue
        marker_count+=1
        try:
            receipt=_json(marker);receipt_sha=run.sha(marker);run._validate_provenance(receipt,'map')
            if (receipt.get('configuration')!=name or receipt.get('anchor')!=anchor
                    or receipt.get('spec_hash')!=config.digest(spec)):
                raise ValueError('Accepted map identity/specification changed')
            if spec.get('branch') is None and primary.get((name,anchor))!=spec:
                raise ValueError('Map differs from primary registration')
            _extra_provenance(root,spec,receipt)
            obj,prep,identity,prep_sha=encoder(anchor)
            if receipt.get('prepared_cache_sha256')!=prep['cache_sha256']:
                raise ValueError('Map preparation receipt changed')
            hashes=receipt['artifact_hashes'];meta=_verified_json(base/'metadata.json',hashes['metadata.json'])
            actions=spec.get('max_actions',17);dictionary=obj.dictionaries[actions]
            offsets=_array(dictionary['offsets'],1);zero=_index(dictionary['zero_action'])
            if zero!=0 or not len(offsets)<=actions or offsets[zero]!=0:
                raise ValueError('Registered action dictionary changed')
            if (meta.get('feasible') is not True or meta.get('input')!=spec['input']
                    or meta.get('configuration')!=name or meta.get('actions')!=actions):
                raise ValueError('Accepted map metadata mismatch')
            q_path=base/'Q.npz';q_hash=_hash(hashes['Q.npz'])
            if run.sha(q_path)!=q_hash:raise ValueError('Channel artifact hash mismatch')
            with np.load(q_path,allow_pickle=False) as saved:
                if saved.files!=['Q']:raise ValueError('Unexpected channel archive fields')
                q=saved['Q'].copy()
            if run.sha(q_path)!=q_hash or run.sha(marker)!=receipt_sha:
                raise ValueError('Accepted channel changed during read')
            record={'configuration':name,'anchor':anchor,'input':spec['input'],'coarse_states':obj.code.n_states('T0'),
                    'Q':q,'offsets':offsets,'encoder_sha256':identity,'receipt_sha256':receipt_sha,
                    'Q_artifact_sha256':q_hash,'solution_sha256':_hash(receipt['sha256']),
                    'prepared_receipt_sha256':prep_sha,'solver_status':meta.get('status'),'solver':meta.get('solver'),
                    'coarse_configuration':receipt.get('coarse_configuration'),
                    'coarse_solution_sha256':receipt.get('coarse_solution_sha256')}
            _record(record);records.append(record);state['status']='accepted_verified'
        except Exception as error:
            state.update(status='invalid',reason='artifact_or_provenance_validation_failed',error_type=type(error).__name__)
        states.append(state)
    if run.sha(root/'RESOURCE_SCHEDULE.json')!=schedule_sha:raise ValueError('Resource schedule changed during collection')
    result=group_maps(records,nominal_specs=list(nominal.values()))
    result.update(resource_schedule_sha256=schedule_sha,configuration_sha256=config.digest(config.configuration()),
                  collector_source_sha256=run.sha(__file__),units=states,
                  verification_scope='Current source/config provenance, registered map spec, preparation/encoder pin, exact encoder and channel artifact bytes, metadata hash; solution and prepared cache hashes remain receipt declarations and their objects are not read')
    result['counts'].update(accepted_marker_units=marker_count,
        pending_missing_map_units=sum(s['status']=='pending_missing' for s in states),
        unfinished_map_units=sum(s['status']=='unfinished' for s in states),
        invalid_map_units=sum(s['status']=='invalid' for s in states))
    result['status']='complete' if len(records)==len(nominal) else 'pending_or_invalid_units'
    result['counts_by_anchor']={str(a):{'nominal':sum(s['anchor']==a for s in states),
                                      'accepted_verified':sum(s['anchor']==a for s in records),
                                      'unique_literal':sum(g['anchor']==a for g in result['literal_groups']),
                                      'unique_reduced':sum(g['anchor']==a for g in result['reduced_groups'])} for a in (0,1,2)}
    if out_path is not None:
        path=Path(out_path);path.parent.mkdir(parents=True,exist_ok=True)
        encoded=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
        with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
            temporary=Path(f.name);f.write(encoded);f.flush();os.fsync(f.fileno())
        try:os.link(temporary,path)
        finally:temporary.unlink(missing_ok=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out-root',type=Path,default=config.OUT)
    parser.add_argument('--specs',type=Path);parser.add_argument('--out-path',type=Path)
    args=parser.parse_args();specs=_json(args.specs) if args.specs else None
    report=collect_equivalence(out_root=args.out_root,specs=specs,out_path=args.out_path)
    print(json.dumps({'status':report['status'],'counts':report['counts']},sort_keys=True))
