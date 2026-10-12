"""New readers on the separate literal historical finite score-wire contract."""
from __future__ import annotations
import argparse,json
from pathlib import Path
LAW_FIELDS={'row_id','unit','code_r1','code_r2','mass'}


def legacy_adapter():
    # Registered random-seed update only; validated scientific reader slate is retained.
    from conditional_release import attacks as legacy
    from .attacks import FiniteScaler
    legacy.StandardScaler=FiniteScaler
    legacy.TRAIN_SAMPLE_SEED_BASE=2026101205
    legacy.SHUFFLE_ATT_SALT='TCR1|legacy-ATT-null|2026101206'
    legacy.SHUFFLE_INNER_SALT='TCR1|legacy-INNER-null|2026101207'
    return legacy


def run(spec_path,phase):
    from .attack_worker import queue_config,reporting_lock,read_reporting_sex
    active,cfg,verify=queue_config(spec_path,'attacks')
    import numpy as np
    from .source import load_manifest,load_role,read_exact
    from .attacks import write_json,sha_file,save_npz
    legacy=legacy_adapter()
    bank=json.loads(verify(cfg['bank']['path'],cfg['bank']['sha256']).read_text())
    verify(cfg['manifest'],cfg['manifest_sha256']);manifest=load_manifest(cfg['manifest'],cfg['manifest_sha256'])
    if phase=='fit':
        verify(cfg['science_lock'],cfg['science_lock_sha256']);lock=json.loads(verify(cfg['attack_lock'],cfg['attack_lock_sha256']).read_text())
        if lock.get('status')!='RELEASES_AND_ENDPOINTS_FROZEN' or not any(i['arm_id']==cfg['arm_id'] and i['seed']==cfg['seed'] for i in lock.get('legacy_instances',[])):raise PermissionError('legacy release missing from global stage freeze')
        verify(cfg['law_origin_receipt']['path'],cfg['law_origin_receipt']['sha256'])
        for key in ('att_law','inner_law','public_prior'):verify(cfg[key]['path'],cfg[key]['sha256'])
        att_law=read_exact(cfg['att_law'],LAW_FIELDS);inner_law=read_exact(cfg['inner_law'],LAW_FIELDS)
        att=load_role(manifest,cfg['seed'],'ATTACK_FIT_NEW',{'features','sex'});inner=load_role(manifest,cfg['seed'],'INNER_SELECTION',{'features'})
        for law,source in ((att_law,att),(inner_law,inner)):
            if not all(np.array_equal(law[k],source[k]) for k in ('row_id','unit')):raise PermissionError('historical finite law people differ from admitted role')
        with np.load(cfg['public_prior']['path'],allow_pickle=False) as z:
            if z.files!=['pair']:raise PermissionError('public prior schema differs')
            prior=z['pair'].copy()
        def guard():
            from .scheduler import assert_active_from_env
            assert_active_from_env(unit_id=active['unit_id'],stage='attacks')
            verify(cfg['science_lock'],cfg['science_lock_sha256']);verify(cfg['attack_lock'],cfg['attack_lock_sha256'])
        receipt=legacy.fit_complete_readers(bank,att_law,{k:att[k] for k in ('row_id','unit','sex')},uniform_public_pair_mass=prior,seed=cfg['seed'],private_root=cfg['private_output'],queue_guard=guard,source_bindings={'arm_id':cfg['arm_id'],'law_origin_receipt_sha256':cfg['law_origin_receipt']['sha256'],'att_law_sha256':cfg['att_law']['sha256'],'inner_law_sha256':cfg['inner_law']['sha256']})
        save_npz(Path(cfg['private_output'])/'INNER_LAW.npz',**inner_law)
        out={**receipt,'arm_id':cfg['arm_id'],'unit_id':active['unit_id'],'contract':'legacy_complete_score_wire_codes_probabilities_decisions_metadata','not_equal_capacity_new_features':True,'frozen_complete_sha256':sha_file(Path(cfg['private_output'])/'FROZEN_COMPLETE.json'),'inner_law_sha256':sha_file(Path(cfg['private_output'])/'INNER_LAW.npz')}
    else:
        reporting_lock(cfg,verify)
        for key in ('inner_law','freeze','score_tables'):verify(cfg[key]['path'],cfg[key]['sha256'])
        law=read_exact(cfg['inner_law'],LAW_FIELDS);sex=read_reporting_sex(cfg['inner_sex'],verify,law)
        with np.load(cfg['score_tables']['path'],allow_pickle=False) as z:tables={k:z[k].copy() for k in z.files}
        report,_=legacy.score_frozen_readers(bank,law,sex,tables)
        out={**report,'schema':'TCR1-fresh-legacy-complete-score-wire-report-v1','arm_id':cfg['arm_id'],'seed':cfg['seed'],'unit_id':active['unit_id'],'contract':'legacy_complete_score_wire_codes_probabilities_decisions_metadata','not_equal_capacity_new_features':True,'freeze_sha256':cfg['freeze']['sha256'],'diagnostic_only':False}
    write_json(cfg['public_output'],out,False);return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--phase',choices=('fit','report'),default='fit');a=p.parse_args();out=run(a.spec,a.phase);print(json.dumps({'status':out.get('status','REPORTED'),'arm_id':out['arm_id']}))
if __name__=='__main__':main()
