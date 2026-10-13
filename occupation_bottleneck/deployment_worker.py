"""Queue-only package export, raw83 execution and snapshot-only numerical replay."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path


def run(config):
    from .evaluation_worker import queue_config
    cfg0=json.loads(Path(config).read_text());active,cfg,verify,guard=queue_config(config,cfg0.get('stage','verification'))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .deployment import load_package,export_checkpoint,sha_file
    from .source import write_json
    operation=cfg['operation']
    if operation=='export_live':
        from .evaluation_worker import manifest_and_model,forward_release
        from .source import load_role
        manifest,source=manifest_and_model(cfg,verify)
        features=manifest['roles'][cfg.get('role','INNER_SELECTION')]['features'];verify(features['path'],features['sha256'])
        data=load_role(manifest,cfg['seed'],cfg.get('role','INNER_SELECTION'),{'features'})
        campaign=Path(__file__).resolve().parents[1]/'.private/occupation_bottleneck'
        for key in ('package','input','expected'):
            if not Path(cfg[key]).resolve().is_relative_to(campaign):raise PermissionError('private campaign output required')
        variant=cfg.get('release_variant','full');binding=export_checkpoint(cfg['checkpoint'],cfg['checkpoint_sha256'],cfg['package'],variant)
        runtime,meta=load_package(cfg['package'],binding);actual=runtime.release(data['X'],data['feature_names'].tolist())
        projection=None;audits=[]
        if variant=='head_row_space':
            from .geometry import build_geometry,audit_geometry
            projection=[build_geometry(h.weight.detach().numpy(),h.bias.detach().numpy(),source.class_orders[i]) for i,h in enumerate(source.head)]
            original=source.route_raw(data['X'],data['feature_names'])
            audits=[audit_geometry(v['representation'],g) for v,g in zip(original,projection)]
            if not all(a['passed'] for a in audits):raise AssertionError('projection decision-preservation check failed')
        expected=forward_release(source,data,projection=projection)
        differences=[float(np.max(np.abs(actual[i]['full']-expected[f'r{i+1}']))) for i in range(2)]
        if any(differences):raise AssertionError('standalone release differs from actual source wire')
        for key in ('input','expected'):Path(cfg[key]).parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with Path(cfg['input']).open('xb') as f:np.savez(f,X=data['X'],feature_names=data['feature_names'])
        with Path(cfg['expected']).open('xb') as f:np.savez(f,r1=actual[0]['full'],r2=actual[1]['full'])
        for key in ('input','expected'):Path(cfg[key]).chmod(0o600)
        output=str(Path(cfg['expected']).with_suffix('.cli.npz'));guard()
        command=[sys.executable,'-m','occupation_bottleneck.deployment','--package',cfg['package'],'--manifest-sha256',binding,'--input',cfg['input'],'--input-sha256',sha_file(cfg['input']),'--output',output]
        completed=subprocess.run(command,check=True,capture_output=True,text=True)
        with np.load(output,allow_pickle=False) as z:
            if any(not np.array_equal(z[f'r{i+1}'],actual[i]['full']) for i in range(2)):raise AssertionError('CLI wire differs')
        # Real schema negatives, using actual rows. No labels are loaded.
        rejects=[];names=data['feature_names'].tolist();x=data['X'][:1]
        for label,call in [('reordered',lambda:runtime.release(x,names[::-1])),('extra_column',lambda:runtime.release(np.column_stack((x,x[:,0])),names+['SEX'])),('hidden_flag',lambda:runtime.release(x,names,hidden_flag=True)),('teacher_output',lambda:runtime.release(x,names,teacher_score=0)),('nonfinite',lambda:runtime.release(np.full_like(x,np.nan),names)),('SEX_column',lambda:runtime.release(x,['SEX']+names[1:])),('Y_column',lambda:runtime.release(x,['Y']+names[1:])),('ID_column',lambda:runtime.release(x,['row_id']+names[1:])),('income_label_column',lambda:runtime.release(x,['income']+names[1:])),('occupation_label_column',lambda:runtime.release(x,['occupation']+names[1:])),('manifest_pin_mismatch',lambda:load_package(cfg['package'],'0'*64))]:
            try:call()
            except ValueError:rejects.append(label)
            else:raise AssertionError('runtime admitted forbidden input '+label)
        from importlib.metadata import version
        result={'schema':'OB1-live-deployment-receipt-v1','status':'PASS','arm_id':cfg['arm_id'],'seed':cfg['seed'],'release_variant':variant,'rows':len(data['X']),'package_manifest_sha256':binding,'model_sha256':meta['model_sha256'],'projection':meta['projection'],'checkpoint_sha256':cfg['checkpoint_sha256'],'input_sha256':sha_file(cfg['input']),'expected_sha256':sha_file(cfg['expected']),'CLI_output_sha256':sha_file(output),'CLI_receipt':json.loads(completed.stdout),'max_absolute_wire_difference':differences,'projection_audits':audits,'full_dimensions':[21,29,50],'labels_read':False,'source_teacher_co_release':False,'negative_schema_cases_rejected':rejects,'environment_pins':{name:version(name) for name in ('numpy','torch','scipy','scikit-learn','joblib')}}
    elif operation=='restore':
        root=Path(cfg['restore_root']).resolve()
        if not Path(__file__).resolve().is_relative_to(root):raise PermissionError('numerical restore must execute snapshot code')
        from importlib.metadata import version
        if cfg['environment_pins']!={name:version(name) for name in cfg['environment_pins']}:raise PermissionError('restore environment differs')
        for key in ('package','input','expected'):
            if not Path(cfg[key]).resolve().is_relative_to(root):raise PermissionError('restore bytes outside restored tree')
        for pin in cfg['package_files']:verify(pin['path'],pin['sha256'])
        runtime,meta=load_package(cfg['package'],cfg['package_manifest_sha256'])
        for key in ('input','expected'):verify(cfg[key],cfg[key+'_sha256'])
        with np.load(cfg['input'],allow_pickle=False) as z:
            if set(z.files)!={'X','feature_names'}:raise PermissionError('runtime input contains hidden fields')
            actual=runtime.release(z['X'],z['feature_names'].tolist())
        with np.load(cfg['expected'],allow_pickle=False) as z:
            if set(z.files)!={'r1','r2'}:raise PermissionError('unexpected replay outputs')
            differences=[float(np.max(np.abs(actual[i]['full']-z[f'r{i+1}']))) for i in range(2)]
        if any(differences):raise AssertionError('restored wire differs')
        checks=[]
        if not cfg.get('readers'):raise PermissionError('at least one representative frozen-reader replay required')
        from .readers import tables,kind_for
        from task_conditioned_representation.attacks import predict_reader,orient
        import joblib
        tab=tables(actual[0]['full'],actual[1]['full'])
        for record in cfg['readers']:
            for key in ('path','expected'):
                if not Path(record[key]).resolve().is_relative_to(root):raise PermissionError('reader outside restore')
                verify(record[key],record[key+'_sha256'])
            guard();reader=joblib.load(record['path']);scores=orient(predict_reader(reader,tab[record['view']][kind_for(record['reader_name'])]),record['orientation'])
            with np.load(record['expected'],allow_pickle=False) as z:expected=z[record['score_field']]
            diff=float(np.max(np.abs(scores-expected)))
            if diff!=0:raise AssertionError('frozen reader restore differs')
            checks.append({'view':record['view'],'reader_name':record['reader_name'],'orientation':record['orientation'],'score_field':record['score_field'],'max_absolute_score_difference':diff,'model_sha256':record['path_sha256']})
        result={'schema':'OB1-snapshot-only-numerical-replay-v1','status':'PASS','arm_id':cfg['arm_id'],'seed':cfg['seed'],'release_variant':cfg['release_variant'],'rows':len(actual[0]['full']),'package_manifest_sha256':cfg['package_manifest_sha256'],'max_absolute_wire_difference':differences,'reader_checks':checks,'original_sources_used':False,'labels_read':False,'runtime_code_from_snapshot':True}
    else:raise ValueError('unregistered deployment operation')
    write_json(cfg['public_output'],result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();out=run(a.spec);print(json.dumps({'status':out['status'],'arm_id':out['arm_id'],'seed':out['seed']}))
