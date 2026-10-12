"""Queue-authenticated package export, live release and restore parity worker."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path


def run(spec_path):
    spec=json.loads(Path(spec_path).read_text())
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage=spec.get('stage','verification'))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .deployment import export_checkpoint,load_package,sha_file
    from .source import load_manifest,load_role,write_json
    operation=spec['operation']
    if operation=='export_live':
        from .models import load_checkpoint,actual_wires
        model,payload=load_checkpoint(spec['checkpoint'],spec['checkpoint_sha256'])
        manifest=load_manifest(spec['source_manifest'],spec['source_manifest_sha256'])
        # Features only: neither task labels nor SEX are needed to deploy.
        data=load_role(manifest,spec['seed'],spec.get('role','INNER_SELECTION'),{'features'})
        binding=export_checkpoint(spec['checkpoint'],spec['checkpoint_sha256'],spec['package'])
        runtime,meta=load_package(spec['package'],binding)
        x=data['X'];names=list(data['feature_names'])
        actual=runtime.release(x,names)
        with torch.inference_mode():expected=actual_wires(model,torch.from_numpy(x))
        diffs=[]
        for i in range(2):
            diff=float(np.max(np.abs(actual[i]['full']-expected[i]['full'].numpy())))
            if diff!=0.:raise AssertionError('deployed raw83 wire differs from selected checkpoint')
            diffs.append(diff)
        # Save feature-only deployment inputs and the actual wires privately.
        np.savez(spec['input'],X=x,feature_names=data['feature_names'])
        np.savez(spec['expected'],recipient1=actual[0]['full'],recipient2=actual[1]['full'],score1=actual[0]['output_only'],score2=actual[1]['output_only'])
        for field in ('input','expected'):os.chmod(spec[field],0o600)
        cli_output=str(Path(spec['expected']).with_suffix('.cli.npz'))
        command=[sys.executable,'-m','task_conditioned_representation.deployment','--package',spec['package'],'--manifest-sha256',binding,'--input',spec['input'],'--input-sha256',sha_file(spec['input']),'--output',cli_output,'--contract','full']
        completed=subprocess.run(command,capture_output=True,text=True,check=True)
        with np.load(cli_output,allow_pickle=False) as z:
            if any(not np.array_equal(z[f'recipient{i+1}'],actual[i]['full']) for i in range(2)):raise AssertionError('standalone CLI wire differs')
        from importlib.metadata import version
        versions={name:version(name) for name in ('numpy','torch','scipy','scikit-learn','joblib')}
        result={'schema':'TCR1-live-deployment-receipt-v1','status':'PASS','family':meta['provenance'].get('family'),'seed':spec['seed'],'rows':len(x),'package_manifest_sha256':binding,'model_sha256':meta['model_sha256'],'source_checkpoint_sha256':spec['checkpoint_sha256'],'input_sha256':sha_file(spec['input']),'expected_sha256':sha_file(spec['expected']),'max_abs_wire_difference':diffs,'full_dimensions':[21,29,50],'score_dimensions':[3,7,10],'true_labels_or_SEX_read':False,'actual_new_saved_encoders_heads':True,'standalone_CLI_executed':True,'standalone_CLI_exact_parity':True,'standalone_CLI_output_sha256':sha_file(cli_output),'standalone_CLI_receipt':json.loads(completed.stdout),'environment_pins':versions}
    elif operation=='restore':
        # Deliberately requires no source manifest or checkpoint. All numerical
        # inputs, deployed weights and reader estimators are within restore root.
        root=Path(spec['restore_root']).resolve()
        if not Path(__file__).resolve().is_relative_to(root):raise PermissionError('numerical restore must execute its backed-up runtime code')
        for field in ('package','input','expected'):
            if not Path(spec[field]).resolve().is_relative_to(root):raise PermissionError('restore input outside restored tree')
        runtime,meta=load_package(spec['package'],spec['package_manifest_sha256'])
        for field in ('input','expected'):
            if sha_file(spec[field])!=spec[field+'_sha256']:raise ValueError('restored replay pin differs')
        with np.load(spec['input'],allow_pickle=False) as z:
            if set(z.files)!={'X','feature_names'}:raise ValueError('restored deployment inputs contain hidden fields')
            actual=runtime.release(z['X'],z['feature_names'].tolist())
        with np.load(spec['expected'],allow_pickle=False) as z:
            differences=[float(np.max(np.abs(actual[i]['full']-z[f'recipient{i+1}']))) for i in range(2)]
            score_differences=[float(np.max(np.abs(actual[i]['output_only']-z[f'score{i+1}']))) for i in range(2)]
        if any(differences+score_differences):raise AssertionError('restored numerical wire differs')
        reader_checks=[]
        for record in spec.get('readers',[]):
            import joblib
            for key in ('path','expected'):
                if not Path(record[key]).resolve().is_relative_to(root) or sha_file(record[key])!=record[key+'_sha256']:raise PermissionError('reader outside restore or hash differs')
            from .attacks import feature_tables,reader_kind,predict_reader,orient
            view=record['view'];name=record['reader_name']
            wire=feature_tables(actual[0]['full'],actual[1]['full'])[view][reader_kind(name)]
            reader=joblib.load(record['path']);scores=orient(predict_reader(reader,wire),record['orientation'])
            with np.load(record['expected'],allow_pickle=False) as z:target=z[record.get('score_field',view+'__'+name)]
            diff=float(np.max(np.abs(scores-target)))
            if diff!=0.:raise AssertionError('restored reader differs')
            reader_checks.append({'view':view,'reader_name':name,'orientation':record['orientation'],'max_abs_score_difference':diff,'reader_sha256':record['path_sha256']})
        result={'schema':'TCR1-restored-numerical-deployment-receipt-v1','status':'PASS','rows':len(actual[0]['full']),'family':meta['provenance'].get('family'),'package_manifest_sha256':spec['package_manifest_sha256'],'max_abs_wire_difference':differences,'max_abs_score_wire_difference':score_differences,'reader_checks':reader_checks,'original_sources_used':False,'labels_required':False}
    else:raise ValueError('unregistered deployment operation')
    write_json(spec['public'],result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();print(json.dumps(run(a.spec),sort_keys=True))
