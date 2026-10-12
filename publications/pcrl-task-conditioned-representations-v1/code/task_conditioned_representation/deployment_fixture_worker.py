"""Data-free deployment contract checks; executes only in admission queue."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(spec_path):
    spec=json.loads(Path(spec_path).read_text())
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.manual_seed(2026101207)
    from .deployment import RuntimeModel,load_package,export_checkpoint,sha_file
    from .source import write_json
    directory=Path(spec['private_output']);directory.mkdir(parents=True,exist_ok=False)
    names=[f'permitted_column_{i}' for i in range(83)]
    model=RuntimeModel(names);model.eval()
    checkpoint=directory/'synthetic_checkpoint.pt'
    torch.save({'schema':'TCR1-trainable-own-head-checkpoint-v1','feature_names':names,'class_orders':[[0,1],list(range(6))],'model':model.state_dict(),'family':'DATA_FREE_ENGINEERING','seed':0},checkpoint)
    binding=export_checkpoint(checkpoint,sha_file(checkpoint),directory/'package')
    runtime,meta=load_package(directory/'package',binding)
    x=np.random.default_rng(707).normal(size=(9,83)).astype(np.float32)
    views=runtime.release(x,names)
    checks={'raw83_only_without_labels':True,'actual_saved_state_parity':all(np.array_equal(a['full'],b['full']) for a,b in zip(views,model.release(x,names))), 'wire_dimensions': [v['full'].shape[1] for v in views]==[21,29], 'score_dimensions':[v['output_only'].shape[1] for v in views]==[3,7]}
    def refused(name,fn):
        try:fn()
        except (ValueError,RuntimeError,TypeError):checks[name]=True
        else:raise AssertionError(name+' was accepted')
    refused('reordered_columns_refused',lambda:runtime.release(x,names[::-1]))
    refused('missing_columns_refused',lambda:runtime.release(x[:,:82],names[:-1]))
    refused('extra_columns_refused',lambda:runtime.release(np.c_[x,x[:,0]],names+['extra']))
    bad=x.copy();bad[0,0]=np.nan
    refused('nonfinite_input_refused',lambda:runtime.release(bad,names))
    refused('SEX_input_refused',lambda:runtime.release(x,names,SEX=np.zeros(9)))
    refused('true_task_input_refused',lambda:runtime.release(x,names,true_income=np.zeros(9)))
    refused('hidden_output_refused',lambda:runtime.release(x,names,original_U0_scores=np.zeros((9,2))))
    refused('output_flags_refused',lambda:runtime.release(x,names,return_source_features=True))
    refused('wrong_manifest_hash_refused',lambda:load_package(directory/'package','0'*64))
    for i in range(2):
        with torch.inference_mode():
            z=torch.from_numpy(views[i]['representation']);l=runtime.head[i](z.double());l-=l.mean(1,keepdim=True)
            checks[f'own_head_reads_actual_z_{i}']=np.array_equal(l.numpy(),views[i]['centered_logits'])
    # Compensating feature/head scale must preserve task prediction and wire
    # must retain changed representation; no source bypass is allowed.
    with torch.no_grad():
        runtime.enc[0][4].weight.mul_(2);runtime.enc[0][4].bias.mul_(2);runtime.head[0].weight.div_(2)
    scaled=runtime.release(x,names)
    checks['own_feature_scale_transmitted']=np.array_equal(scaled[0]['representation'],views[0]['representation']*2)
    checks['compensated_predictions_preserved']=np.allclose(scaled[0]['probabilities'],views[0]['probabilities'],rtol=0,atol=1e-14)
    if not all(checks.values()):raise AssertionError('deployment fixture failed')
    result={'schema':'TCR1-deployment-fixture-v1','status':'PASS','checks':checks,'data_free':True,'source_arrays_opened':False,'source_services_loaded':False,'parameter_count':meta['parameter_count'],'package_manifest_sha256':binding}
    write_json(spec['public'],result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();print(json.dumps(run(a.spec),sort_keys=True))
