"""Queue-owned independent model/reader replay from restored bytes only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile


class RestoredInputs:
    def __init__(self, config):
        from compressed_income_safeguard.scheduler import assert_active_from_env
        self.guard = assert_active_from_env
        active = self.guard(stage='custody')
        state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
        self.pins = {str(Path(p['path']).absolute()): p['sha256'] for p in state['units'][active['unit_id']]['spec']['sources']}
        self.cfg = json.loads(self.path(config, metadata_config=True).read_text())
        if self.cfg['unit_id'] != active['unit_id']:
            raise PermissionError('restore verifier unit differs')
        self.root = Path(self.cfg['restore_root']).absolute()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise PermissionError('restore root symlink refused')
        self.root = self.root.resolve(strict=True)
        from compressed_income_safeguard_independent import replay
        self.replay = replay
        for code in (Path(__file__), Path(replay.__file__)):
            self.path(code)

    def path(self, item, metadata_config=False):
        self.guard(stage='custody')
        p = Path(item['path'] if isinstance(item, dict) else item).absolute()
        if any(q.is_symlink() for q in (p, *p.parents)):
            raise PermissionError('restore symlink source refused')
        if not metadata_config and not p.resolve(strict=True).is_relative_to(self.root):
            raise PermissionError('numerical restore source/code outside restored tree')
        low = p.as_posix().lower()
        if any(name in low for name in ('head_validation', 'old_assessment', 'osf_development_assessment', 'all_role', 'all-role')):
            raise PermissionError('barred restore source')
        if ('inner_selection' in low or 'inner_truth' in low) and any(p.stem.lower()==name or p.stem.lower().endswith('_'+name) for name in ('sex','income','occupation')):
            raise PermissionError('restore verifier never opens or hashes labels')
        expected = self.pins.get(str(p))
        if not expected or isinstance(item, dict) and item['sha256'] != expected:
            raise PermissionError('restore source not exactly queue registered')
        with p.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise PermissionError('restored bytes differ')
        return p

    def arrays(self, item, fields):
        import numpy as np
        p = self.path(item)
        with ZipFile(p) as archive:
            names = [v.filename for v in archive.infolist()]
            if len(names) != len(set(names)) or set(names) != {name+'.npy' for name in fields}:
                raise PermissionError('exact restored label-free array schema required')
        with np.load(p, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in fields}


def restored_wires(inputs, item):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    manifest = json.loads(inputs.path(item['deployment']).read_text())
    fields = {'schema','feature_names','feature_schema_sha256','class_orders','model_sha256','source_checkpoint_sha256','provenance','contract','parameter_count','projection'}
    if set(manifest) != fields or manifest['schema'] != 'OB1-raw83-own-head-deployment-v1' or manifest['class_orders'] != [[0,1],list(range(6))]:
        raise ValueError('restored deployment schema/order differs')
    names = manifest['feature_names']
    forbidden = {'sex','race','income','occupation','fnlwgt','row_id','unit','id','person_id','sampling_id','group_id','hidden_flag','y','label','target','teacher','teacher_score'}
    if len(names)!=83 or len(set(names))!=83 or any(not isinstance(n,str) or not n or n.lower().split('=')[0] in forbidden for n in names):
        raise ValueError('restored exact83 feature schema differs')
    if hashlib.sha256(json.dumps(names,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()!=manifest['feature_schema_sha256']:
        raise ValueError('restored feature schema hash differs')
    contract = {'recipient_full_dimensions':[21,29],'pair_full_dimensions':50,'recipient_score_dimensions':[3,7],'pair_score_dimensions':10,'deterministic':True,'head_reads_released_representation':True}
    if manifest['contract']!=contract or manifest['parameter_count']!=21288 or manifest['provenance']['seed']!=item['seed'] or manifest['model_sha256']!=item['model_state']['sha256']:
        raise ValueError('restored model/head contract or identity differs')
    state = torch.load(inputs.path(item['model_state']),map_location='cpu',weights_only=True)
    shapes = {}
    for i,classes in ((0,2),(1,6)):
        for layer,ins,outs in ((0,83,64),(2,64,64),(4,64,16)):
            shapes[f'enc.{i}.{layer}.weight']=(outs,ins); shapes[f'enc.{i}.{layer}.bias']=(outs,)
        shapes[f'head.{i}.weight']=(classes,16); shapes[f'head.{i}.bias']=(classes,)
    if set(state)!=set(shapes) or any(not isinstance(v,torch.Tensor) or tuple(v.shape)!=shapes[k] or v.dtype!=(torch.float32 if k.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for k,v in state.items()):
        raise ValueError('restored encoder/head tensor schema differs')
    raw = inputs.arrays(item['input'],{'X','feature_names'})
    if raw['feature_names'].tolist()!=names or raw['X'].ndim!=2 or raw['X'].shape[1]!=83 or raw['X'].dtype.kind!='f' or not np.isfinite(raw['X']).all():
        raise ValueError('restored raw83 input differs')
    with np.errstate(over='ignore'):
        x=np.ascontiguousarray(raw['X'],dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('restored raw83 overflows encoder precision')
    wires=inputs.replay.forward({'schema':'TCR1-trainable-own-head-checkpoint-v1','class_orders':manifest['class_orders'],'model':state},x)
    projection = manifest['projection']
    if projection is not None:
        if item['release_variant']!='head_row_space' or projection['head_binding_sha256']!=manifest['model_sha256'] or projection['release']!='factorized_Pz16' or projection['representation_dimension']!=16 or projection['sha256']!=item['projection']['sha256']:
            raise ValueError('restored projection binding differs')
        matrices=inputs.arrays(item['projection'],{'B0','P0','B1','P1'})
        for i,classes in ((0,2),(1,6)):
            basis=matrices[f'B{i}'];projector=matrices[f'P{i}'];rank=projection['ranks'][i];weight=state[f'head.{i}.weight'].numpy();bias=state[f'head.{i}.bias'].numpy();a=weight-weight.mean(0)
            if not isinstance(rank,int) or not 0<=rank<classes or basis.dtype!=np.float64 or basis.shape!=(rank,16) or projector.dtype!=np.float64 or projector.shape!=(16,16) or not np.isfinite(basis).all() or not np.array_equal(basis.T@basis,projector) or not np.allclose(basis@basis.T,np.eye(rank),atol=1e-10,rtol=1e-10) or not np.allclose(a@projector,a,atol=1e-10,rtol=1e-10):
                raise ValueError('restored own-head basis/projector differs')
            z=(wires[i][:,:16]@basis.T)@basis;logits=z@weight.T+bias;logits-=logits.mean(1,keepdims=True);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
            wires[i]=np.column_stack((z,logits,p,p.argmax(1)))
    elif item['release_variant']!='full':
        raise ValueError('restored release variant differs')
    expected=inputs.arrays(item['expected_wire'],{'r1','r2'})
    for i in (0,1):
        if not np.array_equal(wires[i],expected[f'r{i+1}']):raise AssertionError('independently forwarded restored wire differs')
    return wires


def run(inputs):
    import numpy as np
    import joblib
    cfg=inputs.cfg;records=[]
    if len(cfg['representatives'])!=2 or {(p['arm_id'],p['seed']) for p in cfg['representatives']}!={('U0',0),('RAW_JOINT_R03_E5',0)}:
        raise ValueError('two fixed independent restored representatives required')
    for item in cfg['representatives']:
        inputs.guard(stage='custody');wires=restored_wires(inputs,item);checks=[]
        if not item['readers']:raise ValueError('representative independent frozen reader required')
        for reader in item['readers']:
            inputs.guard(stage='custody');saved=joblib.load(inputs.path(reader['model']))
            if not isinstance(saved,dict) or set(saved)!={'model','mean','scale'} or not np.array_equal(saved['model'].classes_,[0,1]):raise ValueError('independent restored reader schema/classes differ')
            table=inputs.replay.reader_table(wires[0],wires[1],reader['reader'],reader.get('view','pair'));mean=np.asarray(saved['mean']);scale=np.asarray(saved['scale'])
            if mean.shape!=(table.shape[1],) or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any() or reader['orientation'] not in (-1,1):raise ValueError('restored frozen reader preprocessing/orientation differs')
            scores=saved['model'].predict_proba((table-mean)/scale)[:,1]
            if reader['orientation']==-1:scores=1-scores
            expected=inputs.arrays(reader['expected'],{'row_id','unit','score'})
            if expected['score'].shape!=scores.shape or expected['row_id'].shape!=scores.shape or expected['unit'].shape!=scores.shape or len(np.unique(expected['row_id']))!=len(scores):raise ValueError('restored independent score person dimensions differ')
            if not np.array_equal(scores,expected['score']):raise AssertionError('independently replayed restored reader score differs')
            checks.append({'reader':reader['reader'],'view':reader.get('view','pair'),'orientation':reader['orientation'],'model_sha256':reader['model']['sha256'],'expected_score_sha256':reader['expected']['sha256'],'max_absolute_score_difference':0.,'status':'PASS'})
        records.append({'arm_id':item['arm_id'],'seed':item['seed'],'release_variant':item['release_variant'],'rows':len(wires[0]),'max_absolute_wire_difference':[0.,0.],'deployment_sha256':item['deployment']['sha256'],'model_sha256':item['model_state']['sha256'],'reader_checks':checks,'status':'PASS'})
    return {'schema':'OB1-independent-backup-only-replay-v1','status':'PASS','records':records,'scientific_runner_imports':False,'runtime_code_from_restored_tree':True,'all_numerical_sources_from_restored_tree':True,'original_model_or_reader_sources_used':False,'labels_read_or_hashed':False,'new_fits':0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=RestoredInputs(args.spec);result=run(inputs)
    output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'independent_restored_representatives':len(result['records'])}))


if __name__=='__main__':main()
