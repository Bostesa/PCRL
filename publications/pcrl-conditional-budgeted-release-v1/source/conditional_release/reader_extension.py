"""Replay admitted frozen local safeguards on any original-bank action support.

No reader is fitted here. The source OOF models/folds remain frozen. Labels are
allowed FIT SEX, supplied by the caller solely to evaluate inherited CE guards.
"""
from __future__ import annotations
import json
from zipfile import ZipFile
import numpy as np
from .source import pinned,read_exact


def frozen_local_loss_tables(manifest,seed,space,folds,sex):
    from incumbent_view.r2_readers import wire_features,categorical_probability
    from incumbent_view.r2_package import array_sha
    record=manifest['seeds'][str(seed)]['fixed_readers']
    index=json.loads(pinned(record['index']).read_text())
    with ZipFile(pinned(record['arrays'])) as z:
        keys={name[:-4] for name in z.namelist() if name.endswith('.npy')}
        if len(keys)!=len(z.namelist()):raise PermissionError('nonarray source guard archive')
    arrays=read_exact(record['arrays'],keys)
    sex=np.asarray(sex);folds=np.asarray(folds)
    if sex.dtype!=np.int64 or sex.shape!=folds.shape or not np.isin(sex,(0,1)).all():
        raise ValueError('FIT guard SEX alignment differs')
    result={};catalog=[]
    for i,reader in enumerate(index['readers']):
        view=reader['view']
        if view not in ('r1','r2'):continue
        family=reader['name'].rsplit('/',1)[1];width=len(getattr(space,'probability'+view[-1]))
        feature=wire_features(space,view,np.arange(width,dtype=np.int64))
        table=np.empty((3,2,width),np.float64)
        for fold in range(3):
            prefix=f'r{i}_f{fold}'
            if family=='prior':
                prediction=np.full(width,float(arrays[prefix][0]),np.float64)
            elif family=='categorical':
                count=arrays[prefix]
                # The source prior is based on raw people, not action count
                # accumulation roundoff. All prior reader in this same round
                # has the identical training-fold prior.
                prior_reader=next(j for j,r in enumerate(index['readers'])
                                  if r['name'].rsplit('/',1)[0]==reader['name'].rsplit('/',1)[0]
                                  and r['name'].endswith('/prior'))
                prior=float(arrays[f'r{prior_reader}_f{fold}'][0])
                prediction=categorical_probability(count,prior)
            elif family=='nonlinear':
                xx=(feature-arrays[prefix+'_mean'])/arrays[prefix+'_scale']
                h=np.maximum(xx@arrays[prefix+'_w0']+arrays[prefix+'_w1'],0)
                h=np.maximum(h@arrays[prefix+'_w2']+arrays[prefix+'_w3'],0)
                logit=(h@arrays[prefix+'_w4']+arrays[prefix+'_w5']).ravel()
                prediction=np.empty(width);positive=logit>=0
                prediction[positive]=1/(1+np.exp(-logit[positive]))
                e=np.exp(logit[~positive]);prediction[~positive]=e/(1+e)
            else:raise ValueError('unregistered source fixed-reader family')
            prediction=np.clip(prediction,1e-12,1-1e-12)
            table[fold,0]=-np.log1p(-prediction);table[fold,1]=-np.log(prediction)
        result[reader['name']]={'view':view,'table':table,'source_index':i}
        catalog.append({'name':reader['name'],'view':view,'source_model_arrays':reader['model_arrays'],
                        'source_loss_sha256':reader['loss_sha256']})
    return result,catalog,arrays,index


class ActionLosses:
    """One-reader-at-a-time safeguard matrices; bounded WIDE peak memory."""
    def __init__(self,tables,emitted1,emitted2,folds,sex):
        self.tables=tables;self.emitted1=emitted1;self.emitted2=emitted2;self.folds=folds;self.sex=sex
    def items(self):
        for name,item in self.tables.items():
            yield name,item['table'][self.folds[:,None],self.sex[:,None],self.emitted1 if item['view']=='r1' else self.emitted2]
    def values(self):
        for _,value in self.items():yield value
    def __len__(self):return len(self.tables)
    def __getitem__(self,name):
        item=self.tables[name]
        return item['table'][self.folds[:,None],self.sex[:,None],self.emitted1 if item['view']=='r1' else self.emitted2]


def losses_for_actions(tables,emitted1,emitted2,folds,sex):
    return ActionLosses(tables,emitted1,emitted2,folds,sex)


def replay_loss_parity(tables,source_arrays,source_index,prepared,folds,sex,tolerance=1e-12):
    replay=losses_for_actions(tables,prepared['code_r1'],prepared['code_r2'],folds,sex)
    max_gap=0.
    for name,item in tables.items():
        frozen=source_arrays[f"r{item['source_index']}_loss"]
        gap=float(np.max(np.abs(replay[name]-frozen)));max_gap=max(max_gap,gap)
        if gap>tolerance:raise PermissionError('frozen source safeguard loss replay mismatch: '+name)
    return {'scope':'all source retained local reader OOF losses on original16actions',
            'reader_count':len(replay),'positions':sum(v.size for v in replay.values()),
            'maximum_absolute_loss_gap':max_gap,'tolerance':tolerance,'status':'PASS'}
