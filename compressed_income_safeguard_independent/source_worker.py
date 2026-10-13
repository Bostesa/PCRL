"""Independent source folding and actual compressed wire admission."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(inputs):
    import numpy as np
    from .replay import source_audit,torch_load,forward,geometry,close
    cfg=inputs.cfg;admission=source_audit(inputs,cfg['source_admission'])
    manifest=inputs.json(cfg['source_admission']['manifest']);receipt=inputs.json(cfg['source_receipt'])
    roster=inputs.json(cfg['source_admission']['roster']);records=[];aliases=[]
    for entry in roster['entries']:
        if not entry['exists']:raise ValueError('required source checkpoint absent')
        payload=torch_load(inputs.path({'path':entry['checkpoint'],'sha256':entry['checkpoint_sha256']}))
        binding=receipt['geometry_bindings'][entry['unit_id']]
        # Some public inventories wrap bindings together with file pins.
        binding=binding.get('binding',binding)
        stem='U0' if entry['unit_id'].startswith('U0_') else 'RAW'
        variants=(('full','full'),('head_row_space','head_row_space')) if stem=='U0' else (
            ('full','full'),('head_row_space','full'),('head_row_space','head_row_space'))
        for role,pins in manifest['roles'].items():
            f=inputs.arrays(pins['features'],{'row_id','unit','X','feature_names'})
            if tuple(payload['feature_names'])!=tuple(map(str,f['feature_names'])):raise ValueError('source feature order differs')
            raw=forward(payload,f['X']);checks=[]
            for i in (0,1):
                g=geometry(payload['model'][f'head.{i}.weight'].numpy(),payload['model'][f'head.{i}.bias'].numpy(),raw[i][:,:16],cfg['rule'])
                close(g['B'],binding['heads'][i]['B'],'independent admitted B',1e-10)
                if g['decision_mismatch_rows'] or g['near_tie_mismatch_rows'] or g['AP_max_absolute_residual']>1e-10 or g['max_absolute_probability_difference']>1e-10 or g['max_absolute_logit_difference']>1e-10:
                    raise AssertionError('admitted own-head projection parity')
                checks.append({k:v for k,v in g.items() if k not in ('B','P','projected_z')})
            wires={}
            for rules in variants:
                contract=dict(zip(('income','occupation'),rules));wires[rules]=forward(payload,f['X'],wire_contract=contract,geometry_binding=binding)
                for i in (0,1):
                    if not np.array_equal(wires[rules][i][:,-1],raw[i][:,-1]):raise AssertionError('actual Torch projected decision mismatch')
                    close(wires[rules][i][:,16:-1],raw[i][:,16:-1],'actual projected Torch logits/probabilities',1e-10)
            if stem=='RAW':
                if not np.array_equal(wires['head_row_space','full'][0],wires['head_row_space','head_row_space'][0]):raise AssertionError('RAW_HYBRID/RAW_PROJ income alias differs')
                if not np.array_equal(wires['head_row_space','full'][1],wires['full','full'][1]):raise AssertionError('RAW_HYBRID/RAW_FULL occupation alias differs')
                aliases.append({'seed':entry['seed'],'role':role,'RAW_HYBRID_RAW_PROJ_income':'EXACT',
                                'RAW_HYBRID_RAW_FULL_occupation':'EXACT'})
            records.append({'unit_id':entry['unit_id'],'seed':entry['seed'],'role':role,
                            'rows':len(f['X']),'projection_checks':checks,'wire_variant_count':len(variants),
                            'source_checkpoint_sha256':entry['checkpoint_sha256'],'status':'PASS'})
    return {'schema':'CIS1-independent-source-admission-v1','status':'PASS','source_head_folding':admission,
            'actual_wire_records':records,'literal_aliases':aliases,'source_role_forward_count':len(records),
            'new_role_batch_size':1024,'legacy_U0_reference_batching':'whole role',
            'scientific_runner_imports':False,'INNER_truth_opened_or_hashed':False}


def main():
    from .replay import Inputs
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=Inputs(args.spec)
    result=run(inputs);output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','source_role_forward_count':result['source_role_forward_count']}))

if __name__=='__main__':main()
