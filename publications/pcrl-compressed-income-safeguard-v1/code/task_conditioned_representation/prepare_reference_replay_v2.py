"""Metadata-only immutable V2 jobs for the original-reference key correction."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .prepare_evaluation_units import BASE,PRIVATE,PUBLIC,COMMON,MODEL,LEGACY,pin,read,write,make_unit,source
from .prepare_reporting_units import make


def clone_sources(old_spec,old_config):
    return [source(i) for i in old_spec['sources'] if i['path']!=old_config['path'] and i['path']!=str(Path(old_spec['cwd'])/'task_conditioned_representation/prepare_reference_replay_v2.py')]


def prepare_utility(output,python,amendment):
    old_cfg=pin(PRIVATE/'configs/R5_ORIGINAL_UTILITY_ALL40_V1.json');cfg=read(old_cfg);old_spec=read(pin(PRIVATE/'specs/R5_ORIGINAL_UTILITY_ALL40_V1.json'))
    uid='R5_ORIGINAL_UTILITY_ALL40_V2';cfg.update(unit_id=uid,private_output=str(PRIVATE/'utility'/uid),public_output=str(PUBLIC/f'{uid}.json'))
    # Scientific source pins refresh to the corrected code; role/checkpoint,
    # nomination and all other scientific inputs retain their original pins.
    science_code={str((BASE/p).resolve()) for p in (*COMMON,*MODEL,'task_conditioned_representation/utility_worker.py')}
    inputs=[i for i in clone_sources(old_spec,old_cfg) if i['path'] not in science_code]
    unit=make_unit(cfg,'task_conditioned_representation.utility_worker',inputs=inputs,code=(*COMMON,*MODEL,'task_conditioned_representation/utility_worker.py','task_conditioned_representation/prepare_reference_replay_v2.py'),python=python,amendment_pin=amendment,wall=600)
    return write(output,{'schema':'TCR1-prepared-original-reference-utility-v2','unit':unit,'old_config':old_cfg,'original_V1_preserved':True,'executed':False})


def prepare_consumers(output,python,amendment):
    utility=read(pin(PRIVATE/'utility/R5_ORIGINAL_UTILITY_ALL40_V2/UTILITY_COMPLETE.json'))
    inputs_pin={'path':str(PRIVATE/'utility/R5_ORIGINAL_UTILITY_ALL40_V2/UTILITY_INPUTS.npz'),'sha256':utility['utility_input_sha256']}
    old_legacy_cfg=pin(PRIVATE/'configs/R5_LEGACY_ORIGINAL_UTILITY_V1.json');legacy_cfg=read(old_legacy_cfg);old_legacy_spec=read(pin(PRIVATE/'specs/R5_LEGACY_ORIGINAL_UTILITY_V1.json'))
    uid='R5_LEGACY_ORIGINAL_UTILITY_V2';old_inputs=legacy_cfg['utility_inputs']['path'];legacy_cfg.update(unit_id=uid,utility_inputs=inputs_pin,private_output=str(PRIVATE/'legacy_utility/LEGACY_ORIGINAL_UTILITY_V2.json'),public_output=str(PUBLIC/'LEGACY_ORIGINAL_UTILITY_V2.json'))
    code=(*COMMON,*LEGACY,'task_conditioned_representation/utility_worker.py','task_conditioned_representation/legacy_utility_worker.py','task_conditioned_representation/prepare_reference_replay_v2.py')
    code_paths={str((BASE/p).resolve()) for p in code};legacy_inputs=[i for i in clone_sources(old_legacy_spec,old_legacy_cfg) if i['path'] not in code_paths and i['path']!=old_inputs]
    legacy=make_unit(legacy_cfg,'task_conditioned_representation.legacy_utility_worker',inputs=[*legacy_inputs,inputs_pin],code=code,python=python,amendment_pin=amendment,wall=180,deps=['R5_ORIGINAL_UTILITY_ALL40_V2'])
    old_infer_cfg=pin(PRIVATE/'configs/R5_FINAL_PAIRED_INFERENCE_V1.json');infer_cfg=read(old_infer_cfg);old_infer_spec=read(pin(PRIVATE/'specs/R5_FINAL_PAIRED_INFERENCE_V1.json'))
    uid='R5_FINAL_PAIRED_INFERENCE_V2';old_infer_inputs=infer_cfg['utility_inputs']['path'];infer_cfg.update(unit_id=uid,utility_inputs=inputs_pin,private_output=str(PRIVATE/'inference'/uid),public_output=str(PUBLIC/'FINAL_PAIRED_INFERENCE_V2.json'))
    code=(*COMMON,*LEGACY,'conditional_release/inference.py','task_conditioned_representation/utility_worker.py','task_conditioned_representation/inference_worker.py','task_conditioned_representation/prepare_reference_replay_v2.py')
    code_paths={str((BASE/p).resolve()) for p in code};infer_inputs=[i for i in old_infer_spec['sources'] if i['path'] not in code_paths and i['path'] not in (old_infer_cfg['path'],old_infer_inputs)]
    infer=make(infer_cfg,'task_conditioned_representation.inference_worker',[*infer_inputs,inputs_pin],code,python,amendment,3600,deps=['R5_ORIGINAL_UTILITY_ALL40_V2','R5_LEGACY_ORIGINAL_UTILITY_V2'])
    return write(output,{'schema':'TCR1-prepared-original-reference-consumers-v2','legacy_utility':legacy,'inference':infer,'original_V1_preserved':True,'fitted_models_readers_nominations_caps_K_changed':False,'executed':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('utility','consumers'),required=True);p.add_argument('--output',required=True);p.add_argument('--python',required=True);p.add_argument('--attacks-amendment',required=True);a=p.parse_args()
    print(json.dumps((prepare_utility if a.phase=='utility' else prepare_consumers)(a.output,a.python,pin(a.attacks_amendment))))
if __name__=='__main__':main()
