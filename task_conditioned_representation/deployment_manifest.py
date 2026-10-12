"""Metadata-only seal of finished live deployment receipts; never opens rows."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .build_deployment_units import BASE,pin,write


def seal(index_path,private_output,public_output):
    index=json.loads(Path(index_path).read_text());state=json.loads((BASE/'.private/task_conditioned_representation/compute/state.json').read_text());entries=[];public=[];versions=None
    for item in index['units']:
        spec=json.loads(Path(item['spec']['path']).read_text());uid=spec['unit_id'];unit=state['units'].get(uid)
        if not unit or unit['status']!='done':raise PermissionError('all actual live deployment jobs must finish: '+uid)
        receipt_path=Path(item['public']);receipt_pin=pin(receipt_path);r=json.loads(receipt_path.read_text())
        if unit['output_sha256'].get(str(receipt_path))!=receipt_pin['sha256'] or r.get('status')!='PASS' or r['arm_id']!=item['arm_id'] or r['seed']!=item['seed']:raise PermissionError('live receipt or queue output binding differs')
        if r['max_abs_wire_difference']!=[0.,0.] or not r['standalone_CLI_executed'] or not r['standalone_CLI_exact_parity'] or r['true_labels_or_SEX_read']:raise PermissionError('deployment or standalone CLI parity not established')
        if r['full_dimensions']!=[21,29,50] or r['score_dimensions']!=[3,7,10]:raise PermissionError('actual release contract differs')
        if versions is None:versions=r['environment_pins']
        if versions!=r['environment_pins']:raise PermissionError('deployment environment changed across frozen models')
        package=Path(item['package']);manifest=pin(package/'DEPLOYMENT.json');weights=pin(package/'model_state.pt')
        if manifest['sha256']!=r['package_manifest_sha256'] or weights['sha256']!=r['model_sha256']:raise PermissionError('deployed package changed after numerical verification')
        details={'arm_id':item['arm_id'],'seed':item['seed'],'unit_id':uid,'actual_model_point_eligible':r['actual_model_point_eligible'],'historical_diagnostic_reference':r['historical_diagnostic_reference'],'exact_source_alias':r['exact_source_alias'],'package_manifest_sha256':manifest['sha256'],'model_state_sha256':weights['sha256'],'selected_checkpoint_sha256':r['source_checkpoint_sha256'],'rows':r['rows'],'live_receipt_sha256':receipt_pin['sha256'],'input_sha256':r['input_sha256'],'expected_sha256':r['expected_sha256'],'standalone_CLI_output_sha256':r['standalone_CLI_output_sha256'],'max_abs_wire_difference':r['max_abs_wire_difference'],'standalone_CLI_exact_parity':True,'public_receipt':str(receipt_path.relative_to(BASE))}
        public.append(details);entries.append({**details,'package':str(package),'manifest':manifest,'weights':weights,'input':item['input'],'expected':item['expected'],'receipt':receipt_pin})
    result={'schema':'TCR1-all-actual-live-deployment-seal-v1','status':'PASS','instances':len(entries),'point_eligible_instances':sum(e['actual_model_point_eligible'] for e in entries),'historical_diagnostic_instances':sum(e['historical_diagnostic_reference'] for e in entries),'exact_source_alias_instances':sum(e['exact_source_alias'] for e in entries),'environment_pins':versions,'source_preparation_index':pin(index_path),'entries':entries,'numeric_person_arrays_opened_by_sealer':False,'full_wire_dimensions':[21,29,50],'score_wire_dimensions':[3,7,10],'standalone_model_dependencies':['numpy','torch'],'reader_restore_dependencies':['numpy','scipy','scikit-learn','joblib','task_conditioned_representation.attacks','task_conditioned_representation.inference'],'restore_status':'PENDING_VERSIONED_BACKUP_AND_NUMERICAL_RESTORE'}
    private=write(private_output,result)
    view={k:v for k,v in result.items() if k not in ('source_preparation_index','entries')};view['entries']=public;view['private_runtime_index_sha256']=private['sha256'];view['source_preparation_index_sha256']=result['source_preparation_index']['sha256']
    return write(public_output,view)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('index');p.add_argument('private_output');p.add_argument('public_output');a=p.parse_args();print(json.dumps(seal(a.index,a.private_output,a.public_output)))
