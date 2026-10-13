"""Aggregate already public receipts; never load person arrays or fit models."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
    temp.replace(path)

def run(root, queue):
    frontier=[]; diagnostics=[]; certificates=[]; counts={'policy_lp_solve_calls':0,'control_solver_calls':0,'fresh_reader_fits':0}
    for path in sorted(root.glob('r[345]/*RECEIPT.json')):
        d=json.loads(path.read_text())
        seed=d.get('seed',d.get('source_seed'))
        counts['policy_lp_solve_calls']+=d.get('lp_solve_calls',0)
        counts['control_solver_calls']+=d.get('solver_calls',0)
        if 'setups' in d or 'arms' in d or ('arm_id' in d and 'solver_calls' not in d):
            entries=d.get('setups',d.get('arms',[d]))
            for x in entries:
                arm=x.get('arm_id') or f"{x['mode']}_{x['budget']}"
                base={'arm_id':arm,'seed':seed,'budget':x.get('budget','HISTORICAL_REFERENCE'),'support':x.get('support','FROZEN_DONOR'),
                      'status':x.get('status','COMPLETE'),'point_eligible':x.get('inner_point_eligible'),
                      'fitted_objective':x.get('fitted_objective'),'actual_change_rate':x.get('actual_change_rate'),
                      'maximum_actual_change':x.get('maximum_actual_change'),'receipt':str(path.relative_to(root)),'receipt_sha256':sha(path)}
                diagnostics.append(base)
                for clause in x.get('inner_point_utility',[]):
                    frontier.append({**base,**clause,'excess_cost':clause['margin'],'headroom':clause['cap']-clause['margin']})
                certificate={'arm_id':arm,'seed':seed,'status':x.get('status','COMPLETE'),'receipt_sha256':sha(path),
                             'scope':'fixed empirical law and registered constraints only',
                             'full_declared_support_columns':x.get('full_declared_support_columns')}
                for key in ('protection','infeasibility'):
                    value=x.get(key)
                    if isinstance(value,dict):
                        c=value.get('certificate',value)
                        certificate[key]={k:c[k] for k in ('scope','primal_objective','dual_objective','primal_dual_gap','maximum_primal_violation','maximum_dual_violation','dual_lower_bound','success','feasible','status','max_primal_violation','max_dual_violation','duality_gap') if k in c}
                certificates.append(certificate)
        elif 'kind' in d and 'solver_calls' in d:
            base={'arm_id':d.get('arm_id',{'DETERMINISTIC':'D_SELECTED','LOCAL':'L_SELECTED','PRODUCT':'P_SELECTED','W_SEQ12':'S12_SELECTED','W_SEQ21':'S21_SELECTED'}.get(d['kind'],d['kind'])),'seed':d['seed'],'budget':d['budget'],'support':d['support'],'status':d['status'],
                  'point_eligible':d.get('inner_point_eligible'),'fitted_objective':d.get('fitted_objective'),'actual_change_rate':d.get('actual_change_rate'),
                  'maximum_actual_change':None,'receipt':str(path.relative_to(root)),'receipt_sha256':sha(path)}
            diagnostics.append(base)
            for clause in d.get('inner_point_utility',[]):frontier.append({**base,**clause,'excess_cost':clause['margin'],'headroom':clause['cap']-clause['margin']})
            certificates.append({**base,'solver_class':d['solver_class'],'solver_calls':d['solver_calls'],'global_optimality_claimed':False,
                                 'remaining_gap':d.get('exact_remaining_global_gap'),'heuristic_gap_upper_bound':d.get('heuristic_global_gap_upper_bound'),
                                 'scope':'bounded registered control procedure; independent certificates supersede status strings'})
    def csvwrite(path, rows):
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    for path in sorted(root.glob('r4/*FAILED_UNIT_ACCOUNT.json')):
        d=json.loads(path.read_text())
        counts['control_solver_calls']+=d['scientific_optimizer_calls']
        certificates.append({'unit_id':d['unit_id'],'status':d['status'],'solver_calls':d['scientific_optimizer_calls'],
                             'receipt':str(path.relative_to(root)),'receipt_sha256':sha(path),
                             'scope':'Preserved optimizer calls from failed engineering attempt; no deployed law or privacy result'})
    csvwrite(root/'FIT_FRONTIER.csv',frontier)
    csvwrite(root/'ALLOCATION_DIAGNOSTICS.csv',diagnostics)
    write(root/'SOLVER_CERTIFICATES.json',{'schema':'CBR1-aggregate-solver-index-v1','certificates':certificates,
          'independent_reports_directory':'r6','note':'See independent reports for complete reconstructed margins and valid bound gaps. No population guarantee.'})
    state=json.loads(queue.read_text());units=[]
    for uid,u in sorted(state['units'].items()):
        receipt=queue.parent/'units'/uid/'RECEIPT.json';actual=json.loads(receipt.read_text()) if receipt.is_file() else {}
        units.append({'unit_id':uid,'stage':u['spec']['stage'],'status':u['status'],'spec_sha256':u['spec_sha256'],
                      'returncode':actual.get('returncode'),'charged_cpu_s':actual.get('charged_cpu_s',0),
                      'max_wall_s':u['spec']['max_wall_s'],'reserved_cpu_s':u['spec']['cpu_reserve_s'],
                      'observed_peak_rss_bytes':actual.get('sampled_tree_rss_peak_bytes'),'receipt_sha256':sha(receipt) if receipt.is_file() else None})
    # Reader compound receipts use a different naming convention.
    for path in root.glob('r5/R5_FIT_*.json'):
        d=json.loads(path.read_text());counts['fresh_reader_fits']+=d.get('numerical_fit_count',d.get('fit_count',d.get('fits',0)))
    charges={}
    for name, value in state.get('charges',{}).items():
        charges[name]={k:v for k,v in value.items() if k!='receipt_path'}
        if 'receipt_path' in value:charges[name]['receipt_alias']=Path(value['receipt_path']).name
    write(root/'UNIT_MANIFEST.json',{'schema':'CBR1-complete-unit-accounting-v1','as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'units':units,'scientific_call_counts_from_actual_receipts':counts,'charged_cpu_s':state['used_cpu_s'],
          'used_by_stage_s':state['used_by_stage_s'],'light_charges':charges,'deadline_utc':state['deadline_utc'],
          'note':'Numerical control refusals and engineering attempts remain recorded. Conditional untriggered scientific slots appear in the registered stage indexes.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);a=p.parse_args();run(a.root,a.queue)
