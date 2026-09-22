"""Public aggregate evidence from frozen accepted JSON summaries only.

Collection is explicit: importing this module never discovers outcomes. No
NPZ, model, raw-person, or prediction files are opened. Artifact uniqueness
counts are receipt declarations, not proofs of equal mathematical channels.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re

from . import config, reporting, run
from .selection import default_families

ANCHORS = (0, 1, 2)
WEIGHTS = {'unweighted': 'unweighted', 'PWGTP': 'weighted'}
TASK = 'utility:A/same_residence'
ROLES = tuple('attack:' + r for r in config.PRIMARY + config.SECONDARY) + tuple('utility:' + r for r in config.UTILITY)
HISTORICAL = ('J', 'leace_A0', 'splince_A0', 'optnet16_L1', 'optnet16_L2', 'optnet16_C1')
ENTRY_KEYS = {'ce', 'accuracy', 'support', 'weight_sum', 'ess', 'per_class', 'selection',
              'independent_selection', 'independent_ce', 'selection_source', 'fixed_decoder_ce'}
VERSION_SCOPE = 'Receipt-only provenance/byte checks; no score, model or prediction replay.'
VERSION_HASHES = ('original_map_receipt_sha256', 'original_solution_sha256', 'original_audit_receipt_sha256',
    'original_registry_sha256', 'claim_sha256', 'retry_receipt_sha256', 'installation_receipt_sha256',
    'current_map_receipt_sha256', 'current_solution_sha256', 'current_audit_receipt_sha256', 'current_registry_sha256')
VERSION_COUNTS = tuple(prefix+field for prefix in ('original_', 'current_')
    for field in ('audit_role_units', 'new_role_fits', 'reused_role_audits'))


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError('Invalid registered configuration name')
    return value


def _anchor(value):
    if isinstance(value, bool) or str(value) not in ('0', '1', '2'):
        raise ValueError('Expected one of the three registered anchors')
    return int(value)


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('Invalid SHA-256 provenance')
    return value


def _number(value, *, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Aggregate must be finite numeric data')
    if nonnegative and value < 0:raise ValueError('Aggregate cannot be negative')
    return float(value)


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Count must be a nonnegative integer')
    return value


def _scores(value, *, empty=False):
    if not isinstance(value, dict) or set(value) != {'unweighted', 'weighted', 'balanced'}:
        raise ValueError('Expected native unweighted/weighted/balanced CE schema')
    if empty:
        if any(v is not None for v in value.values()):raise ValueError('Absent classes must have unavailable CE')
    else:
        for v in value.values():_number(v)
        if not math.isclose(value['balanced'], .5*(value['unweighted']+value['weighted']), rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Balanced CE disagrees with the two report weightings')


def _native(native, choices):
    """Strict native allowlist prevents accidental person-level publication."""
    if not isinstance(native, dict) or set(native) != {'test'} or set(native['test']) != set(ROLES):
        raise ValueError('Native evaluation summary requires test and all sixteen registered roles')
    if set(choices) != set(ROLES):raise ValueError('Frozen predictor role schema differs')
    for role, entry in native['test'].items():
        if not isinstance(entry, dict) or set(entry)-ENTRY_KEYS:
            raise ValueError('Nonpublic or unexpected native summary field')
        if not (ENTRY_KEYS-{'fixed_decoder_ce'}) <= set(entry):raise ValueError('Incomplete evaluation aggregate')
        for field in ('selection', 'independent_selection'):
            if entry[field] != choices[role].get(field) or not isinstance(entry[field], str):
                raise ValueError('Evaluation predictor differs from frozen validation choice')
        if entry['selection_source'] != 'frozen validation registry; no evaluation selection':
            raise ValueError('Unknown evaluation selection source')
        for field in ('ce', 'independent_ce'):_scores(entry[field])
        if 'fixed_decoder_ce' in entry:
            if role != TASK:raise ValueError('Fixed residence decoder assigned to another role')
            _scores(entry['fixed_decoder_ce'])
        n = _count(entry['support']); weight = _number(entry['weight_sum']); ess = _number(entry['ess'])
        if not n or not weight or not 0 < ess <= n+1e-8:raise ValueError('Invalid observed-label support')
        if set(entry['accuracy']) != {'unweighted','weighted'} or any(not 0 <= _number(v) <= 1 for v in entry['accuracy'].values()):
            raise ValueError('Invalid aggregate accuracy')
        classes = entry['per_class']; nc = 9 if role.endswith('/RAC1P') else 2
        if not isinstance(classes,list) or [c.get('class') for c in classes] != list(range(nc)):
            raise ValueError('Per-class support must retain the entire ordered class schema')
        for c in classes:
            if set(c) != {'class','support','weight_sum','ess','ce'}:raise ValueError('Unexpected per-class field')
            count = _count(c['support']); cw = _number(c['weight_sum']); ce = _number(c['ess'])
            if (count == 0 and (cw != 0 or ce != 0)) or (count > 0 and (cw <= 0 or not 0 < ce <= count+1e-8)):
                raise ValueError('Inconsistent per-class support/weight/ESS')
            _scores(c['ce'], empty=count == 0)
        if sum(c['support'] for c in classes) != n or not math.isclose(math.fsum(c['weight_sum'] for c in classes), weight, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Per-class support does not partition observed-label support')
    return copy.deepcopy(native)


def _ledger(records=None):
    records = config.release_ledger()['records'] if records is None else records
    if isinstance(records,dict):records=records['records']
    result = {}
    for record in records:
        key = (_name(record['configuration']), _anchor(record['anchor']))
        if key in result:raise ValueError('Duplicate nominal release/anchor ledger unit')
        result[key] = {'configuration':key[0], 'anchor':key[1], 'kind':record.get('kind','control'),
                       'scheduled':record.get('scheduled',True)}
    return result


def _receipts(values, *, fields, extra=()):
    result = {}
    for value in values:
        if set(value)-{'configuration','anchor',*fields,*extra}:raise ValueError('Unexpected nonpublic receipt field')
        key = (_name(value['configuration']), _anchor(value['anchor']))
        if key in result:raise ValueError('Duplicate accepted receipt')
        record = {'configuration':key[0], 'anchor':key[1]}
        for field in fields:record[field] = _hash(value[field])
        for field in extra:record[field] = _count(value[field])
        result[key] = record
    return result


def _empty_numerical_versions():
    return {'schema':1,'registered':False,'registration_sha256':None,'registered_slots':0,
        'new_nominal_configurations':0,'coverage_complete':True,'records':[],'scope':VERSION_SCOPE}


def _public_numerical_versions(value):
    value = _empty_numerical_versions() if value is None else value
    if (not isinstance(value,dict) or set(value)!=set(_empty_numerical_versions())
            or value['schema']!=1 or type(value['registered']) is not bool or value['coverage_complete'] is not True
            or value['new_nominal_configurations']!=0 or value['scope']!=VERSION_SCOPE
            or not isinstance(value['records'],list) or _count(value['registered_slots'])!=len(value['records'])):
        raise ValueError('Invalid public numerical-version schema/completion')
    if value['registered']:_hash(value['registration_sha256'])
    elif value['records'] or value['registration_sha256'] is not None:
        raise ValueError('Unregistered numerical versions')
    allowed={'configuration','anchor','attempts','retry_completed','retry_accepted','installation_decision','reaudit_completed',
             *VERSION_HASHES,*VERSION_COUNTS}
    seen=set()
    for row in value['records']:
        if not isinstance(row,dict) or set(row)!=allowed:raise ValueError('Unexpected nonpublic numerical-version field')
        key=(_name(row['configuration']),_anchor(row['anchor']))
        if key in seen:raise ValueError('Duplicate numerical version slot')
        seen.add(key)
        installed=row['installation_decision']=='installed_accepted_retry'
        if (row['installation_decision'] not in ('installed_accepted_retry','retained_original')
                or row['attempts']!=1 or row['retry_completed'] is not True
                or row['retry_accepted'] is not installed or row['reaudit_completed'] is not installed):
            raise ValueError('Incomplete or contradictory numerical version decision')
        for field in VERSION_HASHES:_hash(row[field])
        for field in VERSION_COUNTS:_count(row[field])
        for prefix in ('original_','current_'):
            if row[prefix+'audit_role_units']!=len(ROLES) or row[prefix+'new_role_fits']+row[prefix+'reused_role_audits']!=len(ROLES):
                raise ValueError('Numerical audit role counts differ from the registered slate')
    return copy.deepcopy(value)


def collect_numerical_versions(*,out_root=config.OUT):
    """Require completed Branch-D decisions and re-audits; read only JSON receipts.

    Original/current model and solution hashes are accepted receipt declarations;
    their weights are deliberately left to independent replay, not opened here.
    """
    root=Path(out_root).resolve();path=root/'NUMERICAL_RECOVERY.json'
    if not path.exists():return _empty_numerical_versions()
    from . import numerical_recovery_run as retry
    def read(path, *, expected=None, identity=None, stage=None):
        path=Path(path)
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Required numerical completion receipt missing or escaping root')
        sha=run.sha(path)
        if expected is not None and sha!=expected:raise ValueError('Numerical receipt hash mismatch')
        row=json.loads(path.read_text())
        if identity is not None and (row.get('anchor'),row.get('configuration'))!=identity:
            raise ValueError('Numerical receipt identity mismatch')
        if stage is not None:run._validate_provenance(row,stage)
        return row,sha
    registry,pin=read(path)
    if (registry.get('schema')!=1 or registry.get('branch')!='D' or registry.get('attempts_per_slot')!=1
            or registry.get('new_nominal_configurations')!=0
            or [(s['anchor'],s['configuration']) for s in registry.get('slots',[])]!=list(retry.SLOTS)):
        raise ValueError('Numerical registration differs from the four original slots')
    if (registry.get('config_hash')!=config.digest(config.configuration())
            or registry.get('scientific_source_hashes')!=run.source_fingerprint()
            or registry.get('runner_sha256')!=run.sha(run.__file__)
            or registry.get('recovery_source_hashes')!=retry._recovery_sources()
            or registry.get('resource_schedule_sha256')!=run.sha(root/'RESOURCE_SCHEDULE.json')):
        raise ValueError('Numerical registration source/schedule provenance changed')
    records=[]
    for slot in registry['slots']:
        anchor,name=slot['anchor'],slot['configuration'];identity=(anchor,name)
        stage=root/'private/numerical_recovery'/f'anchor_{anchor}'/name
        active=root/'private/run'/f'anchor_{anchor}'
        claim,claim_sha=read(stage/'CLAIM.json',identity=identity)
        complete,complete_sha=read(stage/'COMPLETE.json',identity=identity)
        installation,install_sha=read(stage/'INSTALLATION.json',identity=identity)
        if (claim.get('registration_sha256')!=pin or claim.get('attempt')!=1
                or complete.get('registration_sha256')!=pin or complete.get('attempts')!=1
                or type(complete.get('accepted')) is not bool
                or installation.get('retry_receipt_sha256')!=complete_sha):
            raise ValueError('Numerical claim/completion/installation provenance mismatch')
        installed=installation.get('decision')=='installed_accepted_retry'
        if installed != complete['accepted'] or installation.get('decision') not in ('installed_accepted_retry','retained_original'):
            raise ValueError('Numerical installation has no completed accepted/retained decision')
        old=(root/'private/numerical_originals'/f'anchor_{anchor}'/name) if installed else None
        original_map,old_map_sha=read(old/'map/ACCEPTED.json' if installed else active/'maps'/name/'ACCEPTED.json',
            expected=slot['original']['receipt_sha256'],identity=identity,stage='map')
        original_audit,old_audit_sha=read(old/'audit/COMPLETE.json' if installed else active/'audits'/name/'COMPLETE.json',
            expected=slot['audit']['receipt_sha256'],identity=identity,stage='audit')
        current_map,map_sha=read(active/'maps'/name/'ACCEPTED.json',identity=identity,stage='map')
        current_audit,audit_sha=read(active/'audits'/name/'COMPLETE.json',identity=identity,stage='audit')
        if (original_map.get('sha256')!=slot['original']['solution_sha256']
                or original_audit.get('map_solution_sha256')!=original_map['sha256']
                or original_audit.get('registry_sha256')!=slot['audit']['registry_sha256']
                or current_audit.get('map_solution_sha256')!=current_map.get('sha256')):
            raise ValueError('Numerical map/audit dependency mismatch')
        if installed:
            expected={'registration_sha256':pin,'retry_receipt_sha256':complete_sha,
                'original_accepted_sha256':old_map_sha,'original_solution_sha256':original_map['sha256'],
                'recovery_source_hashes':registry['recovery_source_hashes']}
            if (complete.get('feasible') is not True or current_map.get('numerical_retry')!=expected
                    or any(installation.get(k)!=v for k,v in expected.items())
                    or installation.get('new_solution_sha256')!=current_map['sha256']
                    or installation.get('new_accepted_sha256')!=map_sha or installation.get('reaudit_required') is not True
                    or complete.get('artifact_hashes',{}).get('map/solution.joblib')!=current_map['sha256']):
                raise ValueError('Installed numerical version/re-audit receipt is inconsistent')
        row={'configuration':name,'anchor':anchor,'attempts':1,'retry_completed':True,
            'retry_accepted':complete['accepted'],'installation_decision':installation['decision'],'reaudit_completed':installed,
            'original_map_receipt_sha256':old_map_sha,'original_solution_sha256':original_map['sha256'],
            'original_audit_receipt_sha256':old_audit_sha,'original_registry_sha256':original_audit['registry_sha256'],
            'claim_sha256':claim_sha,'retry_receipt_sha256':complete_sha,'installation_receipt_sha256':install_sha,
            'current_map_receipt_sha256':map_sha,'current_solution_sha256':current_map['sha256'],
            'current_audit_receipt_sha256':audit_sha,'current_registry_sha256':current_audit['registry_sha256']}
        for prefix,audit in (('original_',original_audit),('current_',current_audit)):
            row.update({prefix+'audit_role_units':audit['role_audits'],prefix+'new_role_fits':audit['new_role_fits'],
                        prefix+'reused_role_audits':audit['reused_role_audits']})
        records.append(row)
    return _public_numerical_versions({'schema':1,'registered':True,'registration_sha256':pin,
        'registered_slots':len(records),'new_nominal_configurations':0,'coverage_complete':True,'records':records,'scope':VERSION_SCOPE})


def collect_evaluation_grid(*, out_root=config.OUT, ledger=None):
    """Read only accepted JSON summaries/receipts after the global freeze gate.

    Missing evaluation receipts remain explicitly incomplete. Present receipts
    with stale provenance fail closed. No unaccepted summary is opened.
    """
    root = Path(out_root); path = root/'SELECTION.json'; selection = json.loads(path.read_text())
    cfg_hash = config.digest(config.configuration())
    if selection.get('selection_frozen') is not True:raise ValueError('Evaluation evidence requires frozen selection')
    if (selection.get('config_hash') != cfg_hash or selection.get('configuration_digest') != cfg_hash
            or selection.get('source_hashes') != run.source_fingerprint()
            or selection.get('inference_source_hashes') != reporting.inference_source_fingerprint()):
        raise ValueError('Frozen selection configuration/source provenance changed')
    selection_sha = run.sha(path); nominal = _ledger(ledger)
    names = selection['evaluation_configurations']
    if len(names) != len(set(names)):raise ValueError('Duplicate frozen evaluation configuration')
    pairs = set(nominal) | {( _name(name), a) for name in names for a in ANCHORS}
    grid = {'schema':1, 'phase':'evaluation', 'pool':'test', 'selection_frozen':True,
            'selection_sha256':selection_sha, 'config_hash':cfg_hash,
            'source_hashes':selection['source_hashes'], 'inference_source_hashes':selection['inference_source_hashes'],
            'records':{name:{} for name in names}, 'receipts':[], 'audit_receipts':[], 'map_receipts':[],
            'numerical_versions':collect_numerical_versions(out_root=root)}
    audit_records = {}
    for name, anchor in sorted(pairs):
        base = root/'private/run'/f'anchor_{anchor}'
        audit_path = base/'audits'/name/'COMPLETE.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text()); run._validate_provenance(audit,'audit')
            if audit.get('anchor') != anchor or audit.get('configuration') != name:raise ValueError('Audit receipt identity changed')
            receipt = {k:audit[k] for k in ('registry_sha256','role_audits','new_role_fits','reused_role_audits')}
            receipt.update(configuration=name,anchor=anchor,receipt_sha256=run.sha(audit_path))
            grid['audit_receipts'].append(receipt); audit_records[(name,anchor)] = receipt
        map_path = base/'maps'/name/'ACCEPTED.json'
        if map_path.exists():
            mapping = json.loads(map_path.read_text()); run._validate_provenance(mapping,'map')
            if mapping.get('anchor') != anchor or mapping.get('configuration') != name:raise ValueError('Map receipt identity changed')
            grid['map_receipts'].append({'configuration':name,'anchor':anchor,'solution_sha256':_hash(mapping['sha256']),
                                         'receipt_sha256':run.sha(map_path)})
    for name in names:
        for anchor in ANCHORS:
            base = root/'private/run'/f'anchor_{anchor}'/'evaluation'/name; marker = base/'COMPLETE.json'
            if not marker.exists():continue
            pin = selection.get('frozen_audits',{}).get(name,{}).get(str(anchor)); audit = audit_records.get((name,anchor))
            if not pin or not audit or any(audit[k] != pin[k] for k in ('registry_sha256','receipt_sha256')):
                raise ValueError('Selection-time audit registry/receipt pin changed')
            record = json.loads(marker.read_text()); run._validate_provenance(record,'evaluation')
            if (record.get('anchor') != anchor or record.get('configuration') != name
                    or record.get('selection_sha256') != selection_sha or record.get('registry_sha256') != pin['registry_sha256']):
                raise ValueError('Evaluation receipt differs from frozen provenance')
            summary = base/'summary.json'; summary_sha = run.sha(summary)
            if record.get('artifact_hashes',{}).get('summary.json') != summary_sha:raise ValueError('Accepted summary hash mismatch')
            native = _native(json.loads(summary.read_text()),selection['predictor_choices'][name][str(anchor)])
            grid['records'][name][str(anchor)] = native
            grid['receipts'].append({'configuration':name,'anchor':anchor,'summary_sha256':summary_sha,
                                     'receipt_sha256':run.sha(marker)})
    if run.sha(path) != selection_sha:raise ValueError('Selection changed while collecting evidence')
    return grid


def _execution(grid, nominal, receipt_keys):
    audits = _receipts(grid.get('audit_receipts',[]),fields=('registry_sha256','receipt_sha256'),
                       extra=('role_audits','new_role_fits','reused_role_audits'))
    maps = _receipts(grid.get('map_receipts',[]),fields=('solution_sha256','receipt_sha256'))
    versions=_public_numerical_versions(grid.get('numerical_versions'))
    repeated=[r for r in versions['records'] if r['installation_decision']=='installed_accepted_retry']
    for row in versions['records']:
        key=(row['configuration'],row['anchor']);audit=audits.get(key,{});mapping=maps.get(key,{})
        if (key not in nominal or audit.get('receipt_sha256')!=row['current_audit_receipt_sha256']
                or audit.get('registry_sha256')!=row['current_registry_sha256']
                or mapping.get('receipt_sha256')!=row['current_map_receipt_sha256']
                or mapping.get('solution_sha256')!=row['current_solution_sha256']
                or audit.get('role_audits')!=row['current_audit_role_units']
                or audit.get('new_role_fits')!=row['current_new_role_fits']
                or audit.get('reused_role_audits')!=row['current_reused_role_audits']):
            raise ValueError('Numerical version differs from active nominal map/audit receipts')
    new = reused = roles = 0
    for record in grid.get('audit_receipts',[]):
        n, r, total = (_count(record[k]) for k in ('new_role_fits','reused_role_audits','role_audits'))
        if n+r != total or total != len(ROLES):raise ValueError('Role-fit/reuse counts do not match accepted audit schema')
        new += n; reused += r; roles += total
    if any(type(r.get('scheduled',True)) is not bool for r in nominal.values()):
        raise ValueError('Ledger scheduled flags must be booleans')
    scheduled = {key for key,r in nominal.items() if r.get('scheduled',True)}
    unscheduled = set(nominal)-scheduled
    expected_eval = {(name,a) for name in grid['records'] for a in ANCHORS}
    def missing(keys):return [{'configuration':n,'anchor':a} for n,a in sorted(keys)]
    return {'nominal_release_anchor_units':len(nominal),
            'nominal_map_units':sum(r['kind']=='finite_map' for r in nominal.values()),
            'nominal_role_audit_units':len(nominal)*len(ROLES),
            'scheduled_release_anchor_units':len(scheduled),
            'scheduled_map_units':sum(nominal[key]['kind']=='finite_map' for key in scheduled),
            'scheduled_role_audit_units':len(scheduled)*len(ROLES),
            'resource_unscheduled_release_anchor_units':len(unscheduled),
            'resource_unscheduled_units':missing(unscheduled),
            'accepted_audit_units':len(audits),'accepted_map_units':len(maps),
            'accepted_evaluation_units':len(receipt_keys),'scheduled_evaluation_units':len(expected_eval),
            'incomplete_audit_units':len(scheduled-set(audits)),
            'incomplete_evaluation_units':len(expected_eval-set(receipt_keys)),
            'incomplete_audits':missing(scheduled-set(audits)),
            'incomplete_evaluations':missing(expected_eval-set(receipt_keys)),
            'extra_accepted_audit_units':len(set(audits)-set(nominal)),
            'accepted_resource_unscheduled_audit_units':len(set(audits)&unscheduled),
            'registered_numerical_retry_slots':versions['registered_slots'],
            'completed_numerical_retry_attempts':len(versions['records']),
            'installed_numerical_replacements':len(repeated),
            'retained_original_after_rejected_retry':len(versions['records'])-len(repeated),
            'preserved_original_map_versions':len(repeated),'preserved_original_audit_versions':len(repeated),
            'repeated_audit_units':len(repeated),
            'total_accepted_map_versions':len(maps)+len(repeated),
            'total_accepted_audit_versions':len(audits)+len(repeated),
            'total_role_audit_units_with_repeats':roles+sum(r['original_audit_role_units'] for r in repeated),
            'total_new_role_fit_units_with_repeats':new+sum(r['original_new_role_fits'] for r in repeated),
            'unique_registry_artifacts':len({r['registry_sha256'] for r in audits.values()}),
            'unique_solution_artifacts':len({r['solution_sha256'] for r in maps.values()}),
            'accepted_role_audit_units':roles,'new_role_fit_units':new,'reused_role_audit_units':reused,
            'mathematical_duplicate_release_count':None,
            'exact_equivalence_status':'pending separate exact channel/action-dictionary proof',
            'count_scope':'accepted receipt declarations; active nominal units and preserved numerical versions are separate; repeated-role counts include superseded completed audits; operational scheduler incidents without completed fit receipts are not scientific failures or extra fitted versions; new role-fit units are not individual learner counts; artifact hashes do not prove mathematical channel equality'}


def _public_formula(formula):
    family_fields={'incomplete_required_families','eligible_comparator_families','empty_required_families'}
    if not isinstance(formula,dict) or set(formula)-{'kind','clauses','endpoint','check','reason','passed'}-family_fields:
        raise ValueError('Unexpected nonpublic claim formula field')
    for field in family_fields & set(formula):
        if (formula.get('kind')!='blocked' or not isinstance(formula[field],list)
                or any(not isinstance(name,str) for name in formula[field])):
            raise ValueError('Public family metadata requires string lists on blocked claim nodes')
    if not isinstance(formula.get('passed'),bool):raise ValueError('Claim verdict must be a boolean')
    if 'clauses' in formula:
        if formula.get('kind') not in ('all','any') or not formula['clauses']:raise ValueError('Invalid claim conjunction/disjunction')
        for clause in formula['clauses']:_public_formula(clause)
        fn=all if formula['kind']=='all' else any
        if formula['passed']!=fn(c['passed'] for c in formula['clauses']):raise ValueError('Inconsistent adjusted claim verdict')
    if formula.get('kind')=='blocked' and formula['passed']:raise ValueError('Blocked claim cannot pass')


def _public_bounds(bounds):
    scalar={'estimate','bootstrap_se','lower','upper','lower_one_sided_adjusted','upper_one_sided_adjusted','critical_value_adjusted'}
    arrays={'anchor_estimates','anchor_person_counts','anchor_household_counts','anchor_positive_weight_households'}
    for record in bounds.values():
        if set(record)-scalar-arrays-{'identical_pairs'}:raise ValueError('Unexpected nonpublic bounds field')
        for key in ('estimate','lower','upper'):_number(record[key],nonnegative=False)
        for key in scalar & set(record):_number(record[key],nonnegative=key in ('bootstrap_se','critical_value_adjusted'))
        if not record['lower']<=record['estimate']<=record['upper']:raise ValueError('Inconsistent interval bounds')
        for key in arrays & set(record):
            if len(record[key])!=3:raise ValueError('Bound diagnostic must retain all three anchors')
            for value in record[key]:_number(value,nonnegative=key!='anchor_estimates')
        if 'identical_pairs' in record and not isinstance(record['identical_pairs'],bool):raise ValueError('Invalid exact-pair flag')


def build_evidence(grid, *, selection, claims=None, bounds=None, ledger=None):
    """Pure, aggregate-only derivation; no selection or inference is performed."""
    if (grid.get('phase') != 'evaluation' or grid.get('pool') != 'test'
            or grid.get('selection_frozen') is not True or selection.get('selection_frozen') is not True):
        raise ValueError('Only frozen test evaluation aggregates are supported')
    if set(grid)-{'schema','phase','pool','selection_frozen','selection_sha256','config_hash','source_hashes',
                  'inference_source_hashes','records','receipts','audit_receipts','map_receipts','numerical_versions'}:
        raise ValueError('Unexpected nonpublic grid field')
    if 'config_hash' in grid:_hash(grid['config_hash'])
    for key,permitted in (('source_hashes',run.source_fingerprint()),('inference_source_hashes',reporting.inference_source_fingerprint())):
        if key in grid:
            if set(grid[key])!=set(permitted):raise ValueError('Unexpected public source-hash schema')
            for value in grid[key].values():_hash(value)
    selection_sha = _hash(grid['selection_sha256']); nominal = _ledger(ledger)
    receipts = _receipts(grid['receipts'],fields=('summary_sha256','receipt_sha256'))
    names = set(selection['evaluation_configurations'])
    if set(grid['records']) != names:raise ValueError('Grid configuration set differs from frozen selection')
    values = {}; configs = {}; keys = set()
    for name, raw in sorted(grid['records'].items()):
        _name(name); anchors = {}
        for key, native in raw.items():
            a = _anchor(key)
            if a in anchors:raise ValueError('Duplicate evaluation anchor aliases')
            if (name,a) not in receipts:raise ValueError('Missing accepted summary receipt')
            anchors[a] = _native(native,selection['predictor_choices'][name][str(a)])['test'];keys.add((name,a))
        absent = sorted(set(ANCHORS)-set(anchors))
        if absent:
            configs[name] = {'status':'incomplete','missing_anchors':absent,'available_anchors':sorted(anchors)}
            continue
        values[name] = anchors; rows = {}
        for role in ROLES:
            metrics = {}
            for output, field in (('selected','ce'),('independent','independent_ce'),('fixed','fixed_decoder_ce')):
                available = [field in anchors[a][role] for a in ANCHORS]
                metrics[output] = ({w:math.fsum(anchors[a][role][field][k] for a in ANCHORS)/3
                                     for w,k in WEIGHTS.items()} if all(available) else None)
                if output == 'fixed':metrics['fixed_unavailable_anchors']=[a for a,v in zip(ANCHORS,available) if not v]
            rows[role] = {**metrics,
                'support_by_anchor':{str(a):{k:copy.deepcopy(anchors[a][role][k]) for k in ('support','weight_sum','ess','per_class')} for a in ANCHORS},
                'accuracy':{w:math.fsum(anchors[a][role]['accuracy'][k] for a in ANCHORS)/3 for w,k in WEIGHTS.items()},
                'predictor_ids_by_anchor':{str(a):{k:anchors[a][role][k] for k in ('selection','independent_selection')} for a in ANCHORS}}
        configs[name] = {'status':'complete','roles':rows,'task':{k:copy.deepcopy(rows[TASK][k]) for k in ('selected','independent','fixed','fixed_unavailable_anchors')},
                         'primary_sensitive':{},'original_source_utility':{},'secondary_roles':{},
                         'evidence':[receipts[(name,a)] for a in ANCHORS]}
    if set(receipts) != keys:raise ValueError('Accepted receipts do not exactly cover supplied summaries')
    for name, row in configs.items():
        if row['status'] != 'complete':continue
        for role in ROLES:
            target = row['roles'][role]
            for baseline in ('H','J'):
                ref = configs.get(baseline,{})
                target['loss_delta_'+baseline] = ({w:target['selected'][w]-ref['roles'][role]['selected'][w] for w in WEIGHTS}
                    if ref.get('status') == 'complete' else None)
            if role in ('attack:'+r for r in config.PRIMARY):
                h = configs.get('H',{}); j = configs.get('J',{})
                row['primary_sensitive'][role.split(':',1)[1]] = {
                    w:{'selected_ce':target['selected'][w],
                       'recovery_over_H':None if target['loss_delta_H'] is None else -target['loss_delta_H'][w],
                       'recovery_increment_over_J':None if target['loss_delta_J'] is None else -target['loss_delta_J'][w],
                       'J_minus_H_loss':j['roles'][role]['selected'][w]-h['roles'][role]['selected'][w]
                           if h.get('status') == j.get('status') == 'complete' else None} for w in WEIGHTS}
            elif role.startswith('utility:') and role != TASK:row['original_source_utility'][role]=copy.deepcopy(target)
            elif role.startswith('attack:'):row['secondary_roles'][role]=copy.deepcopy(target)
        row['task']['loss_delta_H']=row['roles'][TASK]['loss_delta_H'];row['task']['loss_delta_J']=row['roles'][TASK]['loss_delta_J']
    families = default_families()
    for name,spec in selection.get('families',{}).items():families[name]=spec
    family_rows = {family:{'configurations':list(spec['configurations']),
                           'required_configurations':list(spec.get('required_configurations',spec['configurations'])),
                           'complete':[n for n in spec['configurations'] if configs.get(n,{}).get('status')=='complete'],
                           'unavailable':[n for n in spec['configurations'] if configs.get(n,{}).get('status')!='complete']}
                   for family,spec in families.items()}
    inference = {'status':'not_supplied','main':{},'attribution':{},'diagnostic':{}}
    if (claims is None) != (bounds is None):raise ValueError('Claims and paired bounds must be supplied together')
    if claims is not None:
        if (claims.get('provenance') != bounds.get('provenance') or claims.get('provenance',{}).get('selection_sha256') != selection_sha
                or claims.get('family_size') != bounds.get('family_size') or len(bounds['bounds']) != bounds['family_size']
                or bounds['family_size'] != selection.get('contrast_family_size')
                or claims.get('provenance',{}).get('contrasts_sha256') != selection.get('contrasts_sha256')):
            raise ValueError('Selected inference provenance or endpoint family mismatch')
        provenance=claims['provenance']
        if set(provenance)-{'selection_sha256','contrasts_sha256','config_hash','source_hashes','inference_source_hashes'}:
            raise ValueError('Unexpected nonpublic inference provenance field')
        for key in ('config_hash','source_hashes','inference_source_hashes'):
            if key in selection and provenance.get(key)!=selection[key]:raise ValueError('Selected inference source/configuration provenance mismatch')
        _public_bounds(bounds['bounds'])
        for group in ('claim_results','attribution_claim_results','diagnostic_claim_results'):
            for route in claims.get(group,{}).values():
                for formula in route.values():_public_formula(formula)
        inference = {'status':'supplied_adjusted_results','main':copy.deepcopy(claims['claim_results']),
                     'attribution':copy.deepcopy(claims.get('attribution_claim_results',{})),
                     'diagnostic':copy.deepcopy(claims.get('diagnostic_claim_results',{})),
                     'family_size':bounds['family_size'],'bounds':copy.deepcopy(bounds['bounds']),
                     'provenance':copy.deepcopy(claims['provenance'])}
    return {'schema':1,'phase':'evaluation','pool':'test','selection_frozen':True,
            'selection_sha256':selection_sha,'grid_digest':config.digest(grid),
            'aggregation':'equal mean of exactly three anchor estimates, separately for each weighting; class support remains anchor-specific',
            'scope':'descriptive historically used 2018 development evaluation; measured attacker CE recovery, not mutual information',
            'cross_anchor_scope':{'globally_unseen_people':False,'globally_untouched_labels':False,
                'retraining_uncertainty_included':False,'bounds_conditional_on_fitted_models':True,
                'description':'Anchor-specific seals do not imply globally unseen people; non-test assignments overlap other anchors test pools. Shared-household resampling does not account for retraining or validation-selection uncertainty.',
                'disclosure':'DATED_SPLIT_CLARIFICATION.md'},
            'numerical_versions':_public_numerical_versions(grid.get('numerical_versions')),
            'sensitive_signs':{'recovery_over_H':'CE(H)-CE(configuration)','recovery_increment_over_J':'CE(J)-CE(configuration)',
                               'J_minus_H_loss':'CE(J)-CE(H)'},
            'configurations':configs,'baseline_families':family_rows,
            'historical_baselines':{n:configs.get(n,{'status':'unavailable'})['status'] for n in HISTORICAL},
            'execution':_execution(grid,nominal,keys),
            'selected_routes':{r:{k:copy.deepcopy(v[k]) for k in ('nominee','screen_passed','descriptive_only',
                                'competitive_validation_eligible','empty_required_families','incomplete_required_families',
                                'scientifically_ineligible_families','eligible_comparator_families',
                                'strict_all_families_validation_eligible') if k in v}
                               for r,v in selection.get('routes',{}).items()},
            'family_eligibility':{r:{family:{k:copy.deepcopy(spec[k]) for k in ('configuration','required_configurations',
                                 'missing_required_configurations','required_frontier_complete','scientifically_ineligible',
                                 'eligible','empty_eligibility') if k in spec}
                                   for family,spec in route.get('family_nominees',{}).items()}
                                  for r,route in selection.get('routes',{}).items()},
            'selected_inference':inference,
            'provenance':{k:copy.deepcopy(grid[k]) for k in ('config_hash','source_hashes','inference_source_hashes') if k in grid}}


def _fmt(value):return 'unavailable' if value is None else f'{value:.9g}'


def _table(headers,rows):
    def cell(value):return str(value).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
                      *('| '+' | '.join(cell(v) for v in row)+' |' for row in rows)])+'\n'


def render_tables(evidence):
    configs=evidence['configurations'];task=[];roles=[];sensitive=[];baselines=[];eligibility=[]
    for name,row in configs.items():
        if row['status']!='complete':continue
        for w in WEIGHTS:
            t=row['task'];task.append([name,w,*(_fmt(t[k][w]) if t[k] is not None else 'fixed decoder unavailable' if k=='fixed' else 'unavailable'
                                              for k in ('fixed','independent','selected'))])
        for role,metrics in row['roles'].items():
            for w in WEIGHTS:roles.append([name,role,w,_fmt(metrics['selected'][w]),
                _fmt(metrics['loss_delta_H'][w]) if metrics['loss_delta_H'] else 'unavailable',
                _fmt(metrics['loss_delta_J'][w]) if metrics['loss_delta_J'] else 'unavailable'])
        for role,metrics in row['primary_sensitive'].items():
            for w,s in metrics.items():sensitive.append([name,role,w,_fmt(s['recovery_over_H']),_fmt(s['recovery_increment_over_J'])])
    for name,status in evidence['historical_baselines'].items():baselines.append(['historical',name,status])
    for family,spec in evidence['baseline_families'].items():
        for name in spec['configurations']:baselines.append([family,name,configs.get(name,{}).get('status','unavailable')])
    for route,families in evidence.get('family_eligibility',{}).items():
        for family,spec in families.items():eligibility.append([route,family,spec.get('required_frontier_complete','unavailable'),
            ', '.join(spec.get('missing_required_configurations',[])) or 'none',
            spec.get('scientifically_ineligible','unavailable'),spec.get('configuration') or 'none'])
    provenance='Source: EVIDENCE.json; native accepted summaries: EVALUATION_GRID.json. Selection SHA-256: `'+evidence['selection_sha256']+'`. Grid digest: `'+evidence['grid_digest']+'`. Every complete configuration stores its three summary/receipt hashes.\n\n'
    context='Machine-rendered descriptive aggregates. Means require all three anchors; unweighted and PWGTP metrics use the same frozen predictors. Missing diagnostics are unavailable, never imputed. These are dependent analyses of the same inherited 2018 cohort; people are not globally unseen across anchors. The anchor-specific test seal does not imply globally untouched labels. Household bounds are conditional on fitted models; they do not include retraining or validation-selection uncertainty. See DATED_SPLIT_CLARIFICATION.md.\n\n'
    execution=evidence['execution'];count_rows=[[k,v] for k,v in execution.items() if isinstance(v,(int,type(None))) and not isinstance(v,bool)]
    incomplete=[[n,','.join(map(str,r.get('missing_anchors',[])))] for n,r in configs.items() if r['status']!='complete']
    return {
      'UTILITY_DECOMPOSITION.md':'# Utility decomposition\n\n'+provenance+context+_table(['Configuration','Weighting','Fixed decoder CE','Independent probe CE','Selected deployment CE'],task),
      'BASELINES.md':'# Baseline coverage\n\n'+provenance+context+'Availability is not an eligibility or superiority verdict. A complete scientifically ineligible family supplies no comparator and does not veto other eligible families. Missing mandatory settings block the required frontier. At least one eligible comparator family is required. Stricter all-family results are separate diagnostics.\n\n'+_table(['Family','Configuration','Evaluation status'],baselines)+'\n'+_table(['Route','Family','Required frontier complete','Missing mandatory settings','Scientifically ineligible','Frozen nominee'],eligibility),
      'DEVELOPMENT_RESULTS.md':'# Development evaluation tables\n\n'+provenance+context+
          'This is historically used 2018 development evaluation. Recovery is attacker log-loss reduction, not mutual information. Adjusted selected, attribution, and stricter diagnostic verdicts remain separate in EVIDENCE.json and CLAIM_RESULTS.json.\n\n'+
          '## Execution counts\n\n'+_table(['Count','Value'],count_rows)+'\n'+execution['count_scope']+'. Exact channel equivalence remains separately adjudicated.\n\n'+
          '## Incomplete configurations\n\n'+_table(['Configuration','Missing anchors'],incomplete)+
          '\n## Primary sensitive recovery\n\n'+_table(['Configuration','Role','Weighting','CE(H) minus CE(arm)','CE(J) minus CE(arm)'],sensitive)+
          '\n## All registered roles\n\n'+_table(['Configuration','Role','Weighting','Selected CE','CE(arm) minus CE(H)','CE(arm) minus CE(J)'],roles)}


def write_evidence(evidence,grid,*,out_dir):
    if evidence['grid_digest']!=config.digest(grid):raise ValueError('Evidence/grid digest mismatch')
    directory=Path(out_dir);documents=render_tables(evidence)
    outputs={'EVIDENCE.json':json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n',
             'EVALUATION_GRID.json':json.dumps(grid,indent=2,sort_keys=True,allow_nan=False)+'\n',**documents}
    if any((directory/n).exists() for n in outputs):raise FileExistsError('Evidence outputs are immutable; use a new directory')
    directory.mkdir(parents=True,exist_ok=True)
    for name,content in outputs.items():
        with (directory/name).open('x') as handle:handle.write(content)
    return {name:run.sha(directory/name) for name in outputs}


def generate(*,out_root=config.OUT,out_dir=None,ledger=None):
    """Explicit production entry point; invoke only once evaluation is permitted."""
    root=Path(out_root);grid=collect_evaluation_grid(out_root=root,ledger=ledger)
    selected=json.loads((root/'SELECTION.json').read_text())
    cp,bp=root/'CLAIM_RESULTS.json',root/'PAIRED_BOUNDS.json'
    if cp.exists()!=bp.exists():raise ValueError('Incomplete inference result pair')
    claims=json.loads(cp.read_text()) if cp.exists() else None
    bounds=json.loads(bp.read_text()) if bp.exists() else None
    if claims is not None:
        contrast_path=root/'CONTRASTS.json'
        if run.sha(contrast_path)!=selected['contrasts_sha256']:raise ValueError('Frozen contrast file hash changed')
        contrasts=json.loads(contrast_path.read_text())
        checked=reporting.evaluate_claims(contrasts,bounds)
        if any(claims.get(key)!=checked[key] for key in ('family_size','checks','claim_results','attribution_claim_results','diagnostic_claim_results')):
            raise ValueError('Claim results differ from frozen formulas and adjusted bounds')
    evidence=build_evidence(grid,selection=selected,claims=claims,bounds=bounds,ledger=ledger)
    if run.sha(root/'SELECTION.json')!=grid['selection_sha256']:raise ValueError('Selection changed during evidence generation')
    return write_evidence(evidence,grid,out_dir=root if out_dir is None else out_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out-root',type=Path,default=config.OUT)
    parser.add_argument('--out-dir',type=Path);parser.add_argument('--ledger',type=Path)
    args=parser.parse_args();ledger=json.loads(args.ledger.read_text()) if args.ledger else None
    print(json.dumps(generate(out_root=args.out_root,out_dir=args.out_dir,ledger=ledger),indent=2))
