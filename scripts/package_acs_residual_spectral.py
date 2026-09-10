"""Allowlisted aggregate publication evidence; originals and models stay local.

No training, scoring, model deserialization, git staging, or historical writes.
Run --self-test first, --dry-run during execution, then rerun after final reports.
Only paths in PUBLICATION_FILES.txt are approved publication artifacts.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
STUDY='redesign_20260910_acs_residual_spectral_v1'
DEFAULT=ROOT/'results'/STUDY
MAX_PUBLIC_BYTES=45_000_000
COMPRESS_THRESHOLD=250_000
ROOT_ALLOWLIST={
 'PROTOCOL.md','MATHEMATICS.md','IMPLEMENTATION_PLAN.md','PROSPECTIVE_AUDIT.md',
 'PREFIT_FREEZE.json','RELEASE_MANIFEST.json','RUNTIME_EVENTS.json',
 'HISTORICAL_AUDIT.md','HISTORICAL_AUDIT.json','HISTORICAL_REPLAY_NEW.json',
 'IMPLEMENTATION_REVIEW_INDEPENDENT.md','IMPLEMENTATION_REVIEW_INDEPENDENT.json',
 'DATA_ADMISSION.json','DATA_ADMISSION_PRIOR_USE.json','TRANSPORT_ADMISSION.md',
 'TRANSPORT_HISTORY_AUDIT.json','TRANSPORT_TESTS.txt','TRANSPORT_VERIFICATION.json',
 'RESEARCH_DECISION.md','TABLE.md','REPRODUCTION.md','FINAL_RECORD.json',
 'FINAL_VALIDATION.json','VALIDATION.md','REUSE_MANIFEST.json','FITTING_LEDGER.json',
 'summary.json','criteria.json','comparison_rules.json','metrics.json',
 'TABLE.csv','METRICS.csv','PER_CLASS.csv','PER_CANDIDATE.csv','PAIRED.csv',
 'PAIRED_AGGREGATE.csv','PARETO.csv','FEATURE_CAPABILITY.csv','UTILITY_MATCHES.csv',
 'SUPPORT.csv','SOURCE_UTILITY.csv','TRANSFER.csv','AUDITS.csv','AUDIT_CURVES.csv',
 'NUMERICAL_CHECKS.json','NUMERICAL_REPLAY.json','TRAINING_REPLAY.json','SCORE_REPLAY.json',
 'TESTS.txt','FINAL_TESTS.txt','FINAL_TESTS.json','REPORT_REVIEW.md','REPORT_REVIEW.json'}
ROOT_ALLOWLIST.update(name+'.csv' for name in ('PER_SEED','MEANS','ALL_CANDIDATES','PER_CLASS','PAIRED','PAIRED_AGGREGATE','UTILITY_MATCHES','DIRECTIONAL_VECTORS','SOURCE_FEASIBILITY','ORIGINAL_CRITERIA','ORIGINAL_CONTEXT','WITHHOLDING','WITHHOLDING_SCORE_REPLAY','WITHHOLDING_COMPARISONS'))
ROOT_ALLOWLIST.update(name+'.json' for name in ('DECISION','WITHHOLDING_VALIDATION','REPORT_INPUT_HASHES','REPORT_COUNTS','FIGURES'))
ROOT_ALLOWLIST.update(('DECISION_REPORT.md','PUBLICATION_TESTS.txt','START.json','CALIBRATION.json','OPERATIONAL_AMENDMENTS.json','RUNTIME.json','FITTING_COUNTS.json','COMPLETION.json','FINAL_REVIEW.md','PROJECTION_LAYOUT_REPAIR.json','environment.json','SURROGATE_DIAGNOSTICS.csv','INDEPENDENT_REPLAY.json','INDEPENDENT_REPLAY.md','INDEPENDENT_SURROGATES.json','verification_sources/verify_surrogates.py','NEGATIVE_INCREMENT_COUNTS.csv','INDEPENDENT_NEGATIVE_COUNTS.json','INDEPENDENT_LEGACY_CRITERIA.json'))
ROOT_ALLOWLIST.update(('FINAL_INDEPENDENT_REVIEW.json','verification_sources/verify_negative_counts.py','verification_sources/verify_legacy_criteria.py'))
ROOT_ALLOWLIST.discard('RUNTIME_EVENTS.json')
STABLE_PLAIN_REPORTS={'DECISION.json','WITHHOLDING_VALIDATION.json','FIGURES.json','REPORT_COUNTS.json','REPORT_INPUT_HASHES.json'}
FIGURE=re.compile(r'^tradeoff_figures/seed[012]_(unweighted|person_weighted)_(120|360)_(standard_independent|expanded_independent|expanded_catchup|kernel_standard_independent|kernel_expanded_independent|kernel_expanded_catchup)\.(png|pdf)$')
ARMS=('H','E','A0','J','L025','L20',
      'spectral_S0','spectral_M025','spectral_M1','spectral_L025','spectral_L1','spectral_C025','spectral_C1','spectral_L2')
UNIT=re.compile(r'^seed_[012]/(?:'+ '|'.join(map(re.escape,ARMS))+r')/(metrics.json|selection_before_test.json|complete.json|projection_layout_repair.json|audits/audit_selection.json)$')
SEED=re.compile(r'^seed_[012]/(support.json|anchor_parity.json|maps_complete.json|indices.json)$')
NUMERICAL=re.compile(r'^(NUMERICAL_VALIDATION.json|seed_[012]/(matrix_diagnostics.json|heldout_moments.json))$')
OUTPUT_NAMES={'NUMERICAL_SUMMARY.json','LOCAL_ARTIFACTS.json','EVIDENCE_MANIFEST.json',
              'PUBLICATION_FILES.txt','PUBLICATION_SUMMARY.json','PUBLICATION.md'}
PRIVATE_KEYS={'fold_assignments','bandwidth_subset_indices','moment_matrix','covariance_matrix',
 'raw_rows','row_indices','row_ids','person_ids','household_ids','SERIALNO','SPORDER',
 'means','scales','feature_mean','feature_scale','coef','coef_','coefficients',
 'intercept','intercept_','decoder_weights','nuisance_weights','map_weights','prior',
 'whitener','whitening_matrix','projection_matrix','rotation_matrix','omega','phase'}


def canonical_json(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def digest(data):return hashlib.sha256(data).hexdigest()


def canonical_gzip(data):
    buffer=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',fileobj=buffer,mtime=0,compresslevel=9) as f:f.write(data)
    return buffer.getvalue()


def shape(value):
    if isinstance(value,list):
        return [len(value)]+(shape(value[0]) if value and isinstance(value[0],list) else [])
    return None


def project_public(value):
    removed=[]
    def visit(v,path=''):
        if isinstance(v,dict):
            out={}
            for key,item in v.items():
                private=key in PRIVATE_KEYS or (isinstance(item,list) and
                  (key.endswith('_rows') or key.endswith('_indices')) and key!='retained_indices')
                # Counts and exact dictionary-block hash metadata are not person records.
                if key=='raw_rows' and isinstance(item,(int,float)):private=False
                if isinstance(item,str) and re.fullmatch(r'[0-9a-f]{64}',item):private=False
                if key in ('SERIALNO','SPORDER') and path=='/dictionary_comparison' and isinstance(item,dict) and set(item)<={'identical_text','2017_sha256','2018_sha256'}:private=False
                if private:
                    metadata={'sha256':digest(canonical_json(item)),'shape':shape(item),'reason':'row-level or fitted-array content excluded'}
                    out[key+'_omitted']=metadata
                    removed.append({'path':path+'/'+key,**metadata})
                else:out[key]=visit(item,path+'/'+key)
            return out
        if isinstance(v,list):return [visit(x,path+'/'+str(i)) for i,x in enumerate(v)]
        return v
    return visit(value),removed


def guard_root(out):
    if out.resolve()!=DEFAULT.resolve():raise ValueError('Only the new study result directory is writable')
    if out.is_symlink():raise ValueError('Result root must not be a symlink')


def classify(relative):
    if 'revisions' in Path(relative).parts:return 'local'
    if relative.endswith('.csv.gz') and relative[:-3] in ROOT_ALLOWLIST:return 'gzip_input'
    if NUMERICAL.fullmatch(relative):return 'numerical'
    if '/fitted/' in relative or relative.endswith(('.joblib','.pt','.npz','.npy','.log')):return 'local'
    if UNIT.fullmatch(relative):
        return 'gzip' if relative.endswith(('metrics.json','audits/audit_selection.json','selection_before_test.json')) else 'public'
    if SEED.fullmatch(relative) or FIGURE.fullmatch(relative) or relative in ROOT_ALLOWLIST:return 'public'
    return 'local'


def write_exact(path,data,dry_run=False):
    if len(data)>MAX_PUBLIC_BYTES:raise ValueError('Generated public artifact exceeds 45 MB cap: '+str(path))
    if dry_run:return
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()==data:return
    temporary=path.with_name(path.name+'.packaging.tmp')
    temporary.write_bytes(data);temporary.replace(path)


def package(out=DEFAULT,dry_run=False):
    out=Path(out);guard_root(out)
    original_paths=[]
    for p in sorted(out.rglob('*')):
        relative=p.relative_to(out).as_posix()
        if p.is_symlink():raise ValueError('Symlink inside result directory: '+relative)
        if not p.is_file() or relative.startswith('PUBLICATION_EVIDENCE/') or relative in OUTPUT_NAMES or relative.endswith('.packaging.tmp'):continue
        original_paths.append((p,relative))
    public=[];omitted=[];numerical={};exports=[];input_stats={};total_input=0
    for p,relative in original_paths:
        before=p.stat();data=p.read_bytes();after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('File changed during packaging; rerun after it is closed: '+relative)
        input_stats[relative]=(after.st_size,after.st_mtime_ns)
        identity={'path':relative,'bytes':len(data),'sha256':digest(data)};total_input+=len(data)
        mode=classify(relative)
        if mode=='local':
            omitted.append({**identity,'reason':'Not explicitly allowlisted; fitted objects, rows and private state remain local'})
            continue
        source_compressed=mode=='gzip_input'
        transformed=gzip.decompress(data) if source_compressed else data
        removed=[]
        if relative.endswith('.json'):
            projected,removed=project_public(json.loads(data))
            if removed:
                if relative in STABLE_PLAIN_REPORTS:raise ValueError('Stable report contains private fields requiring explicit projection review: '+relative)
                transformed=canonical_json(projected)
        if mode=='numerical':
            numerical[relative]={'source_sha256':identity['sha256'],'summary':projected,'removed_fields':removed}
            omitted.append({**identity,'reason':'Original diagnostics retained locally; aggregate projection is NUMERICAL_SUMMARY.json'})
            continue
        compressed=(mode in ('gzip','gzip_input') or relative.endswith('.csv') or
                    (relative not in STABLE_PLAIN_REPORTS and relative.endswith(('.json','.txt')) and len(transformed)>=COMPRESS_THRESHOLD))
        if removed or compressed:
            target='PUBLICATION_EVIDENCE/'+relative+('.gz' if compressed and not source_compressed else '')
            payload=canonical_gzip(transformed) if compressed else transformed
            if compressed and gzip.decompress(payload)!=transformed:raise AssertionError('Gzip round-trip failure')
            if len(payload)>MAX_PUBLIC_BYTES:raise ValueError('Public artifact exceeds 45 MB cap: '+target)
            write_exact(out/target,payload,dry_run)
            record={'path':target,'bytes':len(payload),'sha256':digest(payload)}
            public.append(record)
            exports.append({'original':identity,'published':record,'codec':'gzip mtime=0' if compressed else 'JSON',
               'lossless_original_bytes':not removed and not source_compressed,
               'source_codec':'gzip' if source_compressed else 'identity',
               'lossless_decompressed_source':source_compressed and not removed,
               'source_uncompressed_sha256':digest(gzip.decompress(data)) if source_compressed else None,
               'projected_payload_sha256':digest(transformed),'removed_fields':removed})
            omitted.append({**identity,'reason':'Original retained locally; canonical public archive/projection listed in exports'})
        else:
            if len(data)>MAX_PUBLIC_BYTES:raise ValueError('Public artifact exceeds 45 MB cap: '+relative)
            public.append(identity)
    numerical_data=canonical_json({'scope':'Aggregate numerical diagnostics only; excludes row/fold/subset assignments and fitted arrays',
      'inputs':numerical,'no_new_fitting_or_scoring':True})
    write_exact(out/'NUMERICAL_SUMMARY.json',numerical_data,dry_run)
    public.append({'path':'NUMERICAL_SUMMARY.json','bytes':len(numerical_data),'sha256':digest(numerical_data)})
    local_data=canonical_json({'scope':'Only files inside this new result directory; historical dependencies retain existing reuse manifests',
      'files':omitted,'file_count':len(omitted),'total_bytes':sum(x['bytes'] for x in omitted)})
    write_exact(out/'LOCAL_ARTIFACTS.json',local_data,dry_run)
    public.append({'path':'LOCAL_ARTIFACTS.json','bytes':len(local_data),'sha256':digest(local_data)})
    summary={'packaging_state':'snapshot_only; does not assert experiment completion',
      'input_files':len(original_paths),'input_bytes':total_input,'public_content_files':len(public),
      'public_content_bytes':sum(x['bytes'] for x in public),'local_original_files':len(omitted),
      'local_original_bytes':sum(x['bytes'] for x in omitted),'exports':len(exports),
      'numerical_source_files':len(numerical),'per_file_public_cap_bytes':MAX_PUBLIC_BYTES,
      'no_model_deserialization':True,'no_fitting_or_scoring':True,'raw_data_directory_not_read_or_added':True,
      'scope':'Counts include public numerical/local-hash summaries, exclude these publication control records to avoid self-referential hashes.'}
    summary_data=canonical_json(summary);write_exact(out/'PUBLICATION_SUMMARY.json',summary_data,dry_run)
    public.append({'path':'PUBLICATION_SUMMARY.json','bytes':len(summary_data),'sha256':digest(summary_data)})
    notes=('# Residual spectral publication package\n\nThis is an evidence-packaging snapshot, not an assertion of completed scientific evaluation. Publish only paths listed in `PUBLICATION_FILES.txt`, together with separately reviewed source/tests/docs. No raw `data/` path, fitted object, prediction array, row identifier, fold assignment, or bandwidth subset is included. Originals remain untouched and local.\n\n`EVIDENCE_MANIFEST.json` lists every public content hash/size and each exact or sanitized export. `LOCAL_ARTIFACTS.json` inventories every omitted original by path, hash, and bytes. No checkpoint download or exact saved-model replay is provided by this public package; refitting is a new reproduction.\n\n`NUMERICAL_SUMMARY.json` projects numerical and held-out-moment diagnostics into compact aggregate evidence. Fitted vectors/matrices and row-level arrays are replaced by hashes/shapes. Eigenvalues, class support and class moment norms remain. Sanitized audit selections retain validation curves and scoring evidence; their original fitted preprocessing vectors remain local. An export marked `lossless_original_bytes: true` decompresses to exact original bytes; existing compressed CSV inputs preserve exact decompressed CSV bytes and receive a canonical gzip header; other exports are explicitly documented projections.\n\nAll gzip archives use an empty filename header and mtime zero. Repeated runs on unchanged input produce identical package bytes. The package fails if a source file changes while read or exceeds the public size cap. Rerun after final evaluation/report logs close to inventory the completed evidence. Neither historical files nor git staging are modified.\n').encode()
    write_exact(out/'PUBLICATION.md',notes,dry_run)
    public.append({'path':'PUBLICATION.md','bytes':len(notes),'sha256':digest(notes)})
    manifest={'result_directory':'results/'+STUDY,'public_files':sorted(public,key=lambda r:r['path']),
       'exports':exports,'controls_not_self_hashed':['EVIDENCE_MANIFEST.json','PUBLICATION_FILES.txt'],
       'public_allowlist_source':'scripts/package_acs_residual_spectral.py',
       'packager_source_sha256':digest(Path(__file__).read_bytes()),'public_file_count_including_controls':len(public)+2}
    manifest_data=canonical_json(manifest);write_exact(out/'EVIDENCE_MANIFEST.json',manifest_data,dry_run)
    names=sorted(['results/'+STUDY+'/'+r['path'] for r in public]+['results/'+STUDY+'/'+n for n in ('EVIDENCE_MANIFEST.json','PUBLICATION_FILES.txt')])
    write_exact(out/'PUBLICATION_FILES.txt',('\n'.join(names)+'\n').encode(),dry_run)
    # Inputs can grow while an evaluation is running. Do not certify a mixed snapshot.
    for relative,(size,mtime) in input_stats.items():
        now=(out/relative).stat()
        if (now.st_size,now.st_mtime_ns)!=(size,mtime):raise ValueError('Input changed during packaging: '+relative)
    print(json.dumps({**summary,'dry_run':dry_run,'total_public_bytes_including_controls':
       sum(x['bytes'] for x in public)+len(manifest_data)+len(('\n'.join(names)+'\n').encode())},indent=2))
    return manifest



def verify_public(out=DEFAULT):
    """Verify distributable evidence using only public files; never read models."""
    out=Path(out);guard_root(out)
    manifest=json.loads((out/'EVIDENCE_MANIFEST.json').read_text())
    if manifest['result_directory']!='results/'+STUDY:raise ValueError('Wrong evidence root')
    for record in manifest['public_files']:
        relative=Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        p=out/relative
        if p.is_symlink():raise ValueError('Public evidence cannot be a symlink')
        data=p.read_bytes()
        assert len(data)==record['bytes'] and digest(data)==record['sha256'],str(relative)
        assert len(data)<=MAX_PUBLIC_BYTES
    for record in manifest['exports']:
        data=(out/record['published']['path']).read_bytes()
        plain=gzip.decompress(data) if record['codec']=='gzip mtime=0' else data
        assert digest(plain)==record['projected_payload_sha256']
        if record['lossless_original_bytes']:assert digest(plain)==record['original']['sha256']
        if record.get('lossless_decompressed_source'):assert digest(plain)==record['source_uncompressed_sha256']
        if record['original']['path'].endswith('.json'):
            _,removed=project_public(json.loads(plain))
            assert not removed,'Private field survived projection'
    _,removed=project_public(json.loads((out/'NUMERICAL_SUMMARY.json').read_text()))
    assert not removed,'Private numerical field survived projection'
    expected=sorted(['results/'+STUDY+'/'+r['path'] for r in manifest['public_files']]+[
       'results/'+STUDY+'/'+x for x in ('EVIDENCE_MANIFEST.json','PUBLICATION_FILES.txt')])
    assert (out/'PUBLICATION_FILES.txt').read_text().splitlines()==expected
    assert all(not p.startswith('data/') for p in expected)
    summary={'verified_public_files':len(expected),'verified_exports':len(manifest['exports']),
       'total_public_bytes':sum((ROOT/p).stat().st_size for p in expected),
       'local_original_bytes':json.loads((out/'LOCAL_ARTIFACTS.json').read_text())['total_bytes'],
       'model_files_read':0,'person_arrays_read':0}
    print(json.dumps(summary,indent=2));return summary


def self_test():
    global ROOT, DEFAULT
    sample={'fold_assignments':[0,1,0], 'bandwidth_subset_indices':[8,2],
       'nested':{'moment_matrix':[[1.,2.]],'feature_mean':[2.,3.],
                 'class_counts':[2,3],'eigenvalues':[1.,.5],'norm':2.},
       'validation_curve':[{'epoch':0,'loss':.5}], 'fit_hashes':{'x':'abc'}}
    public,removed=project_public(sample)
    assert 'fold_assignments' not in public and 'bandwidth_subset_indices' not in public
    assert 'moment_matrix' not in public['nested'] and 'feature_mean' not in public['nested']
    assert public['nested']['class_counts']==[2,3]
    assert public['nested']['eigenvalues']==[1.,.5]
    assert public['validation_curve']==sample['validation_curve']
    assert public['fit_hashes']==sample['fit_hashes']
    metadata={'raw_rows':377575,'dictionary_comparison':{'SERIALNO':{'identical_text':False,'2017_sha256':'a','2018_sha256':'b'}}}
    assert project_public(metadata)==(metadata,[])
    assert project_public({'array_hashes':{'SERIALNO':'a'*64,'SPORDER':'b'*64}})[1]==[]
    assert len(removed)==4 and removed[0]['sha256']
    assert canonical_gzip(b'aggregate')==canonical_gzip(b'aggregate')
    assert classify('seed_0/E/metrics.json')=='gzip'
    assert classify('seed_0/spectral_C1/selection_before_test.json')=='gzip'
    assert classify('seed_0/E/audits/audit_selection.json')=='gzip'
    assert classify('seed_0/maps.joblib')=='local'
    assert classify('seed_0/matrix_diagnostics.json')=='numerical'
    assert classify('seed_0/E/audits/fitted/model.joblib')=='local'
    assert classify('unknown.csv')=='local'
    assert classify('PER_SEED.csv.gz')=='gzip_input'
    assert classify('unknown.csv.gz')=='local'
    assert classify('NEGATIVE_INCREMENT_COUNTS.csv.gz')=='gzip_input'
    assert classify('verification_sources/verify_surrogates.py')=='public'
    assert classify('verification_sources/unknown.py')=='local'
    assert classify('data/raw.csv')=='local'
    assert classify('PROJECTION_LAYOUT_REPAIR.json')=='public'
    assert classify('seed_0/E/projection_layout_repair.json')=='public'
    assert classify('seed_0/E/revisions/pre_projection_layout/metrics.json')=='local'
    try: guard_root(Path('/Users/nathansamson/PCRL/results/old_study'))
    except ValueError: pass
    else:raise AssertionError('Historical root accepted')
    # End-to-end deterministic package/replay in an isolated synthetic fixture.
    import tempfile
    import contextlib
    old_root,old_default=ROOT,DEFAULT
    try:
        with tempfile.TemporaryDirectory(prefix='pcrl-publication-selftest-') as temporary:
            ROOT=Path(temporary);DEFAULT=ROOT/'results'/STUDY
            (DEFAULT/'seed_0/H').mkdir(parents=True)
            (DEFAULT/'seed_0/H/metrics.json').write_bytes(canonical_json({'support':[3,2],'loss':.4}))
            (DEFAULT/'seed_0/matrix_diagnostics.json').write_bytes(canonical_json(sample))
            (DEFAULT/'seed_0/maps.joblib').write_bytes(b'opaque fitted state must not be deserialized')
            (DEFAULT/'unknown.csv').write_text('SERIALNO,private\n123,456\n')
            aggregate_csv=b'seed,loss\n0,0.123456789012345\n'
            (DEFAULT/'PER_SEED.csv.gz').write_bytes(gzip.compress(aggregate_csv,mtime=99))
            with contextlib.redirect_stdout(io.StringIO()):
                package(DEFAULT)
                first={p.relative_to(DEFAULT).as_posix():p.read_bytes() for p in DEFAULT.rglob('*') if p.is_file()}
                package(DEFAULT)
                second={p.relative_to(DEFAULT).as_posix():p.read_bytes() for p in DEFAULT.rglob('*') if p.is_file()}
                assert first==second,'Unchanged input produced different publication bytes'
                verified=verify_public(DEFAULT)
                assert verified['model_files_read']==0
                archived=DEFAULT/'PUBLICATION_EVIDENCE/PER_SEED.csv.gz'
                assert archived.exists() and not archived.with_suffix('.gz.gz').exists()
                assert gzip.decompress(archived.read_bytes())==aggregate_csv
                assert archived.read_bytes()[4:8]==b'\0\0\0\0'
                assert str(archived.relative_to(ROOT)) in (DEFAULT/'PUBLICATION_FILES.txt').read_text()
                names=(DEFAULT/'PUBLICATION_FILES.txt').read_text()
                assert 'maps.joblib' not in names and 'unknown.csv' not in names and 'matrix_diagnostics.json' not in names
                (DEFAULT/'PUBLICATION_EVIDENCE/seed_0/H/metrics.json.gz').write_bytes(b'corrupt')
                try:verify_public(DEFAULT)
                except AssertionError:pass
                else:raise AssertionError('Corrupt public evidence accepted')
    finally:ROOT,DEFAULT=old_root,old_default
    print('Packaging self-test passed: projection, allowlist, gzip/root guards, deterministic end-to-end package, public-only verification, corruption rejection')



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.verify:verify_public(a.out)
    else:package(a.out,a.dry_run)
