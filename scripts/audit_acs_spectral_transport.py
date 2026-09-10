"""Prepare/verify a transport admission package, never fit or score models.

Run from the new worktree. The historical checkout is read-only. Network
retrieval is explicit via --download; raw files and arrays remain in data/.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from experiments import acs_spectral_transport as t
from experiments.acs_transfer_data import sha_file, array_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/redesign_20260910_acs_residual_spectral_v1'
LOCAL=ROOT/'data/acs_spectral_transport'
SOURCES={
 'dictionary_2017.txt':'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.txt',
 'dictionary_2018.txt':'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt',
 'csv_pca_2017.zip':'https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/csv_pca.zip',
}


# Freeze the exact public inputs and manually reviewed prior-use matches.
EXPECTED_SOURCES={
 'dictionary_2017.txt':'ea5bcde2a07b0c76ac145fac108e9013480b686a9d70afc4e3f4a6512ccbfd69',
 'dictionary_2018.txt':'843a18e15b6cb7c7326261d478947a1415c4674c62419a8d01946a5a5df4d5fd',
 'csv_pca_2017.zip':'d28c43dabd5fdc0abcba3d0941101bf6eceda4a84c1bdd269982adb8b89d6617'}
REVIEWED_YEAR_MATCH_HASH='a761693d003e5abb8865d377ee0a79c7ffbfcdb781e2838579d1339f2c160320'


def require_reviewed_history(prior):
    digest=hashlib.sha256(json.dumps(prior['year_token_matches'],sort_keys=True).encode()).hexdigest()
    if digest != REVIEWED_YEAR_MATCH_HASH:
        raise ValueError('Historical year-token matches changed: independent review required before admission')
    return {'matches_sha256':digest,'reviewed_matches':11048,
      'disposition':{'json_counts':8268,'csv_counts':2772,'markdown_support_counts':2,'unrelated_citations':6},
      'json_count_fields':{'n':4572,'validation_rows':3496,'predicted_support':130,'valid':40,'optimizer_steps':28,'actual_terminal_checkpoints':2},
      'csv_count_fields':{'n':2664,'predicted_support':108},
      'documented_2017_or_2016_acs_use':False}


def save_json(path,value,verify=False):
    if verify:
        assert json.loads(path.read_text())==value, 'Changed report: '+str(path)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def historical_inventory(historical):
    # Search all available code/configuration/recorded-use text, including
    # ignored local result manifests and compressed evidence. Record the scope.
    inventory=subprocess.check_output(['rg','--files','--hidden','--no-ignore',
        '-g','!.git/**','-g','!.venv/**','-g','!data/**','-g','!node_modules/**',
        '-g','!__pycache__/**'],cwd=historical,text=True).splitlines()
    extensions=('.py','.json','.json.gz','.yaml','.yml','.toml','.md','.sh','.txt','.csv','.csv.gz')
    paths=sorted(p for p in inventory if p.endswith(extensions))
    year=re.compile(r'(?<![\w.])(?:2016|2017)(?![\w.])')
    matches=[]; configs=[]; scanned=[]
    for relative in paths:
        path=historical/relative
        try:
            text=(gzip.open(path,'rt',encoding='utf-8').read() if relative.endswith('.gz') else path.read_text())
        except (UnicodeError,OSError):continue
        scanned.append(relative)
        if any(s in path.name.lower() for s in ('config','manifest')):
            configs.append({'path':relative,'sha256':sha_file(path)})
        for n,line in enumerate(text.splitlines(),1):
            if year.search(line):
                # A decimal metric/hash is excluded lexically; exact integer
                # support/step counts are retained for transparent review.
                matches.append({'path':relative,'line':n,'text':line[:1500]})
    evidence=['pcrl/data/folktables.py','experiments/acs_transfer_data.py',
      'experiments/run_acs_transfer.py','experiments/run_v2_dataset.py',
      'scripts/verify_acs_bottleneck_scores.py',
      'results/redesign_20260907_acs_transfer_v1/config.json',
      'results/redesign_20260910_acs_locked_evaluation_v1/DATA_USE_INVENTORY.json',
      'results/redesign_20260910_acs_locked_evaluation_v1/DATA_INDEPENDENCE.md']
    # Search data/cache names only: never read any prospective extra release.
    data_names=subprocess.check_output(['rg','--files','--hidden','--no-ignore','data'],cwd=historical,text=True).splitlines()
    return {'historical_checkout':str(historical),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=historical,text=True).strip(),
      'tracked_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=historical,text=True).strip(),
      'available_paths':len(inventory),'text_files_scanned':len(scanned),
      'scanned_paths_sha256':hashlib.sha256('\n'.join(scanned).encode()).hexdigest(),
      'year_token_pattern':year.pattern,'year_token_matches':matches,
      'config_and_manifest_hashes':configs,
      'historical_data_paths_with_year':sorted(p for p in data_names if re.search(r'(?<![0-9])201[678](?![0-9])',p)),
      'direct_pipeline_and_exhaustion_evidence':{p:sha_file(historical/p) for p in evidence},
      'scope_limitation':'Available checkout evidence only; cannot establish absence of undocumented external research or identify the same human across annual public files.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-root',type=Path,default=Path('/Users/nathansamson/PCRL'))
    parser.add_argument('--download',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args(); official=LOCAL/'official';official.mkdir(parents=True,exist_ok=True)
    if args.download:
        for name,url in SOURCES.items():
            path=official/name
            if not path.exists(): urllib.request.urlretrieve(url,path)
    for name in SOURCES:
        if not (official/name).exists(): raise FileNotFoundError('Run --download: '+name)
        if sha_file(official/name)!=EXPECTED_SOURCES[name]: raise ValueError('Changed official input: '+name)
    csv=official/'psam_p06.csv'
    if not csv.exists():
        with zipfile.ZipFile(official/'csv_pca_2017.zip') as z:
            with z.open('psam_p06.csv') as src,csv.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src,dst)
    # Archive integrity and exact selected member equality are checked on every run.
    with zipfile.ZipFile(official/'csv_pca_2017.zip') as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('psam_p06.csv')).hexdigest()==sha_file(csv)
    dictionaries={y:(official/f'dictionary_{y}.txt').read_text() for y in (2017,2018)}
    comparison={}
    for col in (*t.REQUIRED,'ADJINC'):
        a,b=[t.dictionary_block(dictionaries[y],col) for y in (2017,2018)]
        comparison[col]={'identical_text':a==b,'2017_sha256':hashlib.sha256(a.encode()).hexdigest(),
                         '2018_sha256':hashlib.sha256(b.encode()).hexdigest()}
    # Categorical meanings must match exactly; no code mapping is allowed.
    for col in ('SCHL','MAR','RELP','CIT','DIS','DEAR','DEYE','DREM','ESR','PUBCOV','MIG','SEX','RAC1P'):
        assert comparison[col]['identical_text'], 'Changed categorical definition: '+col
    raw=pd.read_csv(csv,usecols=list(t.REQUIRED),dtype={'SERIALNO':str,'SPORDER':str})
    schema=t.validate_frame(raw,2017);frame,cohort=t.eligible_cohort(raw,2017)
    partitions=t.partition_households(frame,2017)
    prior=historical_inventory(args.historical_root)
    prior['review']=require_reviewed_history(prior)
    save_json(OUT/'DATA_ADMISSION_PRIOR_USE.json',prior,args.verify)
    groups=[];reports={}
    for name,rows in partitions.items():
        f=frame.iloc[rows]; household=f.SERIALNO.to_numpy(dtype=str)
        groups.append(set(household)); labels=t.fixed_labels(f)
        arrays={'raw_rows':f._raw_row.to_numpy(np.int64),
            'SERIALNO':household,'SPORDER':f.SPORDER.to_numpy(dtype=str),
            'PWGTP':f.PWGTP.to_numpy(float),'raw_features':f[list(t.FEATURES)].to_numpy(float),**labels}
        path=LOCAL/f'2017_{name}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                assert set(old.files)==set(arrays)
                for key,value in arrays.items():
                    assert array_hash(old[key])==array_hash(value), 'Changed local array '+name+'/'+key
        elif args.verify:raise FileNotFoundError(path)
        else:np.savez_compressed(path,**arrays)
        reports[name]={'rows':len(f),'households_or_gq_persons':len(set(household)),
           'local_path':str(path.relative_to(ROOT)),'file_sha256':sha_file(path),
           'array_hashes':{k:array_hash(v) for k,v in arrays.items()},'support':t.support(f)}
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    # These public identifiers have different annual formats. Equality cannot
    # serve as evidence of longitudinal person/household non-overlap.
    report={'status':'admitted_for_future_file_level_transport_only', 'year':2017,
      'fallback_2016':'not accessed: first candidate schema/provenance suitable',
      'cohort':cohort,'eligibility':'19 <= AGEP <= 34 and PWGTP > 0; all eligible rows; whole SERIALNO groups; no new GQ exclusion',
      'features':list(t.FEATURES),'targets':t.CLASSES,
      'income_definition':'Original raw PINCP > 50000; no ADJINC or between-year inflation adjustment; range [-19998,4209995]',
      'fitting':{'preprocessors':0,'PCA':0,'anchors':0,'representations':0,'heads':0,'attacks':0},
      'prediction_calls':0,'model_outcomes_scored':0,'final_partition_model_scores':0,
      'partition_rule':{'salt':t.PARTITION_SALT,'hash':'SHA256(UTF8(salt|year|06|SERIALNO)); lexicographic digest then SERIALNO',
        'fractions':dict(zip(t.POOLS,t.FRACTIONS)),'boundaries':'numpy rint cumulative fractions times group count',
        'label_blind':True,'seed_shared_across_future_system_seeds':True},
      'schema':schema,'dictionary_comparison':comparison,'partitions':reports,
      'retrieval_date_utc':'2026-09-10',
      'sources':{name:{'url':url,'sha256':sha_file(official/name)} for name,url in SOURCES.items()},
      'raw_csv_sha256':sha_file(csv),'prior_use_report_sha256':sha_file(OUT/'DATA_ADMISSION_PRIOR_USE.json'),
      'source_sha256':{p:sha_file(ROOT/p) for p in ['experiments/acs_spectral_transport.py','scripts/audit_acs_spectral_transport.py','tests/test_acs_spectral_transport.py','experiments/acs_transfer_data.py']},
      'limitations':[
        'Temporal transport to earlier survey year, not confirmation on untouched 2018 households.',
        'No documented 2017 research use found in audited repository evidence; undocumented external use cannot be excluded.',
        'Annual public SERIALNO/SPORDER are not stable longitudinal identifiers. Cross-year same-human overlap cannot be proved absent, especially for movers.',
        'SERIALNO denotes housing unit or GQ person; public data do not expose a common facility identifier for every GQ person. Group disjointness is at published identifier level.',
        'Nominal income threshold, reference-period timing and top/bottom coding may shift meaning or prevalence across years; no silent adjustment performed.',
        'RAC1P Alaska Native alone has zero attacker-validation support and one final-evaluation person. Full-schema macro AUROC and per-class recall/AUROC must be undefined where required support is absent, never repaired by category collapse.',
        'Only aggregate class/weight support was inspected. Future representation choices and head selection must precede final scoring.',
        'PWGTP is retained for weighted sensitivity; this preparation makes no survey-design variance or population representativeness claim.']}
    save_json(OUT/'DATA_ADMISSION.json',report,args.verify)
    print(json.dumps({'status':report['status'],'cohort':cohort,'partitions':{k:v['rows'] for k,v in reports.items()},'prior_matches':len(prior['year_token_matches']),'verified':args.verify}))

if __name__=='__main__':main()
