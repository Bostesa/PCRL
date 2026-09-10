"""Independent frozen-object replay on existing historical fixtures; never fresh scoring.

Loads validation-selected saved states, reconstructs releases and raw projections,
and independently recomputes historical metrics. No old runner is invoked.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.acs_transfer_data import CovariatePreprocessor, load_cohort, split_households
from experiments.acs_selective_teachers import load_teachers
from scripts import verify_acs_bottleneck_scores as check

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('H', 'E', 'A0', 'L025', 'L20', 'J')
TARGETS = ('SEX','RAC1P','income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
SCOPES = ('standard_independent', 'expanded_independent', 'expanded_catchup')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def literal_aux(checkpoint, pca):
    state = checkpoint['model_state']
    x = torch.tensor((pca.astype(float)-state['input_mean'].numpy())/state['input_scale'].numpy(), dtype=torch.float32)
    with torch.no_grad():
        h = torch.relu(torch.nn.functional.linear(x, state['branch.mapper.0.weight'], state['branch.mapper.0.bias']))
        h = torch.nn.functional.linear(h, state['branch.mapper.2.weight'], state['branch.mapper.2.bias'])
        probabilities = []
        for task in ('income_binary', 'civilian_at_work'):
            p = torch.sigmoid(torch.nn.functional.linear(h, state[f'branch.heads.{task}.weight'], state[f'branch.heads.{task}.bias']))
            probabilities.append(torch.cat((1-p, p), 1).double().numpy())
    return h.double().numpy(), probabilities


def local_path(path):
    path = Path(path)
    if not path.is_absolute():
        return ROOT/path
    # Recover paths by repository-relative manifest identity, never by basename.
    text = str(path)
    marker = '/PCRL/'
    return ROOT/text.split(marker, 1)[1] if marker in text else path


class HistoricalReplay:
    def __init__(self, parent):
        self.parent = parent
        self.files = {}
        self.models = {}
        self.outputs = {}
        self.scores = {}
        self.report = {'started_utc': now(), 'scope': 'Historical plumbing replay only; no unused-household scoring or generalization confirmation.',
                       'scientific_model_fits': 0, 'optimizer_updates': 0, 'candidate_reselections': 0,
                       'new_outcomes_scored': 0, 'threads': 1, 'inference_workers': 1,
                       'model_output_parity': 'bitwise array equality on identical historical cached inputs',
                       'metric_absolute_tolerance': check.TOL, 'systems': 0, 'selected_slots': 0,
                       'selected_records': 0, 'selected_prediction_arrays': 0, 'score_dictionaries': 0,
                       'anchor_arrays': 0, 'release_arrays': 0, 'teacher_arrays': 0,
                       'raw_pca_transform_arrays': 0, 'H_projection_witnesses': 0,
                       'selected_singleton_witnesses': 0, 'B_selected_parities': 0,
                       'same_weighting_candidate_identity_checks': 0, 'systems_detail': []}
        self.started = time.perf_counter()

    def checked(self, path):
        path = local_path(path)
        digest = check.sha(path)
        key = str(path.relative_to(ROOT))
        assert self.files.get(key, digest) == digest, ('historical file changed', key)
        self.files[key] = digest
        return path

    def read(self, path):
        return json.loads(self.checked(path).read_text())

    def arrays(self, path):
        with np.load(self.checked(path), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    def model(self, path, metadata):
        path = local_path(path)
        if str(path) not in self.models:
            self.checked(path/'metadata.json')
            self.checked(path/'preprocessing.npz')
            self.checked(path/('model.pt' if metadata['family'] == 'mlp' else 'model.joblib'))
            loaded = check.load_inference(path, metadata)
            self.models[str(path)] = loaded
        return self.models[str(path)]

    def predict(self, path, metadata, x):
        predict, mean, scale, state = self.model(path, metadata)
        key = (str(local_path(path)), check.array_hash(x))
        if key not in self.outputs:
            before = (check.array_hash(mean), check.array_hash(scale), check.state_hash(state) if state else None)
            tick = time.perf_counter()
            actual = predict(x)
            elapsed = time.perf_counter()-tick
            after = (check.array_hash(mean), check.array_hash(scale), check.state_hash(state) if state else None)
            assert before == after, 'in-memory frozen inference state changed'
            self.outputs[key] = actual
            if 'first_inference_seconds' not in self.report:
                self.report.update(first_inference_seconds=elapsed, first_inference_rows=len(x),
                                   first_inference_completed_utc=now(),
                                   finite_work_projection_seconds=elapsed*18*(5+11*2*3)*2,
                                   projection_scope='Conservative undeduplicated historical selected prediction calls; excludes I/O, release replay and metric overhead.')
        return self.outputs[key]

    def score(self, seed, pool, target, y, p, weight, expected, context):
        key = (seed, pool, target, check.array_hash(p), weight is not None)
        if key not in self.scores:
            self.scores[key] = check.independent_scores(y, p, 9 if target == 'RAC1P' else 2, weight)
        check.compare(self.scores[key], expected, context)
        self.report['score_dictionaries'] += 1

    def run(self):
        cfg = self.read(self.parent/'config.json')
        identity = self.read(self.parent/'PREFIT_IDENTITY.json')
        gate = self.read(self.parent/'RELEASE_MANIFEST.json')
        assert gate['all18_frozen'] and gate['training_closed']
        pcfg = self.read(ROOT/cfg['parent_results']/'config.json')
        raw = self.checked(ROOT/pcfg['raw_path'])
        assert check.sha(raw) == 'dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0'
        frame, cohort = load_cohort(raw, pcfg['sample_cap'], pcfg['sample_seed'])
        self.report['historical_fixture'] = cohort
        for seed in (0, 1, 2):
            print(f'{now()} historical seed {seed}', flush=True)
            directory = self.parent/f'seed_{seed}'
            pools = split_households(frame, seed)
            frames = {pool: frame.iloc[ix] for pool, ix in pools.items()}
            label = {(pool, target): check.labels(f, target) for pool, f in frames.items() for target in TARGETS}
            pca = self.arrays(directory/'pca.npz')
            anchors = self.arrays(directory/'anchors.npz')
            teacher_cache = self.arrays(directory/'teacher.npz')
            rows = self.arrays(directory/'split_rows.npz')
            for path, digest in identity[str(seed)]['cached_inputs'].items():
                assert check.sha(self.checked(path)) == digest
            source = ROOT/cfg['parent_results']/f'seed_{seed}'
            premeta = self.read(source/'preprocessing.json')
            pre = CovariatePreprocessor()
            pre.numeric, pre.categories, pre.feature_names = (premeta[k] for k in ('numeric','categories','feature_names'))
            maps = joblib.load(self.checked(source/'release_maps.joblib'))
            frozen_hash = joblib.hash((pre, maps))
            teacher_dir = ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'teachers'
            for path in teacher_dir.iterdir():
                if path.is_file(): self.checked(path)
            teacher = load_teachers(teacher_dir)['maps']['E']
            assert teacher.fingerprint() == identity[str(seed)]['E_fingerprint']
            for pool in pools:
                assert np.array_equal(rows[pool], frames[pool]._raw_row.to_numpy())
                transformed = maps['pca'].transform(pre.transform(frames[pool])).astype(np.float32)
                assert np.array_equal(transformed, pca[pool]), (seed, pool, 'PCA transform')
                self.report['raw_pca_transform_arrays'] += 1
                assert np.array_equal(teacher.apply(pca[pool][:,:16]), teacher_cache[pool])
                self.report['teacher_arrays'] += 1
            assert frozen_hash == joblib.hash((pre, maps))
            for target, record in identity[str(seed)]['anchors'].items():
                path = local_path(record['path'])
                meta = self.read(path/'metadata.json')
                view, col = {'income_binary':('A',0), 'civilian_at_work':('A',2), 'public_coverage':('B',0)}[target]
                for pool in pools:
                    actual = self.predict(path, meta, pca[pool])
                    assert np.array_equal(actual, anchors[pool+'/'+view][:,col:col+2])
                    self.report['anchor_arrays'] += 1
            h_predictions = None
            h_rows = None
            b_predictions = {}
            for arm in ARMS:
                dest = directory/arm
                print(f'{now()} seed {seed} {arm}', flush=True)
                selected = self.read(dest/'selection_before_test.json')
                audits = self.read(dest/'audits/audit_selection.json')
                all_metrics = self.read(dest/'metrics.json')['raw_metrics']
                metrics = [row for row in all_metrics if row['selected_scopes']]
                predictions = self.arrays(dest/'predictions.npz')
                checkpoint_path = self.checked(directory/'training'/arm/'final.pt')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                checkpoint_state_hash = check.state_hash(checkpoint['model_state']) if checkpoint['model_state'] else None
                release_path = directory/'training'/arm/'releases.npz'
                released = self.arrays(release_path)
                frozen = gate['seeds'][str(seed)]['conditions'][arm]
                assert check.sha(checkpoint_path) == frozen['checkpoint_sha256']
                assert check.sha(release_path) == frozen['release_sha256']
                for pool, x in pca.items():
                    aa, bb = anchors[pool+'/A'], anchors[pool+'/B']
                    aux, ps = (literal_aux(checkpoint, x) if arm not in ('H','E') else (teacher_cache[pool].astype(float), None) if arm == 'E' else (None,None))
                    a = aa if aux is None else np.column_stack((aa, aux))
                    wire = {'A':a, 'B':bb, 'AB':np.column_stack((a,bb))}
                    derived = None if ps is None else {'A':np.column_stack((aa,*ps)), 'B':bb, 'AB':np.column_stack((aa,*ps,bb))}
                    for space, views in (('wire',wire),('derived',derived)):
                        if views is None: continue
                        for view, actual in views.items():
                            assert actual.dtype == np.float64 and np.array_equal(actual, released[f'{space}/{view}/{pool}'])
                            self.report['release_arrays'] += 1
                    assert np.array_equal(wire['A'][:,:4], aa) and np.array_equal(wire['B'], bb)
                assert checkpoint_state_hash == (check.state_hash(checkpoint['model_state']) if checkpoint['model_state'] else None)
                slots = sum(len(row['selected_scopes']) for row in metrics)
                assert slots == 5+11*2*3, (seed,arm,slots)
                self.report['selected_slots'] += slots
                for row in metrics:
                    role, view, target, cid, budget = (row[k] for k in ('role','view','target','candidate_id','audit_budget'))
                    if role == 'audit':
                        record = audits['candidates'][str(budget)][view+'/'+target][cid]
                        path = local_path(record['base_candidate_directory'])
                        metadata = self.read(path/'metadata.json')
                        assert check.sha(path/'metadata.json') == record['base_metadata_sha256']
                        space, cols = record['space'], record['projection_columns']
                        for scope in row['selected_scopes']:
                            assert audits['selections'][str(budget)][view+'/'+target][scope] == cid
                    else:
                        record = selected['utility_metadata'][view+'/'+target]
                        path = local_path(record['alias'])/cid if 'alias' in record else dest/'fitted/utility'/view/target/cid
                        metadata = self.read(path/'metadata.json')
                        assert metadata == record['candidates'][cid]
                        assert selected['utility'][view+'/'+target] == cid
                        space, cols = 'wire', None
                    self.report['same_weighting_candidate_identity_checks'] += 1
                    for split, pool in (('validation','downstream_validation' if role=='utility' else 'attacker_validation'),('test','test')):
                        y, mask = label[pool,target]
                        # Project raw released columns BEFORE ancestor normalization.
                        x = released[f'{space}/{view}/{pool}'][mask]
                        if cols is not None: x = x[:,cols]
                        actual = self.predict(path, metadata, np.ascontiguousarray(x))
                        key = f'{role}/{view}/{target}/{budget}/{cid}/{split}'
                        assert np.array_equal(actual, predictions[key]), (seed,arm,key)
                        self.report['selected_prediction_arrays'] += 1
                        for suffix, w in (('',None),('_person_weighted',frames[pool].PWGTP.to_numpy(float)[mask])):
                            self.score(seed,pool,target,y[mask],actual,w,row['scores'][split+suffix],f'{seed}/{arm}/{key}{suffix}')
                        if view == 'B':
                            for scope in row['selected_scopes']:
                                bk = (role,target,budget,scope,split)
                                if arm == 'H': b_predictions[bk] = actual
                                else:
                                    assert np.array_equal(actual,b_predictions[bk])
                                    self.report['B_selected_parities'] += 1
                        if role == 'audit' and cid.startswith('inherited_'):
                            sv, scid = cid.removeprefix('inherited_').split('__',1)
                            assert np.array_equal(actual,predictions[f'audit/{sv}/{target}/{budget}/{scid}/{split}'])
                            self.report['selected_singleton_witnesses'] += 1
                    self.report['selected_records'] += 1
                if arm == 'H':
                    h_predictions, h_rows, h_audits = predictions, metrics, audits
                else:
                    # Every selected H attacker is available, regardless of which augmented candidate won validation.
                    for row in h_rows:
                        if row['role'] != 'audit': continue
                        view,target,cid,budget = (row[k] for k in ('view','target','candidate_id','audit_budget'))
                        record = h_audits['candidates'][str(budget)][view+'/'+target][cid]
                        path = local_path(record['base_candidate_directory'])
                        metadata = self.read(path/'metadata.json')
                        for split,pool in (('validation','attacker_validation'),('test','test')):
                            y,mask = label[pool,target]
                            raw_wire = released[f'wire/{view}/{pool}'][mask]
                            raw_h = raw_wire if view=='B' else raw_wire[:,:4] if view=='A' else raw_wire[:,[0,1,2,3,20,21]]
                            if record['projection_columns'] is not None:
                                raw_h = raw_h[:,record['projection_columns']]
                            actual = self.predict(path, metadata, np.ascontiguousarray(raw_h))
                            assert np.array_equal(actual,h_predictions[f'audit/{view}/{target}/{budget}/{cid}/{split}'])
                            self.report['H_projection_witnesses'] += 1
                complete = self.read(dest/'complete.json')
                for path, digest in complete['files_sha256'].items():
                    assert check.sha(self.checked(path)) == digest
                self.report['systems'] += 1
                self.report['systems_detail'].append({'seed':seed,'condition':arm,'selected_slots':slots,'selected_records':len(metrics),'passed':True})
        assert not check.errors, check.errors[:5]
        assert self.report['systems'] == 18 and self.report['selected_slots'] == 1278
        print(f'{now()} checking {len(self.files)} historical file hashes again', flush=True)
        for path,digest in self.files.items():
            assert check.sha(ROOT/path) == digest, ('changed historical artifact',path)
        self.report.update(completed_utc=now(),runtime_seconds=time.perf_counter()-self.started,
                           unique_loaded_models=len(self.models),unique_inference_outputs=len(self.outputs),
                           unique_score_dictionaries=len(self.scores),numeric_comparisons=check.numeric_comparisons,
                           max_score_absolute_error=check.max_error,historical_files_unchanged=len(self.files),
                           historical_files_manifest_sha256=hashlib.sha256(json.dumps(self.files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                           script_sha256=check.sha(__file__),passed=True)
        return self.report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1')
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Preserve completed evidence; use a new explicit report path')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1), torch.inference_mode():
        replay = HistoricalReplay(args.parent)
        report = replay.run()
    tmp = args.report.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    tmp.replace(args.report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
