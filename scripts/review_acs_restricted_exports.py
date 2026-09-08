"""Read-only standard-library checks of published CSVs against original JSON."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def review(out):
    started = time.perf_counter()
    cache, selections, inputs, comparisons = {}, {}, set(), 0

    def decode(value):
        if value == '': return None
        if value in ('True', 'False'): return value == 'True'
        try: return json.loads(value)
        except ValueError: return value

    def match(actual, expected):
        nonlocal comparisons
        comparisons += 1
        if isinstance(expected, float):
            assert actual is not None and math.isclose(actual, expected, rel_tol=0, abs_tol=2e-12), (actual, expected)
        else:
            assert actual == expected, (actual, expected)

    def source(row):
        path = ROOT/row['origin_metrics']
        if path not in cache:
            cache[path] = json.loads(path.read_text())['raw_metrics']
            selections[path] = json.loads((path.parent/'selection_before_test.json').read_text())
            inputs.update((path, path.parent/'selection_before_test.json'))
        expected = next(r for r in cache[path] if r['role'] == row['role'] and r['release'] == row['origin_release']
                        and r['target'] == row['target'] and r['candidate_id'] == row['candidate_id']
                        and (r.get('audit_budget') or None) == decode(row['audit_budget']))
        for key in ('seed', 'role', 'target', 'candidate_id', 'family', 'selected', 'independent_selected', 'selected_within_family', 'auc_selected'):
            match(decode(row[key]), expected[key])
        selected = selections[path]
        record = selected if row['role'] == 'transfer' else selected['budgets'][row['audit_budget']]
        key = f"{row['role']}/{row['origin_release']}/{row['target']}"
        return expected, record['fitting_records'][key]['candidates'][row['candidate_id']]

    splits = {'validation': 'validation', 'development_evaluation': 'test',
              'validation_person_weighted': 'validation_person_weighted',
              'development_evaluation_person_weighted': 'test_person_weighted'}
    counts = {}
    for filename in ('PER_TARGET.csv', 'PER_CLASS.csv', 'FITTING.csv', 'CURVES.csv'):
        path = out/filename
        inputs.add(path)
        with path.open() as handle: rows = list(csv.DictReader(handle))
        counts[filename] = len(rows)
        for row in rows:
            raw, metadata = source(row)
            if filename == 'PER_TARGET.csv':
                for key, value in raw[splits[row['split']]].items():
                    if key != 'per_class': match(decode(row[key]), value)
            elif filename == 'PER_CLASS.csv':
                data = raw[splits[row['split']]]['per_class'][int(row['class_index'])]
                for key, value in data.items(): match(decode(row[key]), value)
            elif filename == 'FITTING.csv':
                for key in ('fit_rows', 'fit_support', 'fit_coverage_complete', 'optimizer_steps', 'selected_epoch',
                            'selected_optimizer_steps', 'training_row_exposures', 'schedule_hash', 'initialization_seed',
                            'schedule_seed', 'restart_index', 'parameters', 'fit_runtime_seconds'):
                    if key in metadata: match(decode(row[key]), metadata[key])
            else:
                curve = next(value for value in metadata['validation_curve'] if value['epoch'] == int(row['epoch']))
                for key, value in curve.items():
                    if key in row: match(decode(row[key]), value)
                match(decode(row['checkpoint_selected']), int(row['epoch']) == metadata['selected_epoch'])
    pairs = 0
    for teacher in ('E', 'S'):
        for access in ('F', 'K'):
            for seed in range(3):
                path = out/f'{teacher}_{access}'/f'seed_{seed}'/'predictions.npz'
                inputs.add(path)
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        if name.startswith('budget120/'):
                            assert archive.read(name) == archive.read(name.replace('budget120/', 'budget360/', 1)), (path, name)
                            pairs += 1
    assert comparisons == 842940 and pairs == 672
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'Standard-library CSV-to-original-JSON selections, scores, classes, fits and curves; exact byte comparison of all main120/360 prediction-array pairs. No model fitting or numerical-library replay.',
            'source_sha256': sha(__file__), 'report_generator_sha256': sha(ROOT/'scripts/summarize_acs_restricted.py'),
            'metric_absolute_tolerance': 2e-12, 'export_rows': counts,
            'scalar_and_array_fields_compared': comparisons, 'main_nested_prediction_pairs_byte_identical': pairs,
            'input_sha256': {str(path.relative_to(ROOT)): sha(path) for path in sorted(inputs)},
            'runtime_seconds': time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/redesign_20260908_acs_restricted_inputs_v1')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists(): raise FileExistsError(args.report)
    result = review(args.out.resolve())
    if args.report:
        with args.report.open('x') as handle: handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    else: print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
