"""Controlled eligibility extraction and DEVELOPMENT role export (runs only after the committed SCIENCE_LOCK).

This is AUTOMATED ELIGIBILITY EXTRACTION, not metadata admission: it reads the eligibility variables
(including target validity: ESR, OCCP, PINCP, SEX) for every person record of the admitted file, applies the
frozen predicate (schema.PREDICATE) and household roles, and writes:
  - DEVELOPMENT role exports (features, raw permitted inputs, task labels, SEX, weights) for MODEL_FIT,
    CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION; DEV_REPORT features/inputs only, with its
    truth (labels, SEX, weights) written to a separate DEV_REPORT_TRUTH directory that loaders refuse to
    open before the committed REPORTING_LOCK;
  - SEALED_ASSESSMENT: counts and a sha256 of its sorted household keys ONLY (no features, labels or
    predictions are written or computed beyond the eligibility predicate);
  - capped-out development households: counted, never exported.
Receipts carry counts, hashes and encodings only; no row is printed.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import io
import json
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import schema

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
EXPORTS = PRIV / 'exports'
DEV_ROLES = [r for r, _ in schema.DEV_SHARES]
NEEDED = ['RT', 'SERIALNO', 'SPORDER', 'ST', 'ADJINC', 'PWGTP', 'AGEP', 'COW', 'SCHL', 'MAR', 'POBP', 'WKHP', 'ESR', 'OCCP', 'SEX', 'PINCP', 'RELSHIPP']


def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def int_id(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def committed(path):
    out = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%H', '--', str(path)], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', str(path)], capture_output=True, text=True)
    return out.stdout.strip() or None, not dirty.stdout.strip()


def require_lock(lock_path, lock_sha):
    lock_path = Path(lock_path)
    if sha_file(lock_path) != lock_sha:
        raise PermissionError('science lock hash differs')
    lock = json.loads(lock_path.read_text())
    if lock.get('status') != 'SCIENCE_LOCKED':
        raise PermissionError('extraction requires status SCIENCE_LOCKED')
    commit, clean = committed(lock_path)
    if not commit or not clean:
        raise PermissionError('extraction requires the science lock committed and unmodified')
    return lock, commit


def eligibility(row, occ_map):
    """Return (reason or None, parsed). Reasons are evaluated in this fixed order."""
    if row['RT'] != 'P':
        return 'record_type', None
    if row['ST'] != schema.STATE_FIPS:
        return 'state', None
    s = row['SERIALNO']
    if not (len(s) == 13 and s[:6] == schema.YEAR + 'HU' and s[6:].isdigit()):
        return 'group_quarters_or_nonhousing_serial', None
    try:
        age = int(row['AGEP'])
    except ValueError:
        return 'age_invalid', None
    if not 25 <= age <= 64:
        return 'age_outside_25_64', None
    if row['ESR'] not in ('1', '2'):
        return 'not_civilian_employed', None
    w = row['WKHP'].strip()
    if not w.isdigit() or not 1 <= int(w) <= 99:
        return 'hours_blank_or_zero', None
    occ = row['OCCP'].strip()
    if occ not in occ_map['mapping']:
        return 'occupation_blank_military_or_unmapped', None
    p = row['PINCP'].strip()
    try:
        pincp = int(p)
    except ValueError:
        return 'income_blank', None
    if row['ADJINC'].strip() != str(schema.ADJINC_2019):
        return 'adjinc_differs', None
    if row['SEX'] not in schema.SEX_CODE:
        return 'sex_invalid', None
    for c in schema.INPUTS['categorical']:
        if not row[c].strip():
            return f'input_blank_{c}', None
    parsed = {'serial': s, 'sporder': row['SPORDER'], 'AGEP': age, 'WKHP': int(w),
              **{c: row[c].strip() for c in schema.INPUTS['categorical']},
              # exact integer form of PINCP * ADJINC / 1e6 > 50000 (no floating-point rounding at the threshold)
              'income': int(pincp * schema.ADJINC_2019 > 50000 * 1_000_000),
              'occupation': occ_map['mapping'][occ], 'sex': schema.SEX_CODE[row['SEX']], 'pwgtp': int(row['PWGTP']),
              'relshipp': row['RELSHIPP']}
    return None, parsed


def fit_encoding(people):
    enc = {'numeric': {}, 'categorical': {}, 'min_count': schema.CATEGORY_MIN_COUNT}
    for c in schema.INPUTS['numeric']:
        v = np.asarray([p[c] for p in people], dtype=np.float64)
        enc['numeric'][c] = {'mean': float(v.mean()), 'sd': float(v.std())}
    for c in schema.INPUTS['categorical']:
        cnt = Counter(p[c] for p in people)
        enc['categorical'][c] = sorted(k for k, n in cnt.items() if n >= schema.CATEGORY_MIN_COUNT)
    names = list(schema.INPUTS['numeric'])
    for c in schema.INPUTS['categorical']:
        names += [f'{c}={k}' for k in enc['categorical'][c]] + [f'{c}=__OTHER__']
    enc['feature_names'] = names
    return enc


def encode(people, enc):
    names = enc['feature_names']; col = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(people), len(names)), dtype=np.float64)
    for i, p in enumerate(people):
        for c in schema.INPUTS['numeric']:
            st = enc['numeric'][c]; X[i, col[c]] = (p[c] - st['mean']) / st['sd']
        for c in schema.INPUTS['categorical']:
            key = f'{c}={p[c]}'
            X[i, col[key] if key in col else col[f'{c}=__OTHER__']] = 1.0
    return X.astype(np.float32)


def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, 'xb') as f:
        np.savez(f, **arrays)
    path.chmod(0o600)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha_file(path)}


def run(lock_path, lock_sha, admission_path, occ_map_path, out_receipt):
    lock, lock_commit = require_lock(lock_path, lock_sha)
    adm = json.loads(Path(admission_path).read_text())
    zip_path = PRIV / 'raw' / Path(schema.PERSON_ZIP_URL).name
    if sha_file(zip_path) != adm['zip_sha256']:
        raise PermissionError('raw zip differs from the metadata admission')
    occ_map = json.loads(Path(occ_map_path).read_text())
    if hashlib.sha256(json.dumps(occ_map['mapping'], sort_keys=True).encode()).hexdigest() != lock['benchmark']['occupation_mapping_sha256']:
        raise PermissionError('occupation mapping differs from the locked mapping')
    if EXPORTS.exists():
        raise PermissionError('fresh export directory required')
    reasons = Counter(); eligible = []; keys = set(); dup = 0
    relshipp_gq_in_hu = 0
    with zipfile.ZipFile(zip_path) as z, z.open(schema.PERSON_CSV) as fb:
        reader = csv.DictReader(io.TextIOWrapper(fb, encoding='latin-1', newline=''))
        missing = [c for c in NEEDED if c not in reader.fieldnames]
        if missing:
            raise ValueError(f'columns missing: {missing}')
        n = 0
        for row in reader:
            n += 1
            key = (row['SERIALNO'], row['SPORDER'])
            if key in keys:
                dup += 1
            keys.add(key)
            r, p = eligibility(row, occ_map)
            if r:
                reasons[r] += 1; continue
            if p['relshipp'] in ('37', '38'):
                relshipp_gq_in_hu += 1
            eligible.append(p)
    if dup:
        raise ValueError(f'{dup} duplicate person keys: extraction aborted by the frozen rule')
    by_house = defaultdict(list)
    for p in eligible:
        by_house[p['serial']].append(p)
    role_houses = defaultdict(list)
    for s in by_house:
        role_houses[schema.role_of(s)].append(s)
    selected, receipt_roles = {}, {}
    for role in DEV_ROLES:
        houses = sorted(role_houses[role], key=schema.cap_order_key)
        cap = schema.CAPS[role]; total = 0; take = []
        for s in houses:
            if total >= cap:
                break
            take.append(s); total += len(by_house[s])
        selected[role] = take
        receipt_roles[role] = {'eligible_households': len(houses), 'eligible_persons': sum(len(by_house[s]) for s in houses),
                               'selected_households': len(take), 'selected_persons': total, 'cap': cap,
                               'cap_overshoot_persons': max(0, total - cap),
                               'capped_out_households_unopened': len(houses) - len(take),
                               'capped_out_persons_unopened': sum(len(by_house[s]) for s in houses[len(take):])}
    sealed = sorted(role_houses['SEALED_ASSESSMENT'])
    sealed_rec = {'households': len(sealed), 'persons': sum(len(by_house[s]) for s in sealed),
                  'household_key_set_sha256': hashlib.sha256('\n'.join(schema.household_key(s) for s in sealed).encode()).hexdigest(),
                  'features_written': False, 'labels_written': False, 'predictions_computed': False}
    people = {role: [p for s in selected[role] for p in sorted(by_house[s], key=lambda q: q['sporder'])] for role in DEV_ROLES}
    enc = fit_encoding(people['MODEL_FIT'])
    files = {}
    for role in DEV_ROLES:
        P = people[role]
        row_id = np.asarray([int_id(f"{schema.household_key(p['serial'])}|{p['sporder']}") for p in P], dtype=np.int64)
        unit = np.asarray([int_id(schema.household_key(p['serial'])) for p in P], dtype=np.int64)
        if len(np.unique(row_id)) != len(row_id):
            raise ValueError('person id collision')
        X = encode(P, enc)
        f = {'features': save(EXPORTS / role / 'features.npz', row_id=row_id, unit=unit, X=X, feature_names=np.asarray(enc['feature_names'])),
             'raw_inputs': save(EXPORTS / role / 'raw_inputs.npz', row_id=row_id, unit=unit,
                                **{c: np.asarray([str(p[c]) for p in P]) for c in schema.INPUTS['numeric'] + schema.INPUTS['categorical']})}
        truth_dir = EXPORTS / ('DEV_REPORT_TRUTH' if role == 'DEV_REPORT' else role)
        f['income'] = save(truth_dir / 'income.npz', row_id=row_id, unit=unit, income=np.asarray([p['income'] for p in P], dtype=np.int64))
        f['occupation'] = save(truth_dir / 'occupation.npz', row_id=row_id, unit=unit, occupation=np.asarray([p['occupation'] for p in P], dtype=np.int64))
        f['sex'] = save(truth_dir / 'sex.npz', row_id=row_id, unit=unit, sex=np.asarray([p['sex'] for p in P], dtype=np.int64))
        f['weights'] = save(truth_dir / 'weights.npz', row_id=row_id, unit=unit, pwgtp=np.asarray([p['pwgtp'] for p in P], dtype=np.int64))
        files[role] = f
    # Support under the frozen thresholds (development fitting/attacker roles only; DEV_REPORT checked at report time).
    support = {'MODEL_FIT': {'income_counts': np.bincount([p['income'] for p in people['MODEL_FIT']], minlength=2).tolist(),
                             'occupation_counts': np.bincount([p['occupation'] for p in people['MODEL_FIT']], minlength=5).tolist()}}
    for role in ('ATTACK_FIT', 'ATTACK_VALIDATION'):
        support[role] = {'sex_counts': np.bincount([p['sex'] for p in people[role]], minlength=2).tolist()}
    thr = schema.SUPPORT
    support['status'] = {
        'MODEL_FIT_task_classes': all(c >= thr['task_class_min_MODEL_FIT'] for c in support['MODEL_FIT']['income_counts'] + support['MODEL_FIT']['occupation_counts']),
        'attacker_roles_sex': all(c >= thr['sex_class_min_attacker_roles'] for r in ('ATTACK_FIT', 'ATTACK_VALIDATION') for c in support[r]['sex_counts']),
        'DEV_REPORT_subgroup_households': 'evaluated inside the report worker after the REPORTING_LOCK'}
    receipt = {'schema': 'ACSR1-extraction-v1', 'kind': 'AUTOMATED_ELIGIBILITY_EXTRACTION (reads eligibility and target-validity variables)',
               'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'science_lock_sha256': lock_sha,
               'science_lock_commit': lock_commit, 'raw_zip_sha256': adm['zip_sha256'], 'raw_person_records': n,
               'exclusion_counts_first_failing_reason': dict(reasons), 'eligible_persons': len(eligible), 'eligible_households': len(by_house),
               'household_resident_rows_with_GQ_relationship_code': relshipp_gq_in_hu,
               'roles': receipt_roles, 'sealed_assessment': sealed_rec, 'encoding': enc, 'input_width': len(enc['feature_names']),
               'files': files, 'support': support, 'dev_report_truth_gated': True, 'rows_printed': 0}
    with open(out_receipt, 'x') as fo:
        json.dump(receipt, fo, indent=2, sort_keys=True)
    return receipt


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    for a in ('--lock', '--lock-sha', '--admission', '--occupation-map', '--out'):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    r = run(a.lock, a.lock_sha, a.admission, a.occupation_map, a.out)
    print(json.dumps({'eligible_persons': r['eligible_persons'], 'roles': {k: (v['selected_persons'], v['selected_households']) for k, v in r['roles'].items()},
                      'sealed': (r['sealed_assessment']['persons'], r['sealed_assessment']['households']), 'width': r['input_width'],
                      'support_status': r['support']['status']}))
