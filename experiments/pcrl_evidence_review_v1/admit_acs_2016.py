"""Outcome-free admission preparation for the California ACS 2016 one-year person file.

This script prepares a *candidate* confirmation year. It deliberately cannot score
anything: it imports no fitted object, no preprocessor, no anchor and no scorer, and
it never reads a label on the final partition.

What it does:
  1. records provenance and checksums for the public zip, CSV and both dictionaries;
  2. compares the 2016 and 2017 dictionary entries for every feature, target and
     sensitive variable, verbatim block against verbatim block;
  3. validates the 2016 frame against the same code ranges the 2017 admission used,
     reporting violations instead of raising, so an inadmissible year is documented;
  4. builds a prospective household-hash partition under a new declared salt;
  5. reports protected-class support in the fitting and validation portions only.

For the final partition it records exactly two things: the row count and the
household count. No label, outcome distribution or protected-class summary of the
final partition is computed, printed or stored.

Run:
  python -m experiments.pcrl_evidence_review_v1.admit_acs_2016 \
      --data data/acs_2016_admission --out results/pcrl_evidence_review_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

YEAR = 2016
STATE = 6

# A *new* salt: reusing the 2017 salt would make the two years' partitions
# correlated through the identifier hash, and 2016 identifiers are not
# year-prefixed, so collisions with 2017 SERIALNOs are possible by construction.
PARTITION_SALT = 'PCRL-evidence-review-2016-admission-v1'
POOLS = ('fitting', 'validation', 'final_evaluation')
FRACTIONS = (0.50, 0.20, 0.30)
# The 2017 study split its fitting and validation halves into disjoint attacker and
# task pools so that utility probes and attackers never share households. That
# convention is retained inside the 50/20 blocks and documented in DATA_2016_ADMISSION.md.
SUBPOOLS = {'fitting': (('attacker_fit', 0.5), ('task_fit', 0.5)),
            'validation': (('attacker_validation', 0.5), ('task_validation', 0.5))}

NUMERIC = ('AGEP', 'WKHP')
CATEGORICAL = ('SCHL', 'MAR', 'RELP', 'CIT', 'DIS', 'DEAR', 'DEYE', 'DREM')
FEATURES = NUMERIC + CATEGORICAL
SOURCE_COLUMNS = ('PINCP', 'ESR', 'PUBCOV')
RESERVED_COLUMNS = ('MIG', 'JWMNP')
SENSITIVE = ('SEX', 'RAC1P')
KEYS = ('SERIALNO', 'SPORDER', 'PWGTP', 'ST', 'RT', 'ADJINC')
REQUIRED = tuple(dict.fromkeys(FEATURES + SOURCE_COLUMNS + RESERVED_COLUMNS + SENSITIVE + KEYS))

CODES = {'SCHL': range(1, 25), 'MAR': range(1, 6), 'RELP': range(18), 'CIT': range(1, 6),
         'DIS': (1, 2), 'DEAR': (1, 2), 'DEYE': (1, 2), 'DREM': (1, 2),
         'ESR': range(1, 7), 'PUBCOV': (1, 2), 'MIG': (1, 2, 3), 'SEX': (1, 2),
         'RAC1P': range(1, 10)}
RANGES = {'AGEP': (0, 99), 'WKHP': (1, 99), 'PINCP': (-19998, 4209995), 'JWMNP': (1, 200)}
DICTIONARY_VARIABLES = tuple(dict.fromkeys(FEATURES + SOURCE_COLUMNS + RESERVED_COLUMNS + SENSITIVE))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


# ------------------------------------------------------------------ dictionaries
# The two dictionaries use different layouts: the 2016 file writes
# "NAME<TAB><TAB>width", the 2017 file writes "NAME    Character   2". Comparing
# raw text would report a difference for every variable, so the comparison is made
# on the extracted (code -> label) maps and the variable width, which is what the
# feature and target semantics actually depend on.
HEADER_2016 = re.compile(r'^([A-Z][A-Z0-9]*)\t+(\d+)\t*\s*$', re.M)
HEADER_2017 = re.compile(r'^([A-Z][A-Z0-9]*)\s+(?:Numeric|Character)\s+(\d+)\s*$', re.M)
CODE_LINE = re.compile(r'^\s*([0-9]+(?:\.\.[0-9]+)?|b+)\s*\.?\s*\.(.*)$')


def parse_dictionary(text):
    """Return {variable: {'width': int, 'description': str, 'codes': {code: label}}}."""
    header = HEADER_2016 if len(HEADER_2016.findall(text)) > len(HEADER_2017.findall(text)) else HEADER_2017
    matches = list(header.finditer(text))
    out = {}
    for i, m in enumerate(matches):
        name, width = m.group(1), int(m.group(2))
        body = text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        lines = [line for line in body.splitlines() if line.strip()]
        description = ' '.join(lines[0].split()) if lines else ''
        codes = {}
        for line in lines[1:]:
            cm = CODE_LINE.match(line.replace('\t', ' '))
            if cm:
                codes[cm.group(1).lstrip('0') or '0'] = ' '.join(cm.group(2).split())
        if name not in out:
            out[name] = {'width': width, 'description': description, 'codes': codes}
    return out


def compare_dictionaries(text_2016, text_2017):
    d16, d17 = parse_dictionary(text_2016), parse_dictionary(text_2017)
    out = {}
    for name in DICTIONARY_VARIABLES:
        a, b = d16.get(name), d17.get(name)
        entry = {'present_2016': a is not None, 'present_2017': b is not None}
        if a is None or b is None:
            out[name] = entry
            continue
        entry.update({
            'width_2016': a['width'], 'width_2017': b['width'], 'identical_width': a['width'] == b['width'],
            'description_2016': a['description'], 'description_2017': b['description'],
            'identical_description': a['description'] == b['description'],
            'n_codes_2016': len(a['codes']), 'n_codes_2017': len(b['codes']),
            'identical_code_sets': sorted(a['codes']) == sorted(b['codes']),
            'identical_code_labels': a['codes'] == b['codes'],
        })
        if not entry['identical_code_labels']:
            keys = sorted(set(a['codes']) | set(b['codes']), key=lambda k: (len(k), k))
            entry['code_differences'] = [
                {'code': k, '2016': a['codes'].get(k), '2017': b['codes'].get(k)}
                for k in keys if a['codes'].get(k) != b['codes'].get(k)][:8]
        out[name] = entry
    return out


# ------------------------------------------------------------------- validation
def validate_frame(frame):
    """Report, never raise. An inadmissible year must be documented, not crash."""
    report = {'violations': [], 'columns': {}}
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        report['violations'].append({'check': 'missing_columns', 'detail': missing})
        return report
    if not frame.RT.eq('P').all():
        report['violations'].append({'check': 'record_type', 'detail': 'non-P rows present'})
    if not frame.ST.eq(STATE).all():
        report['violations'].append({'check': 'state', 'detail': 'non-California rows present'})

    serial = frame.SERIALNO.astype('string')
    report['serialno_format'] = {
        'all_nine_digits': bool(serial.str.fullmatch(r'\d{9}').fillna(False).all()),
        'has_2017_style_year_prefix': bool(serial.str.startswith(str(YEAR)).any()),
        'example': str(serial.iloc[0]),
        'distinct_lengths': sorted(set(serial.str.len().dropna().astype(int).tolist())),
    }
    sp = pd.to_numeric(frame.SPORDER, errors='coerce').to_numpy(float)
    if not (np.isfinite(sp) & (sp >= 1) & (sp <= 20) & (sp == np.floor(sp))).all():
        report['violations'].append({'check': 'sporder_range', 'detail': 'out of 1..20'})
    w = frame.PWGTP.to_numpy(float)
    if not (np.isfinite(w) & (w >= 0) & (w <= 9999) & (w == np.floor(w))).all():
        report['violations'].append({'check': 'pwgtp_range', 'detail': 'out of 0..9999'})

    for c in (*FEATURES, *SOURCE_COLUMNS, *RESERVED_COLUMNS, *SENSITIVE):
        v = pd.to_numeric(frame[c], errors='coerce').to_numpy(float)
        present = np.isfinite(v)
        if c in CODES:
            valid = np.isin(v, list(CODES[c]))
        else:
            lo, hi = RANGES[c]
            valid = (v >= lo) & (v <= hi) & (v == np.floor(v))
        bad = int(((~valid) & present).sum())
        report['columns'][c] = {
            'missing': int((~present).sum()), 'present': int(present.sum()),
            'out_of_dictionary': bad,
            'observed_codes': (sorted(np.unique(v[present]).astype(int).tolist())
                               if c in CODES else None),
            'observed_range': (None if c in CODES else
                               [float(np.nanmin(v[present])), float(np.nanmax(v[present]))] if present.any() else None),
        }
        if bad:
            report['violations'].append({'check': 'out_of_dictionary', 'detail': f'{c}: {bad} rows'})
    return report


def eligible_cohort(raw):
    raw = raw.copy()
    raw['_raw_row'] = np.arange(len(raw), dtype=np.int64)
    raw['SPORDER'] = pd.to_numeric(raw.SPORDER).astype(int).astype(str)
    eligible = raw.AGEP.between(19, 34) & raw.PWGTP.gt(0)
    f = raw.loc[eligible].copy()
    dup = f.duplicated(['SERIALNO', 'SPORDER'], keep=False)
    conflicts = 0
    for _, g in f.loc[dup].groupby(['SERIALNO', 'SPORDER']):
        if len(g.drop(columns='_raw_row').drop_duplicates()) != 1:
            conflicts += 1
    before = len(f)
    f = f.drop_duplicates(['SERIALNO', 'SPORDER']).reset_index(drop=True)
    return f, {
        'raw_rows': int(len(raw)),
        'eligible_rows_before_dedup': int(before),
        'duplicate_rows_removed': int(before - len(f)),
        'conflicting_duplicate_keys': int(conflicts),
        'eligible_rows': int(len(f)),
        'eligible_published_groups': int(f.SERIALNO.nunique()),
        'group_quarters_person_rows': int(f.RELP.isin([16, 17]).sum()),
        'eligibility_rule': '19 <= AGEP <= 34 and PWGTP > 0; whole SERIALNO groups retained',
    }


def partition_households(frame):
    """Deterministic, label-blind household partition ordered by a salted digest."""
    groups = sorted(frame.SERIALNO.unique(),
                    key=lambda s: (hashlib.sha256(f'{PARTITION_SALT}|{YEAR}|{STATE:02d}|{s}'.encode()).hexdigest(), s))
    n = len(groups)
    bounds = np.rint(np.cumsum((0.0,) + FRACTIONS) * n).astype(int)
    blocks = {name: groups[bounds[i]:bounds[i + 1]] for i, name in enumerate(POOLS)}
    assignment = {}
    for name, members in blocks.items():
        if name in SUBPOOLS:
            sub_bounds = np.rint(np.cumsum((0.0,) + tuple(f for _, f in SUBPOOLS[name])) * len(members)).astype(int)
            for i, (sub, _) in enumerate(SUBPOOLS[name]):
                assignment[f'{name}/{sub}'] = members[sub_bounds[i]:sub_bounds[i + 1]]
        assignment[name] = members
    index = {}
    for name, members in assignment.items():
        index[name] = np.flatnonzero(frame.SERIALNO.isin(set(members)))
    top = np.concatenate([index[p] for p in POOLS])
    if len(np.unique(top)) != len(frame) or len(top) != len(frame):
        raise ValueError('household partition is not a partition of the eligible frame')
    return index, assignment


def fixed_labels(frame):
    """The frozen task definitions, carried over unchanged from the 2017 admission."""
    out = {}
    spec = [('income_binary', 'PINCP', None), ('civilian_at_work', 'ESR', range(1, 7)),
            ('public_coverage', 'PUBCOV', (1, 2)), ('same_residence', 'MIG', (1, 2, 3)),
            ('commute_over20', 'JWMNP', None), ('SEX', 'SEX', (1, 2)), ('RAC1P', 'RAC1P', range(1, 10))]
    for target, column, valid_codes in spec:
        v = pd.to_numeric(frame[column], errors='coerce').to_numpy(float)
        if target == 'income_binary':
            valid = np.isfinite(v) & (v >= -19998) & (v <= 4209995)
            y = v > 50000
        elif target == 'commute_over20':
            valid = np.isfinite(v) & (v >= 1) & (v <= 200) & (v == np.floor(v))
            y = v > 20
        else:
            valid = np.isin(v, list(valid_codes))
            y = (v - 1) if target in SENSITIVE else (v == 1)
        r = np.full(len(frame), -1, dtype=np.int64)
        r[valid] = np.asarray(y[valid], dtype=np.int64)
        out[target] = r
    return out


def planning_support(frame, labels, index):
    """Class support, computed ONLY on fitting and validation pools.

    The final partition contributes its row and household counts and nothing else:
    no label, outcome distribution or protected-class summary is read from it.
    """
    out = {}
    for name, rows in index.items():
        if name.startswith('final_evaluation'):
            out[name] = {'rows': int(len(rows)),
                         'households': int(frame.SERIALNO.iloc[rows].nunique()),
                         'labels_read': False,
                         'note': 'counts only; no label or protected-class summary computed'}
            continue
        entry = {'rows': int(len(rows)),
                 'households': int(frame.SERIALNO.iloc[rows].nunique()),
                 'labels_read': True, 'targets': {}}
        for target, y in labels.items():
            v = y[rows]
            valid = v >= 0
            entry['targets'][target] = {
                'valid': int(valid.sum()), 'invalid_or_inapplicable': int((~valid).sum()),
                'class_counts': np.bincount(v[valid], minlength=9 if target == 'RAC1P' else 2).tolist(),
            }
        out[name] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    official = data / 'official'
    csv = data / 'extracted' / 'ss16pca.csv'
    provenance = {
        'year': YEAR, 'state_fips': STATE,
        'source_zip_url': 'https://www2.census.gov/programs-surveys/acs/data/pums/2016/1-Year/csv_pca.zip',
        'dictionary_2016_url': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict16.txt',
        'dictionary_2017_url': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.txt',
        'sha256': {
            'csv_pca_2016.zip': sha256_file(official / 'csv_pca_2016.zip'),
            'PUMSDataDict16.txt': sha256_file(official / 'PUMSDataDict16.txt'),
            'PUMS_Data_Dictionary_2017.txt': sha256_file(official / 'PUMS_Data_Dictionary_2017.txt'),
            'ss16pca.csv': sha256_file(csv),
        },
    }
    print('provenance hashed', flush=True)

    text16 = (official / 'PUMSDataDict16.txt').read_text(encoding='latin-1')
    text17 = (official / 'PUMS_Data_Dictionary_2017.txt').read_text(encoding='latin-1')
    dictionary = compare_dictionaries(text16, text17)
    print('dictionary compared', flush=True)

    raw = pd.read_csv(csv, usecols=list(REQUIRED), dtype={'SERIALNO': str, 'SPORDER': str, 'RT': str})
    validation = validate_frame(raw)
    adjinc = sorted(pd.to_numeric(raw.ADJINC, errors='coerce').dropna().unique().tolist())
    frame, cohort = eligible_cohort(raw)
    del raw
    print(f"cohort: {cohort['eligible_rows']} rows, {cohort['eligible_published_groups']} groups", flush=True)

    index, assignment = partition_households(frame)
    labels = fixed_labels(frame)
    support = planning_support(frame, labels, index)

    serial = frame.SERIALNO.to_numpy(dtype=str)
    disjoint = {}
    sets = {p: set(serial[index[p]]) for p in POOLS}
    for i, a in enumerate(POOLS):
        for b in POOLS[i + 1:]:
            disjoint[f'{a}|{b}'] = len(sets[a] & sets[b])

    manifest = {
        'study': 'pcrl_evidence_review_v1 / ACS 2016 admission preparation',
        'created_utc': pd.Timestamp.utcnow().isoformat(),
        'provenance': provenance,
        'adjinc_values_observed': adjinc,
        'dictionary_comparison_2016_vs_2017': dictionary,
        'schema_validation': validation,
        'cohort': cohort,
        'partition_rule': {
            'salt': PARTITION_SALT,
            'digest': 'SHA256(UTF8("salt|year|state|SERIALNO")); order by (digest, SERIALNO)',
            'pools': POOLS, 'fractions': list(FRACTIONS),
            'subpools': {k: [list(x) for x in v] for k, v in SUBPOOLS.items()},
            'label_blind': True,
        },
        'partition_counts': {name: {'rows': int(len(rows)),
                                    'households': int(frame.SERIALNO.iloc[rows].nunique())}
                             for name, rows in index.items()},
        'household_overlap_between_top_level_pools': disjoint,
        'person_key_uniqueness': {
            'rows': int(len(frame)),
            'distinct_year_qualified_keys': int(pd.Series(
                [f'{YEAR}|{s}|{o}' for s, o in zip(frame.SERIALNO, frame.SPORDER)]).nunique()),
        },
        'planning_support': support,
        'array_hashes': {
            'serialno': array_hash(serial),
            'sporder': array_hash(frame.SPORDER.to_numpy(dtype=str)),
            'pwgtp': array_hash(frame.PWGTP.to_numpy(float)),
            'raw_row': array_hash(frame._raw_row.to_numpy()),
            **{f'partition/{name}': array_hash(rows) for name, rows in index.items()},
        },
        'final_partition_label_access': 'none; only row and household counts were computed',
    }
    (out / 'ACS_2016_ADMISSION_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False, default=str) + '\n')
    print('manifest written', flush=True)
    print(json.dumps({'violations': validation['violations'],
                      'serialno_format': validation['serialno_format'],
                      'adjinc': adjinc,
                      'partition_counts': manifest['partition_counts'],
                      'overlap': disjoint,
                      'dictionary_identical': {k: v.get('identical_normalised_text')
                                               for k, v in dictionary.items()},
                      'dictionary_same_codes': {k: v.get('identical_code_sets')
                                                for k, v in dictionary.items()}}, indent=2))


if __name__ == '__main__':
    main()
