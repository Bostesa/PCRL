"""Independent check 2 (TASK MAPPING): rebuild OCCP -> 5 broad groups from the official documents.

Sources (read only): the official 2018 ACS PUMS Occupation Code List workbook (sheet 'ACS') and the 2019 PUMS data
dictionary CSV (VAL,OCCP rows). No person record is read. Imports nothing from acs_replication.

Independent rule: the workbook preamble lists the five civilian broad groups with code ranges ('*  0010 - 3550  ...').
Each 4-digit code row in the body is assigned (a) the most recent broad-group header row above it (a row whose code
cell is exactly one of the five ranges, or the military range 9800-9830) and, as a cross-check, (b) the broad range
that numerically contains it. Codes 9800-9830 (military), 9920 (never worked) and 'bbbb' (N/A) are excluded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import openpyxl

WT = Path(__file__).resolve().parents[1]
RES = WT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = WT / '.private' / 'acs_replication'
OUT = RES / 'independent' / 'check02_task_mapping.json'

GROUP_NAMES = ['MANAGEMENT_BUSINESS_SCIENCE_ARTS', 'SERVICE', 'SALES_OFFICE', 'NATRES_CONSTRUCTION_MAINTENANCE',
               'PRODUCTION_TRANSPORT_MATERIAL']
# words that identify the official broad-group titles (order of the preamble)
TITLE_KEYS = [('management', 'business', 'science', 'arts'), ('service',), ('sales', 'office'),
              ('natural resources', 'construction', 'maintenance'), ('production', 'transportation', 'material moving')]
# 2019 dictionary label prefixes per group (cross-check)
PREFIX = {0: {'MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'LGL', 'EDU', 'ENT', 'MED'},
          1: {'HLS', 'PRT', 'EAT', 'CLN', 'PRS'},
          2: {'SAL', 'OFF'},
          3: {'FFF', 'CON', 'EXT', 'RPR'},
          4: {'PRD', 'TRN'}}


def code_str(v):
    """Normalise a workbook code cell to a 4-digit string if it is a single code, else None."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and float(v).is_integer():
        s = '%04d' % int(v)
        return s if len(s) == 4 else None
    s = str(v).strip()
    return s if re.fullmatch(r'\d{4}', s) else None


def range_str(v):
    if v is None:
        return None
    m = re.fullmatch(r'\s*(\d{4})\s*-\s*(\d{4})\s*', str(v))
    return (m.group(1), m.group(2)) if m else None


def main():
    lock = json.loads((RES / 'SCIENCE_LOCK.json').read_text())
    adm = json.loads((PRIV / 'admission' / 'OCCUPATION_MAPPING.json').read_text())
    fails = []
    ev = {}

    wb = openpyxl.load_workbook(PRIV / 'docs' / '2018-ACS-PUMS-and-2018-SIPP-Public-Use-Occupation-Code-List.xlsx',
                                read_only=True, data_only=True)
    rows = list(wb['ACS'].iter_rows(values_only=True))

    # (1) preamble: five civilian broad groups with ranges
    pre = []
    for r in rows[:12]:
        for c in r:
            if isinstance(c, str):
                m = re.match(r'^\*\s*(\d{4})\s*-\s*(\d{4})\s+(.+)$', c.strip())
                if m:
                    pre.append((m.group(1), m.group(2), m.group(3).strip()))
    ranges = []
    for gi, keys in enumerate(TITLE_KEYS):
        hits = [p for p in pre if all(k in p[2].lower() for k in keys)]
        if len(hits) != 1:
            fails.append(f'preamble group {GROUP_NAMES[gi]}: {len(hits)} matches')
            ranges.append(None)
        else:
            ranges.append((hits[0][0], hits[0][1]))
    ev['preamble_ranges'] = {GROUP_NAMES[i]: ranges[i] for i in range(5)}

    # (2) body walk: inherit most recent broad-group header
    broad = {rg: gi for gi, rg in enumerate(ranges) if rg}
    broad[('9800', '9830')] = 'MILITARY'
    current = None
    by_header = {}
    duplicates = []
    started = False
    for r in rows:
        a = r[0] if r else None
        rg = range_str(a)
        if rg and rg in broad:
            current = broad[rg]
            started = True
            continue
        c = code_str(a)
        if c is None or not started:
            continue
        if c == '9920':
            by_header[c] = 'NEVER_WORKED'
            continue
        if c in by_header:
            duplicates.append(c)
        by_header[c] = current
    # numeric-range cross-check
    def by_range(c):
        x = int(c)
        for gi, rg in enumerate(ranges):
            if rg and int(rg[0]) <= x <= int(rg[1]):
                return gi
        if 9800 <= x <= 9830:
            return 'MILITARY'
        return None
    range_disagree = [c for c, g in by_header.items() if g != 'NEVER_WORKED' and by_range(c) != g]
    ev['workbook_codes'] = len(by_header)
    ev['workbook_duplicate_codes'] = duplicates
    ev['header_vs_numeric_range_disagreements'] = range_disagree
    if duplicates or range_disagree:
        fails.append('workbook walk inconsistent')

    # (3) dictionary OCCP codes
    dict_codes = []
    with open(PRIV / 'docs' / 'PUMS_Data_Dictionary_2019.csv', newline='') as f:
        for row in csv.reader(f):
            if len(row) >= 7 and row[0] == 'VAL' and row[1] == 'OCCP':
                if row[4] != row[5]:
                    fails.append(f'dictionary OCCP range row {row[4]}-{row[5]}')
                dict_codes.append((row[4], row[6]))
    ev['dictionary_occp_rows'] = len(dict_codes)
    mapping = {}
    excluded = {}
    missing_in_workbook = []
    prefix_problems = []
    for c, label in dict_codes:
        if c == 'bbbb':
            excluded[c] = 'blank (N/A)'
            continue
        g = by_header.get(c)
        if g is None:
            missing_in_workbook.append(c)
            continue
        if g == 'MILITARY':
            excluded[c] = 'military'
            if not label.startswith('MIL-'):
                prefix_problems.append(c)
            continue
        if g == 'NEVER_WORKED':
            excluded[c] = 'never worked'
            continue
        mapping[c] = g
        pfx = label.split('-', 1)[0]
        if pfx not in PREFIX[g]:
            prefix_problems.append(c)
    workbook_not_in_dictionary = sorted(set(by_header) - {c for c, _ in dict_codes})
    ev['missing_in_workbook'] = missing_in_workbook
    ev['workbook_codes_not_in_2019_dictionary'] = workbook_not_in_dictionary
    ev['label_prefix_inconsistencies'] = prefix_problems
    if missing_in_workbook or prefix_problems:
        fails.append('dictionary codes missing from workbook or label-prefix inconsistency')

    # (4) compare with the admitted mapping
    adm_map = adm['mapping']
    keys_only_mine = sorted(set(mapping) - set(adm_map))
    keys_only_adm = sorted(set(adm_map) - set(mapping))
    value_diff = sorted(c for c in set(mapping) & set(adm_map) if mapping[c] != adm_map[c])
    exact = mapping == adm_map
    counts_mine = {GROUP_NAMES[g]: sum(v == g for v in mapping.values()) for g in range(5)}
    my_sha = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    excl_expected = {'9800', '9810', '9825', '9830', '9920', 'bbbb'}
    ev['comparison'] = {
        'n_mapped_independent': len(mapping), 'n_mapped_admitted': len(adm_map),
        'exact_equality': exact, 'keys_only_independent': keys_only_mine, 'keys_only_admitted': keys_only_adm,
        'value_differences': value_diff,
        'group_order_admitted': adm['groups'], 'group_order_matches': adm['groups'] == GROUP_NAMES == lock['benchmark']['groups'],
        'per_group_counts_independent': counts_mine, 'per_group_counts_admitted': adm['per_group_code_counts'],
        'per_group_counts_SCIENCE_LOCK': lock['benchmark']['occupation_mapping_counts'],
        'mapping_sha256_independent': my_sha, 'mapping_sha256_admitted_file': adm['mapping_sha256'],
        'mapping_sha256_SCIENCE_LOCK': lock['benchmark']['occupation_mapping_sha256'],
        'excluded_independent': sorted(excluded), 'excluded_admitted': sorted(adm['excluded']),
        'excluded_SCIENCE_LOCK': sorted(lock['benchmark']['occupation_mapping_excluded']),
        'n_dictionary_codes_admitted': adm['n_dictionary_codes'], 'admitted_problems': adm['problems'],
    }
    c = ev['comparison']
    ok = (exact and c['group_order_matches'] and counts_mine == adm['per_group_code_counts'] == lock['benchmark']['occupation_mapping_counts']
          and my_sha == adm['mapping_sha256'] == lock['benchmark']['occupation_mapping_sha256']
          and set(excluded) == set(adm['excluded']) == set(lock['benchmark']['occupation_mapping_excluded']) == excl_expected
          and adm['n_dictionary_codes'] == len(dict_codes) and not adm['problems'])
    if not ok:
        fails.append('independent mapping differs from admitted mapping / lock')
    out = {'id': 2, 'name': 'TASK MAPPING', 'status': 'PASS' if not fails else 'FAIL', 'fail_reasons': fails, 'evidence': ev,
           'notes': 'codes are official dictionary/workbook codes (no person data); group index = position in SCIENCE_LOCK.benchmark.groups'}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({'status': out['status'], 'fails': fails, 'n_mapped': len(mapping), 'counts': counts_mine,
                      'sha_match': my_sha == lock['benchmark']['occupation_mapping_sha256'], 'exact': exact,
                      'excluded': sorted(excluded), 'workbook_not_in_dict': workbook_not_in_dictionary,
                      'ranges': ev['preamble_ranges']}, indent=1))


if __name__ == '__main__':
    main()
