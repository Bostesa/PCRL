"""Frozen ACS benchmark definition (custom shared-worker, two-task benchmark; NOT unmodified ACSIncome).

Population: one ACS 1-Year PUMS state-year, admitted by the frozen order in provenance.py.
Everything here is fixed before any outcome value is inspected; the extractor (extract.py) applies it
mechanically after the science lock.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path

SURVEY = 'ACS'
YEAR = '2019'
HORIZON = '1-Year'
STATE_NAME, STATE_POSTAL, STATE_FIPS = 'Florida', 'FL', '12'
PERSON_ZIP_URL = 'https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_pfl.zip'
PERSON_CSV = 'psam_p12.csv'
DOCS = {
    'data_dictionary_csv': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2019.csv',
    'data_dictionary_txt': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2019.txt',
    'code_lists': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/code_lists/ACSPUMS2019CodeLists.xls',
    'readme': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/ACS2019_PUMS_README.pdf',
    'accuracy': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2019AccuracyPUMS.pdf',
    'design_factors': 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2019_PUMS_1yr_Design_Factors.csv',
    'occupation_code_list': 'https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2018-ACS-PUMS-and-2018-SIPP-Public-Use-Occupation-Code-List.xlsx',
    'occupation_crosswalk': 'https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2018-occupation-code-list-and-crosswalk.xlsx',
}
ADJINC_2019 = 1010145  # dictionary: '2019 factor (1.010145)', 6 implied decimals

# ---------------------------------------------------------------- cohort predicate (all must hold)
PREDICATE = {
    'record_type': 'person file rows only (RT == "P")',
    'state': f'ST == "{STATE_FIPS}"',
    'household_residents': 'SERIALNO matches ^2019HU[0-9]{7}$ (housing-unit person); every 2019GQ (group quarters) row excluded',
    'age': 'AGEP integer, 25 <= AGEP <= 64',
    'employment': 'ESR in {"1","2"} (civilian employed, at work / with a job but not at work)',
    'hours': 'WKHP non-blank integer in 1..99 (WKHP > 0)',
    'occupation': 'OCCP non-blank, present in the 2019 dictionary, and mapped to one of the five civilian groups (military 9800-9830, 9920 and blank excluded)',
    'income': 'PINCP non-blank (signed integer) and ADJINC == 1010145',
    'sex': 'SEX in {"1","2"}',
    'inputs': 'AGEP, WKHP numeric as above; COW, SCHL, MAR, POBP non-blank strings (any other blank -> row ineligible and counted)',
    'duplicates': '(SERIALNO, SPORDER) must be unique; any duplicate person key aborts extraction',
}
SEX_CODE = {'1': 0, '2': 1}  # sex01: 0 = male (SEX 1), 1 = female (SEX 2); SEX is never an input
INCOME_RULE = 'income01 = 1 iff PINCP * ADJINC / 1,000,000 > 50000 (2019 dollars; NOT the Adult 1994 threshold)'
GROUPS = ('MANAGEMENT_BUSINESS_SCIENCE_ARTS', 'SERVICE', 'SALES_OFFICE', 'NATRES_CONSTRUCTION_MAINTENANCE', 'PRODUCTION_TRANSPORT_MATERIAL')
GROUP_HEADERS = {  # exact header strings of the official 2018 ACS PUMS Occupation Code List (sheet "ACS", rows 3-7)
    'Management, Business, Science, and Arts Occupations': 0, 'Service Occupations': 1, 'Sales and Office Occupations': 2,
    'Natural Resources, Construction, and Maintenance Occupations': 3, 'Production, Transportation, and Material Moving Occupations': 4}
MILITARY_RANGE_LABEL = 'Military Specific Occupations'

# ---------------------------------------------------------------- inference inputs (common whitelist)
INPUTS = {'numeric': ('AGEP', 'WKHP'), 'categorical': ('COW', 'SCHL', 'MAR', 'POBP')}
FORBIDDEN = ('SEX', 'RAC1P', 'RAC2P', 'RAC3P', 'RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACPI', 'RACSOR', 'RACWHT', 'HISP',
             'OCCP', 'SOCP', 'INDP', 'NAICSP', 'PINCP', 'PERNP', 'WAGP', 'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP', 'POVPIP',
             'ESR', 'RELP', 'RELSHIPP', 'SERIALNO', 'SPORDER', 'PUMA', 'POWPUMA', 'POWSP', 'ST', 'PWGTP', 'ADJINC')
CATEGORY_MIN_COUNT = 10  # categorical level kept iff it occurs >= 10 times among MODEL_FIT people; all else -> per-variable OTHER column
NUMERIC_SCALING = 'z-score with MODEL_FIT mean and population sd (ddof=0), float64 then float32'

# ---------------------------------------------------------------- household roles
ROLE_SALT = 'PCRL-ACS-protected-tree-replication-v1|household-roles|FL-2019|2026-10-15|'
CAP_SALT = 'PCRL-ACS-protected-tree-replication-v1|cap-order|FL-2019|2026-10-15|'
SEALED_SHARE = 0.20
DEV_SHARES = (('MODEL_FIT', 0.45), ('CALIBRATION', 0.05), ('MODEL_SELECT', 0.15), ('ATTACK_FIT', 0.15),
              ('ATTACK_VALIDATION', 0.10), ('DEV_REPORT', 0.10))
CAPS = {'MODEL_FIT': 20000, 'CALIBRATION': 3000, 'MODEL_SELECT': 8000, 'ATTACK_FIT': 8000, 'ATTACK_VALIDATION': 3000, 'DEV_REPORT': 8000}
SUPPORT = {'task_class_min_MODEL_FIT': 100, 'sex_class_min_attacker_roles': 100, 'subgroup_households_min_per_sex_DEV_REPORT': 200}


def household_key(serialno):
    return f'{SURVEY}|{YEAR}|{HORIZON}|{STATE_POSTAL}|{serialno}'


def unit_interval(salt, key):
    return int(hashlib.sha256((salt + key).encode()).hexdigest()[:16], 16) / 2 ** 64


def role_of(serialno):
    """Deterministic household role from the published salt; every member of a household shares it."""
    u = unit_interval(ROLE_SALT, household_key(serialno))
    if u < SEALED_SHARE:
        return 'SEALED_ASSESSMENT'
    v = (u - SEALED_SHARE) / (1 - SEALED_SHARE)
    acc = 0.0
    for name, share in DEV_SHARES:
        acc += share
        if v < acc:
            return name
    return DEV_SHARES[-1][0]


def cap_order_key(serialno):
    return hashlib.sha256((CAP_SALT + household_key(serialno)).encode()).hexdigest()


# ---------------------------------------------------------------- official occupation mapping
def build_occupation_mapping(code_list_xlsx, dictionary_csv):
    """Map every 2019 dictionary OCCP code to one of the five official broad groups.

    Source of the groups: the official '2018 American Community Survey PUMS Occupation Code List'
    (sheet 'ACS'), whose header rows list the five civilian broad groups with their code ranges and whose
    body lists every code under its group. The mapping is taken from that document, code by code:
    each individual 4-digit code row inherits the most recent broad-group header row above it.
    Cross-check: the 2019 dictionary label prefixes (MGR, BUS, ..., MIL) must be consistent with the groups.
    """
    import openpyxl
    wb = openpyxl.load_workbook(code_list_xlsx, read_only=True, data_only=True)
    ws = wb['ACS']
    broad = {}
    current = None
    header_rx = re.compile(r'^\s*([0-9]{4})\s*-\s*([0-9]{4})\s*$')
    for row in ws.iter_rows(values_only=True):
        cells = [c for c in row if c is not None]
        if not cells:
            continue
        first = str(cells[0]).strip()
        desc = str(cells[-1]).strip().rstrip(':').replace('  ', ' ')
        if header_rx.match(first) and desc in GROUP_HEADERS:
            current = GROUP_HEADERS[desc]; continue
        if header_rx.match(first) and desc == MILITARY_RANGE_LABEL:
            current = 'MILITARY'; continue
        if re.fullmatch(r'[0-9]{4}', first) or (isinstance(cells[0], int) and 0 < cells[0] < 10000):
            code = f'{int(first):04d}'
            if current is not None:
                broad.setdefault(code, current)
    rows = list(csv.reader(open(dictionary_csv, encoding='latin-1')))
    dict_codes = {r[4]: r[6] for r in rows if len(r) > 6 and r[0] == 'VAL' and r[1] == 'OCCP'}
    prefix_group = {**{p: 0 for p in ('MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'LGL', 'EDU', 'ENT', 'MED')},
                    **{p: 1 for p in ('HLS', 'PRT', 'EAT', 'CLN', 'PRS')}, **{p: 2 for p in ('SAL', 'OFF')},
                    **{p: 3 for p in ('FFF', 'CON', 'EXT', 'RPR')}, **{p: 4 for p in ('PRD', 'TRN')}, 'MIL': 'MILITARY'}
    mapping, excluded, problems = {}, {}, []
    for code, label in sorted(dict_codes.items()):
        if code == 'bbbb':
            excluded[code] = 'blank (N/A)'; continue
        if code == '9920':
            excluded[code] = 'unemployed, no work experience in last 5 years or never worked'; continue
        g = broad.get(code)
        prefix = label.split('-')[0]
        expected = prefix_group.get(prefix)
        if g is None:
            problems.append({'code': code, 'label': label, 'issue': 'absent from official code list body'}); continue
        if expected is not None and g != expected:
            problems.append({'code': code, 'label': label, 'issue': f'group {g} disagrees with dictionary prefix {prefix}'})
        if g == 'MILITARY':
            excluded[code] = 'military specific occupation'; continue
        mapping[code] = int(g)
    out = {'schema': 'ACSR1-occupation-mapping-v1', 'groups': list(GROUPS), 'mapping': mapping, 'excluded': excluded,
           'problems': problems, 'n_dictionary_codes': len(dict_codes), 'n_mapped': len(mapping),
           'per_group_code_counts': {GROUPS[i]: sum(v == i for v in mapping.values()) for i in range(5)},
           'source_rule': build_occupation_mapping.__doc__.strip()}
    out['mapping_sha256'] = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return out


def frozen_definition():
    return {'survey': SURVEY, 'year': YEAR, 'horizon': HORIZON, 'state': [STATE_NAME, STATE_POSTAL, STATE_FIPS],
            'person_zip_url': PERSON_ZIP_URL, 'person_csv': PERSON_CSV, 'docs': DOCS, 'adjinc': ADJINC_2019,
            'predicate': PREDICATE, 'sex_code': SEX_CODE, 'income_rule': INCOME_RULE, 'groups': list(GROUPS),
            'inputs': INPUTS, 'forbidden_inputs': list(FORBIDDEN), 'category_min_count': CATEGORY_MIN_COUNT,
            'numeric_scaling': NUMERIC_SCALING, 'role_salt': ROLE_SALT, 'cap_salt': CAP_SALT, 'sealed_share': SEALED_SHARE,
            'development_shares': dict(DEV_SHARES), 'caps': CAPS, 'support_thresholds': SUPPORT,
            'cap_rule': 'within each development role, households in ascending cap_order_key; include whole households while the role total is below its cap; the household that reaches or crosses the cap is included and the overshoot recorded; every later household stays unopened (no labels or predictions computed)',
            'group_quarters': 'excluded by the household-resident predicate (2019GQ serials)',
            'weights': 'PWGTP retained privately for secondary weighted summaries only; never a model or attacker feature'}
