"""Independent check 4: household roles re-derived from identifiers only.

Reads ONLY the SERIALNO column of the raw person CSV (identifier column; no other raw column is read for any
household, sealed or not). Re-implements the registered role / cap-order / integer-id rules from the frozen
rule text (SCIENCE_LOCK benchmark.role_salt, sealed_share, development_shares in the frozen schema order; the
frozen schema.py hash is checked against SCIENCE_LOCK.code_sha256_at_lock) without importing acs_replication.

Exported files are read for their identifier arrays (row_id, unit) only. DEV_REPORT_TRUTH identifier arrays
are re-read for alignment (verification re-read of an already-opened role; no label value is used).
Outputs aggregates only.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parent.parent
RES = WT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = WT / '.private' / 'acs_replication'
OUT = RES / 'independent' / 'check04_household_roles.json'

lock = json.loads((RES / 'SCIENCE_LOCK.json').read_text())
summary = json.loads((RES / 'EXTRACTION_SUMMARY.json').read_text())
receipt = json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())
popdec = json.loads((RES / 'POPULATION_DECISION.json').read_text())
bm = lock['benchmark']

ROLE_SALT = bm['role_salt']
CAP_SALT = bm['cap_salt']
SEALED_SHARE = Fraction(str(bm['sealed_share']))
# Frozen order of the development shares (schema.DEV_SHARES; the lock JSON is key-sorted, so the order is taken from
# the frozen schema text, whose file hash is checked against the lock below).
DEV_ORDER = ['MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT']
SHARES = {k: Fraction(str(v)) for k, v in bm['development_shares'].items()}
CAPS = bm['caps']
SURVEY, YEAR, HORIZON, POSTAL = bm['survey'], bm['year'], bm['horizon'], bm['state'][1]
HU_RX = re.compile(r'^2019HU[0-9]{7}$')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def hkey(serial):
    return f'{SURVEY}|{YEAR}|{HORIZON}|{POSTAL}|{serial}'


def int_id(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def role_exact(serial):
    """Exact rational evaluation of the registered rule (u = first 64 bits of sha256(salt+key) / 2^64)."""
    u = Fraction(int(hashlib.sha256((ROLE_SALT + hkey(serial)).encode()).hexdigest()[:16], 16), 2 ** 64)
    if u < SEALED_SHARE:
        return 'SEALED_ASSESSMENT'
    v = (u - SEALED_SHARE) / (1 - SEALED_SHARE)
    acc = Fraction(0)
    for name in DEV_ORDER:
        acc += SHARES[name]
        if v < acc:
            return name
    return DEV_ORDER[-1]


def role_float(serial):
    """Floating-point evaluation (as an implementation would do it), used only to count boundary sensitivity."""
    u = int(hashlib.sha256((ROLE_SALT + hkey(serial)).encode()).hexdigest()[:16], 16) / 2 ** 64
    if u < float(bm['sealed_share']):
        return 'SEALED_ASSESSMENT'
    v = (u - float(bm['sealed_share'])) / (1 - float(bm['sealed_share']))
    acc = 0.0
    for name in DEV_ORDER:
        acc += float(bm['development_shares'][name])
        if v < acc:
            return name
    return DEV_ORDER[-1]


def cap_key(serial):
    return hashlib.sha256((CAP_SALT + hkey(serial)).encode()).hexdigest()


ev = {}
problems = []

# ---------------------------------------------------------------- frozen code pins
import subprocess


def git(*a):
    return subprocess.run(['git', '-C', str(WT), *a], capture_output=True, text=True, check=True).stdout


LOCK_COMMIT = '4d50dea13b9b19bdf9f5d77ee66656572d9e4692'
code_pins = {}
for fn in ('schema.py', 'extract.py'):
    cur = sha256_file(WT / 'acs_replication' / fn)
    at_lock = hashlib.sha256(subprocess.run(['git', '-C', str(WT), 'show', f'{LOCK_COMMIT}:acs_replication/{fn}'], capture_output=True, check=True).stdout).hexdigest()
    rec = {'current_sha256': cur, 'lock_sha256': lock['code_sha256_at_lock'][fn], 'blob_at_lock_commit_sha256': at_lock,
           'current_matches_lock': cur == lock['code_sha256_at_lock'][fn], 'lock_commit_blob_matches_lock': at_lock == lock['code_sha256_at_lock'][fn]}
    if not rec['lock_commit_blob_matches_lock']:
        problems.append(f'{fn} at the lock commit differs from SCIENCE_LOCK code hash')
    if not rec['current_matches_lock']:
        changed = [l for l in git('diff', LOCK_COMMIT, 'HEAD', '--', f'acs_replication/{fn}').splitlines()
                   if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        rec['changed_lines_since_lock'] = len(changed)
        only_a1 = (len(changed) == 2 and all('occupation_mapping_sha256' in l for l in changed)
                   and any("lock['benchmark']['occupation_mapping_sha256']" in l for l in changed if l.startswith('+')))
        rec['change_is_documented_amendment_A1_only'] = only_a1
        rec['role_cap_id_logic_touched'] = not only_a1
        if not only_a1:
            problems.append(f'{fn} changed after the lock beyond the documented amendment A1')
    code_pins[fn] = rec
schema_txt = (WT / 'acs_replication' / 'schema.py').read_text()
order_in_schema = re.search(r"DEV_SHARES = \((.*?)\)\n", schema_txt, re.S).group(1)
order_found = re.findall(r"\('([A-Z_]+)', ([0-9.]+)\)", order_in_schema)
code_pins['dev_share_order_in_frozen_schema'] = [o[0] for o in order_found]
code_pins['dev_share_order_used'] = DEV_ORDER
if [o[0] for o in order_found] != DEV_ORDER or any(Fraction(o[1]) != SHARES[o[0]] for o in order_found):
    problems.append('development share order/values differ from the frozen schema')
for frag in (f"ROLE_SALT = '{ROLE_SALT}'", f"CAP_SALT = '{CAP_SALT}'", 'SEALED_SHARE = 0.20', "hexdigest()[:16], 16) / 2 ** 64", "hexdigest()[:15], 16)"):
    if frag not in schema_txt + (WT / 'acs_replication' / 'extract.py').read_text():
        problems.append(f'rule fragment not found in frozen code: {frag}')
ev['frozen_rule_pins'] = code_pins

# ---------------------------------------------------------------- raw identifiers (SERIALNO column only)
zip_path = PRIV / 'raw' / 'csv_pfl.zip'
zip_sha = sha256_file(zip_path)
ev['raw_zip_sha256'] = zip_sha
ev['raw_zip_matches_population_decision'] = zip_sha == popdec['official_source']['zip_sha256']
if not ev['raw_zip_matches_population_decision']:
    problems.append('raw zip hash differs from POPULATION_DECISION')
with zipfile.ZipFile(zip_path) as z, z.open(bm['person_csv']) as fb:
    ser = pd.read_csv(io.TextIOWrapper(fb, encoding='latin-1', newline=''), usecols=['SERIALNO'], dtype=str)['SERIALNO']
raw_rows = int(len(ser))
rows_per_serial = Counter(ser.tolist())
del ser
serials = sorted(rows_per_serial)
hu = [s for s in serials if HU_RX.match(s)]
gq = [s for s in serials if not HU_RX.match(s)]
ev['raw_identifier_scan'] = {
    'columns_read': ['SERIALNO'], 'raw_person_rows': raw_rows, 'distinct_serials': len(serials),
    'distinct_housing_unit_serials': len(hu), 'distinct_non_HU_serials': len(gq),
    'non_HU_person_rows': int(sum(rows_per_serial[s] for s in gq)),
    'summary_raw_person_records': summary['raw_person_records'],
    'summary_group_quarters_rows_excluded': summary['exclusion_counts_first_failing_reason'].get('group_quarters_or_nonhousing_serial'),
    'lock_metadata_distinct_HU_serials': lock['population']['metadata_admission']['distinct_housing_unit_serials'],
    'lock_metadata_distinct_GQ_serials': lock['population']['metadata_admission']['distinct_group_quarters_serials'],
}
s_ = ev['raw_identifier_scan']
if raw_rows != summary['raw_person_records']:
    problems.append('raw person rows differ from summary')
if s_['non_HU_person_rows'] != s_['summary_group_quarters_rows_excluded']:
    problems.append('non-HU rows differ from the GQ exclusion count')
if len(hu) != s_['lock_metadata_distinct_HU_serials'] or len(gq) != s_['lock_metadata_distinct_GQ_serials']:
    problems.append('distinct HU/GQ serial counts differ from the metadata admission')

# roles, cap keys, unit ids for every serial (identifiers only)
role_of = {s: role_exact(s) for s in hu}
float_disagree = sum(role_float(s) != role_of[s] for s in hu)
unit_of = {}
collide = 0
for s in serials:
    u = int_id(hkey(s))
    if u in unit_of:
        collide += 1
    unit_of[u] = s
ev['rule_implementation'] = {
    'role_rule': 'u = int(sha256(role_salt + key)[:16], 16) / 2^64 (exact rational); SEALED iff u < 0.2; else v = (u - 0.2)/0.8 and the first cumulative development share (frozen order) exceeding v',
    'household_key': "'ACS|2019|1-Year|FL|' + SERIALNO",
    'unit_id_rule': 'int(sha256(key)[:15], 16)', 'row_id_rule': "int(sha256(key + '|' + SPORDER)[:15], 16) (SPORDER recovered by enumeration, never read from raw)",
    'cap_order_rule': 'ascending sha256(cap_salt + key) hex',
    'exact_vs_float_role_disagreements_over_all_HU_serials': float_disagree,
    'unit_id_collisions_over_all_serials': collide,
    'HU_serials_per_role_all_households': dict(Counter(role_of.values())),
}
if collide:
    problems.append('integer unit id collision among raw serials')

# ---------------------------------------------------------------- exported identifiers
FILES = ['features', 'raw_inputs', 'income', 'occupation', 'sex', 'weights']
roles_out = {}
unit_sets, row_sets = {}, {}
for role in DEV_ORDER:
    ids = {}
    for f in FILES:
        d = 'DEV_REPORT_TRUTH' if (role == 'DEV_REPORT' and f in ('income', 'occupation', 'sex', 'weights')) else role
        with np.load(PRIV / 'exports' / d / f'{f}.npz', allow_pickle=False) as z:
            ids[f] = (z['row_id'].astype(np.int64), z['unit'].astype(np.int64))
    r0, u0 = ids['features']
    aligned = all(np.array_equal(ids[f][0], r0) and np.array_equal(ids[f][1], u0) for f in FILES)
    units = u0.tolist(); rows = r0.tolist()
    unmapped = sum(1 for u in units if u not in unit_of)
    mapped = [unit_of.get(u) for u in units]
    non_hu = sum(1 for s in mapped if s is not None and not HU_RX.match(s))
    roles_found = Counter(role_of.get(s, 'NON_HU_OR_UNMAPPED') for s in mapped)
    wrong_role = sum(n for r, n in roles_found.items() if r != role)
    sealed_hits = roles_found.get('SEALED_ASSESSMENT', 0)
    # household contiguity and cap order
    order, seen, contiguous = [], set(), True
    prev = None
    for u in units:
        if u != prev:
            if u in seen:
                contiguous = False
            seen.add(u); order.append(u); prev = u
    keys = [cap_key(unit_of[u]) for u in order if u in unit_of]
    cap_sorted = all(keys[i] < keys[i + 1] for i in range(len(keys) - 1))
    members = Counter(units)
    # row ids: enumerate SPORDER candidates per household (two textual formats), match exported row ids
    fmt_hits = Counter(); unmatched = 0; sp_by_house = defaultdict(list)
    cand_cache = {}
    for u, rid in zip(units, rows):
        s = unit_of.get(u)
        if s is None:
            unmatched += 1; continue
        if s not in cand_cache:
            kmax = max(rows_per_serial[s], 1) + 5
            c = {}
            for k in range(1, kmax + 1):
                fmts = (('two_digit_ge10', str(k)),) if k >= 10 else (('plain_1digit', str(k)), ('zero_padded_2digit', f'{k:02d}'))
                for fmt, txt in fmts:
                    c.setdefault(int_id(f'{hkey(s)}|{txt}'), []).append((fmt, k))
            cand_cache[s] = c
        hit = cand_cache[s].get(rid)
        if not hit:
            unmatched += 1; continue
        fmt_hits[hit[0][0]] += 1; sp_by_house[u].append(hit[0][1])
    sp_over_raw = sum(1 for u, sps in sp_by_house.items() if max(sps) > rows_per_serial[unit_of[u]])
    members_over_raw = sum(1 for u, n in members.items() if u in unit_of and n > rows_per_serial[unit_of[u]])
    dup_sp = sum(1 for sps in sp_by_house.values() if len(sps) != len(set(sps)))
    n_persons, n_house = len(units), len(order)
    rs = summary['roles'][role]
    cap = CAPS[role]
    last_size = members[order[-1]] if order else 0
    cap_rule = {
        'persons_before_last_household_below_cap': (n_persons - last_size) < cap,
        'reached_cap_if_capped_out': (n_persons >= cap) if rs['capped_out_households_unopened'] > 0 else True,
        'overshoot_recomputed': max(0, n_persons - cap), 'overshoot_summary': rs['cap_overshoot_persons'],
    }
    # identifier-only bounds on eligibility-dependent counts
    role_hu = [s for s in hu if role_of[s] == role]
    cutoff = keys[-1] if keys else ''
    after = [s for s in role_hu if cap_key(s) > cutoff]
    exported_serials = {unit_of[u] for u in order if u in unit_of}
    gaps_before_cutoff = sum(1 for s in role_hu if cap_key(s) <= cutoff and s not in exported_serials)
    bounds = {
        'HU_households_in_role_all': len(role_hu), 'eligible_households_summary': rs['eligible_households'],
        'HU_households_in_role_ge_eligible': len(role_hu) >= rs['eligible_households'],
        'raw_person_rows_in_role_HU': int(sum(rows_per_serial[s] for s in role_hu)), 'eligible_persons_summary': rs['eligible_persons'],
        'raw_rows_ge_eligible_persons': int(sum(rows_per_serial[s] for s in role_hu)) >= rs['eligible_persons'],
        'HU_households_after_cutoff': len(after), 'capped_out_households_summary': rs['capped_out_households_unopened'],
        'HU_after_cutoff_ge_capped_out': len(after) >= rs['capped_out_households_unopened'],
        'raw_rows_after_cutoff': int(sum(rows_per_serial[s] for s in after)), 'capped_out_persons_summary': rs['capped_out_persons_unopened'],
        'raw_rows_after_cutoff_ge_capped_out_persons': int(sum(rows_per_serial[s] for s in after)) >= rs['capped_out_persons_unopened'],
        'non_exported_HU_households_before_cutoff_(must_all_be_ineligible;_not_verifiable)': gaps_before_cutoff,
    }
    arith = {
        'eligible_minus_selected_eq_capped_out_households': rs['eligible_households'] - rs['selected_households'] == rs['capped_out_households_unopened'],
        'eligible_minus_selected_eq_capped_out_persons': rs['eligible_persons'] - rs['selected_persons'] == rs['capped_out_persons_unopened'],
        'uncapped_role_selected_eq_eligible': (rs['selected_households'] == rs['eligible_households']) if rs['capped_out_households_unopened'] == 0 else None,
    }
    rec = {
        'exported_persons': n_persons, 'exported_households': n_house,
        'summary_selected_persons': rs['selected_persons'], 'summary_selected_households': rs['selected_households'],
        'identifier_arrays_aligned_across_6_files': aligned, 'row_ids_unique': len(set(rows)) == len(rows),
        'units_unmapped_to_raw_serial': unmapped, 'units_mapped_to_non_HU_serial': non_hu,
        'rows_whose_household_maps_to_other_role': wrong_role, 'rows_mapping_to_SEALED_ASSESSMENT': sealed_hits,
        'households_contiguous': contiguous, 'households_in_ascending_cap_order': cap_sorted,
        'row_ids_matched_to_key_and_sporder': n_persons - unmatched, 'row_id_unmatched': unmatched,
        'sporder_textual_format_hits': dict(fmt_hits), 'duplicate_sporder_within_household': dup_sp,
        'households_with_more_exported_members_than_raw_rows': members_over_raw,
        'households_with_sporder_above_raw_row_count': sp_over_raw,
        'max_exported_members_per_household': max(members.values()) if members else 0,
        'cap_rule': cap_rule, 'identifier_only_bounds': bounds, 'summary_arithmetic': arith,
    }
    roles_out[role] = rec
    unit_sets[role] = set(units); row_sets[role] = set(rows)
    bad = []
    if not aligned: bad.append('identifier arrays not aligned')
    if not rec['row_ids_unique']: bad.append('row id duplicates')
    if unmapped or non_hu or wrong_role or sealed_hits: bad.append('role mapping failure')
    if not contiguous or not cap_sorted: bad.append('household order/cap order failure')
    if unmatched or dup_sp or members_over_raw: bad.append('row id / membership failure')
    if n_persons != rs['selected_persons'] or n_house != rs['selected_households']: bad.append('counts differ from summary')
    if not cap_rule['persons_before_last_household_below_cap'] or not cap_rule['reached_cap_if_capped_out'] or cap_rule['overshoot_recomputed'] != cap_rule['overshoot_summary']:
        bad.append('cap rule failure')
    if not all(v for k, v in bounds.items() if isinstance(v, bool)): bad.append('identifier-only bound violated')
    if not all(v for v in arith.values() if v is not None): bad.append('summary arithmetic failure')
    if bad:
        problems.append(f'{role}: ' + '; '.join(bad))
ev['roles'] = roles_out

pair_overlap = {}
for i, a in enumerate(DEV_ORDER):
    for b in DEV_ORDER[i + 1:]:
        nu, nr = len(unit_sets[a] & unit_sets[b]), len(row_sets[a] & row_sets[b])
        if nu or nr:
            pair_overlap[f'{a}&{b}'] = {'units': nu, 'rows': nr}
ev['cross_role_overlap'] = pair_overlap or 'none (all 15 role pairs disjoint in units and person rows)'
if pair_overlap:
    problems.append('a unit or row appears in two roles')

# ---------------------------------------------------------------- totals and sealed set
tot_h = sum(summary['roles'][r]['eligible_households'] for r in DEV_ORDER) + summary['sealed_assessment']['households']
tot_p = sum(summary['roles'][r]['eligible_persons'] for r in DEV_ORDER) + summary['sealed_assessment']['persons']
ev['summary_totals'] = {'dev_plus_sealed_eligible_households': tot_h, 'eligible_households': summary['eligible_households'],
                        'dev_plus_sealed_eligible_persons': tot_p, 'eligible_persons': summary['eligible_persons'],
                        'consistent': tot_h == summary['eligible_households'] and tot_p == summary['eligible_persons'],
                        'summary_equals_private_receipt_roles_and_sealed': summary['roles'] == receipt['roles'] and summary['sealed_assessment'] == receipt['sealed_assessment']}
if not ev['summary_totals']['consistent'] or not ev['summary_totals']['summary_equals_private_receipt_roles_and_sealed']:
    problems.append('summary totals inconsistent or differ from private receipt')

sealed_all = sorted(s for s in hu if role_of[s] == 'SEALED_ASSESSMENT')
h_all = hashlib.sha256('\n'.join(hkey(s) for s in sealed_all).encode()).hexdigest()
elig_share = {r: summary['roles'][r]['eligible_households'] / summary['eligible_households'] for r in DEV_ORDER}
elig_share['SEALED_ASSESSMENT'] = summary['sealed_assessment']['households'] / summary['eligible_households']
expected_share = {r: float(SHARES[r] * (1 - SEALED_SHARE)) for r in DEV_ORDER}
expected_share['SEALED_ASSESSMENT'] = float(SEALED_SHARE)
ev['sealed_assessment'] = {
    'registered_serialization': "sha256('\\n'.join(sorted household keys)) over ELIGIBLE sealed households (extract.py: role_houses built from eligible persons)",
    'summary_household_key_set_sha256': summary['sealed_assessment']['household_key_set_sha256'],
    'summary_sealed_eligible_households': summary['sealed_assessment']['households'],
    'summary_sealed_eligible_persons': summary['sealed_assessment']['persons'],
    'all_HU_sealed_households_from_identifiers': len(sealed_all),
    'all_HU_sealed_person_rows_from_identifiers': int(sum(rows_per_serial[s] for s in sealed_all)),
    'all_HU_sealed_key_set_sha256': h_all,
    'all_HU_version_matches_summary_hash': h_all == summary['sealed_assessment']['household_key_set_sha256'],
    'eligible_sealed_le_all_HU_sealed': summary['sealed_assessment']['households'] <= len(sealed_all),
    'exact_hash_reproducible_from_identifiers_only': False,
    'reason': 'the registered set contains only households with >= 1 ELIGIBLE member; eligibility needs AGEP/ESR (and the other predicate columns), which may not be read for sealed households under the access rule; the all-housing-unit version is reported for transparency and is expected to differ',
    'eligible_household_share_by_role_summary': {k: round(v, 5) for k, v in elig_share.items()},
    'registered_expected_share': {k: round(v, 5) for k, v in expected_share.items()},
}
if not ev['sealed_assessment']['eligible_sealed_le_all_HU_sealed']:
    problems.append('sealed eligible households exceed all HU sealed households')

out = {
    'id': 4, 'name': 'HOUSEHOLD ROLES',
    'status': 'PASS' if not problems else 'FAIL',
    'evidence': ev,
    'problems': problems,
    'scope': ('identifier-only: raw SERIALNO column only (no other raw column read for any household); exported row_id/unit arrays only; '
              'DEV_REPORT_TRUTH identifier arrays re-read for alignment (verification re-read of an already-opened role, no label used). '
              'Eligibility-dependent counts (eligible/capped-out per capped role, the sealed eligible set and its hash) cannot be recomputed '
              'without eligibility columns; they are checked by identifier-only bounds and summary arithmetic instead.'),
    'sealed_hash_status': 'NOT_VERIFIABLE_UNDER_ACCESS_RULE',
    'DEV_REPORT_truth_reread': 'identifier arrays (row_id, unit) only',
    'sealed_assessment_touched': False,
}
OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
print(json.dumps({'status': out['status'], 'problems': problems,
                  'roles': {r: (v['exported_persons'], v['exported_households'], v['rows_whose_household_maps_to_other_role'], v['row_id_unmatched']) for r, v in roles_out.items()},
                  'float_disagree': float_disagree, 'sealed_all_HU': len(sealed_all), 'sealed_hash_all_HU_match': ev['sealed_assessment']['all_HU_version_matches_summary_hash']}, indent=1))
