"""EXR_OVERLAP_V1: household (unit) and person (row_id) overlap between the six admitted development roles.

PCRL frozen-finalist validation and confirmation-readiness study (development robustness evidence on already-used
public-use ACS 2019 1-Year PUMS person records for Florida). Exposure reviewer, label-free.

What is read:
- for each of MODEL_FIT, CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT, the admitted,
  hash-checked features export through ffv.common.load_role(role, ('features',)). Only its `row_id` and `unit` fields
  are used; the encoded feature matrix and the feature names are dropped unused. No SEX, task label, raw input,
  weight or prediction is requested (load_role never opens a SEX file unless 'sex' is requested, so no SEX read is
  logged by this unit);
- public records only for cross-checks: the replication's EXTRACTION_SUMMARY.json (role counts) and SCIENCE_LOCK.json
  (code hashes at lock), the frozen-hybrids FROZEN_BANK_MANIFEST.json (row_id array hashes), and the source code
  files acs_replication/schema.py and acs_replication/extract.py (hashed, and the role constants imported).

What is not checkable here: overlap with SEALED_ASSESSMENT. Its ids are not admitted (every loader refuses the role;
only a key-set hash was ever written). The output records what the role-assignment code implies and marks the
empirical check UNKNOWN.

Id convention (acs_replication/extract.py): unit = int(sha256('ACS|2019|1-Year|FL|<SERIALNO>').hexdigest()[:15], 16)
and row_id = the same of '<household key>|<SPORDER>'. Equal units therefore mean the same ACS housing-unit serial,
up to a 60-bit hash collision; the collision bound is recorded.

Output (write-once): ../HOUSEHOLD_OVERLAP_<unit id>.json (counts, set hashes and checks only; no identifier).

Revision for unit EXR_OVERLAP_V2 (after EXR_OVERLAP_V1, whose overlap counts were complete and whose only failing check
was this code-provenance comparison): acs_replication/extract.py legitimately differs from its SCIENCE_LOCK hash by
the documented engineering amendment A1 (LOCK_AMENDMENTS.md: the occupation-mapping lock-key path). The check now
hashes the file as of the science-lock commit (read with `git show`), confirms that hash equals the SCIENCE_LOCK value,
and confirms that the only differing line pair is the A1 path change, so household grouping and role assignment are
the locked code. schema.py (role_of, unit_interval, salts) must still equal the SCIENCE_LOCK hash exactly.
"""
from __future__ import annotations

import difflib
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ffv import common as C  # noqa: E402
from acs_replication import schema as S  # noqa: E402

UNIT = os.environ.get('FFV_UNIT_ID') or 'NO_UNIT'
OUT = Path(__file__).resolve().parents[1] / f'HOUSEHOLD_OVERLAP_{UNIT}.json'
HYB_BANK_ROLES = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'MODEL_SELECT')
A1_OLD = "lock['schema']['occupation_mapping_sha256']"
A1_NEW = "lock['benchmark']['occupation_mapping_sha256']"


def git_blob(rev, rel):
    p = subprocess.run(['git', '-C', str(ROOT), 'show', f'{rev}:{rel}'], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f'git show {rev}:{rel} failed')
    return p.stdout


def code_provenance(lock, lock_commit):
    """schema.py must equal the SCIENCE_LOCK hash; extract.py must equal it up to amendment A1 only."""
    out = {}
    for name in ('schema.py', 'extract.py'):
        rel = f'acs_replication/{name}'
        cur = (ROOT / rel).read_bytes()
        at_lock = git_blob(lock_commit, rel)
        rec = {'sha256': hashlib.sha256(cur).hexdigest(), 'science_lock_sha256': lock['code_sha256_at_lock'][name],
               'sha256_at_science_lock_commit': hashlib.sha256(at_lock).hexdigest(), 'science_lock_commit': lock_commit}
        rec['science_lock_commit_blob_equals_lock_hash'] = rec['sha256_at_science_lock_commit'] == rec['science_lock_sha256']
        rec['equals_science_lock'] = rec['sha256'] == rec['science_lock_sha256']
        diff = [l for l in difflib.unified_diff(at_lock.decode().splitlines(), cur.decode().splitlines(), lineterm='', n=0)
                if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        removed = [l[1:] for l in diff if l.startswith('-')]
        added = [l[1:] for l in diff if l.startswith('+')]
        rec['changed_lines_vs_lock'] = {'removed': len(removed), 'added': len(added)}
        if name == 'extract.py':
            rec['only_difference_is_amendment_A1'] = bool(
                len(removed) == 1 and len(added) == 1 and A1_OLD in removed[0] and A1_NEW in added[0]
                and removed[0].replace(A1_OLD, A1_NEW) == added[0])
            rec['amendment'] = ('LOCK_AMENDMENTS.md A1 (2026-10-16T00:11Z): occupation-mapping lock-key path corrected; '
                                'no change to eligibility, household grouping, role assignment, caps or ids')
            rec['ok'] = rec['science_lock_commit_blob_equals_lock_hash'] and (rec['equals_science_lock'] or rec['only_difference_is_amendment_A1'])
        else:
            rec['ok'] = rec['science_lock_commit_blob_equals_lock_hash'] and rec['equals_science_lock']
        out[name] = rec
    return out


def set_sha(a):
    """sha256 of the sorted distinct values as little-endian int64 bytes (a set fingerprint, not an id list)."""
    z = np.ascontiguousarray(np.unique(np.asarray(a, dtype=np.int64)).astype('<i8'))
    return hashlib.sha256(z.tobytes()).hexdigest()


def main():
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once; use a new unit id and file)')
    ext = json.loads((C.SB.SRC_RESULTS / 'EXTRACTION_SUMMARY.json').read_text())
    lock = json.loads((C.SB.SRC_RESULTS / 'SCIENCE_LOCK.json').read_text())
    hyb = json.loads((C.SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())

    units, rows, per_role = {}, {}, {}
    for role in C.ROLES6:
        d = C.load_role(role, ('features',))
        row_id = np.asarray(d['row_id'], dtype=np.int64)
        unit = np.asarray(d['unit'], dtype=np.int64)
        del d  # the feature matrix and feature names are not used
        if row_id.shape != unit.shape or row_id.ndim != 1:
            raise ValueError(f'{role}: row_id / unit shapes differ')
        u, inv = np.unique(unit, return_inverse=True)
        sizes = np.bincount(inv)
        units[role], rows[role] = u, np.unique(row_id)
        rec = ext['roles'][role]
        per_role[role] = {
            'persons': int(row_id.size),
            'distinct_row_ids': int(rows[role].size),
            'households_distinct_units': int(u.size),
            'max_persons_per_household': int(sizes.max()),
            'unit_ids_nonnegative_below_2_60': bool(unit.min() >= 0 and unit.max() < 2 ** 60),
            'extraction_summary_selected_persons': rec['selected_persons'],
            'extraction_summary_selected_households': rec['selected_households'],
            'matches_extraction_summary': bool(row_id.size == rec['selected_persons'] and u.size == rec['selected_households']),
            'unit_set_sha256': set_sha(u),
            'row_id_set_sha256': set_sha(row_id),
        }
        if role in HYB_BANK_ROLES:
            ref = {f'S{s}': hyb['per_seed'][f'S{s}']['row_id_sha256'][role] for s in C.SEEDS}
            mine = C.SB.arr_sha(row_id)
            per_role[role]['row_id_array_sha256_hybrids_convention'] = mine
            per_role[role]['equals_frozen_hybrids_bank_row_ids'] = {k: bool(v == mine) for k, v in ref.items()}

    pairs = {}
    for a, b in itertools.combinations(C.ROLES6, 2):
        pairs[f'{a} x {b}'] = {
            'shared_households': int(np.intersect1d(units[a], units[b], assume_unique=True).size),
            'shared_persons': int(np.intersect1d(rows[a], rows[b], assume_unique=True).size),
        }
    all_u = np.concatenate([units[r] for r in C.ROLES6])
    all_r = np.concatenate([rows[r] for r in C.ROLES6])
    union = {
        'households_summed_over_roles': int(all_u.size),
        'households_distinct_in_union': int(np.unique(all_u).size),
        'persons_summed_over_roles': int(all_r.size),
        'persons_distinct_in_union': int(np.unique(all_r).size),
    }
    union['households_disjoint'] = union['households_summed_over_roles'] == union['households_distinct_in_union']
    union['persons_disjoint'] = union['persons_summed_over_roles'] == union['persons_distinct_in_union']
    n_hh = union['households_distinct_in_union']

    dev_roles = [r for r, _ in S.DEV_SHARES]
    elig_dev_hh = sum(ext['roles'][r]['eligible_households'] for r in dev_roles)
    elig_dev_p = sum(ext['roles'][r]['eligible_persons'] for r in dev_roles)
    capped_hh = sum(ext['roles'][r]['capped_out_households_unopened'] for r in dev_roles)
    capped_p = sum(ext['roles'][r]['capped_out_persons_unopened'] for r in dev_roles)
    sealed = ext['sealed_assessment']
    partition_counts = {
        'eligible_households_total': ext['eligible_households'],
        'eligible_persons_total': ext['eligible_persons'],
        'development_eligible_households_summed_over_six_roles': elig_dev_hh,
        'development_eligible_persons_summed_over_six_roles': elig_dev_p,
        'sealed_households': sealed['households'],
        'sealed_persons': sealed['persons'],
        'households_add_up_exactly': bool(elig_dev_hh + sealed['households'] == ext['eligible_households']),
        'persons_add_up_exactly': bool(elig_dev_p + sealed['persons'] == ext['eligible_persons']),
        'capped_out_development_households_not_exported': capped_hh,
        'capped_out_development_persons_not_exported': capped_p,
        'note': ('count-level consistency with a household partition (each eligible household counted in exactly one role); '
                 'consistent with, not a proof of, disjointness from the sealed set'),
    }

    code_files = code_provenance(lock, ext['science_lock_commit'])

    sealed_block = {
        'empirical_overlap_check': 'UNKNOWN (sealed ids are not admitted and were not opened; only a key-set hash exists)',
        'published_household_key_set_sha256': sealed['household_key_set_sha256'],
        'implied_by_role_assignment_code': (
            'acs_replication/schema.py role_of(serialno) draws one unit-interval value u per SERIALNO from '
            'sha256(ROLE_SALT + household_key) and returns SEALED_ASSESSMENT iff u < SEALED_SHARE, otherwise exactly one '
            'development role; extract.py groups eligible persons by SERIALNO (by_house) and assigns each household once '
            '(role_houses[role_of(s)]), so by construction no household is both sealed and in a development role, and every '
            'eligible member of a household shares its role. Capped-out development households keep their development '
            'role draw but are not exported (cap order by CAP_SALT hash).'),
        'role_salt': S.ROLE_SALT, 'cap_salt': S.CAP_SALT, 'sealed_share': S.SEALED_SHARE,
        'development_shares': dict(S.DEV_SHARES),
        'scope_caveat': ('disjointness is by ACS housing-unit serial within the FL 2019 1-Year file; it is not person-level '
                         'independence (same survey, same PUMAs and population) and says nothing about people in other files'),
    }

    checks = {
        'all_15_pairs_share_no_household': all(v['shared_households'] == 0 for v in pairs.values()),
        'all_15_pairs_share_no_person': all(v['shared_persons'] == 0 for v in pairs.values()),
        'every_role_matches_extraction_summary_counts': all(v['matches_extraction_summary'] for v in per_role.values()),
        'row_ids_unique_within_every_role': all(v['persons'] == v['distinct_row_ids'] for v in per_role.values()),
        'hybrids_bank_row_order_equal_for_4_roles': all(all(v['equals_frozen_hybrids_bank_row_ids'].values())
                                                        for r, v in per_role.items() if r in HYB_BANK_ROLES),
        'partition_counts_add_up': partition_counts['households_add_up_exactly'] and partition_counts['persons_add_up_exactly'],
        'role_assignment_code_is_the_locked_code_up_to_amendment_A1': all(v['ok'] for v in code_files.values()),
    }
    out = {
        'schema': 'FFV1-household-overlap-v1',
        'unit_id': UNIT,
        'previous_attempt': ('EXR_OVERLAP_V1 (failed only its code-provenance check, which compared extract.py to the '
                             'SCIENCE_LOCK hash without allowing the documented amendment A1; its overlap counts are '
                             'retained in HOUSEHOLD_OVERLAP_EXR_OVERLAP_V1.json)') if UNIT != 'EXR_OVERLAP_V1' else None,
        'utc': C.now(),
        'campaign': C.CAMPAIGN,
        'reads': ("ffv.common.load_role(role, ('features',)) for the six admitted roles; row_id and unit only; "
                  'public EXTRACTION_SUMMARY.json, SCIENCE_LOCK.json and FROZEN_BANK_MANIFEST.json; acs_replication code'),
        'labels_read': 'none (no SEX, task label, raw input, weight or prediction requested)',
        'per_role': per_role,
        'pairwise': pairs,
        'union': union,
        'id_collision_bound': {'unit_bits': 60, 'households_in_union': n_hh,
                               'p_any_collision_upper_bound': n_hh * (n_hh - 1) / 2 / 2 ** 60,
                               'note': 'birthday bound for two different serials sharing a 60-bit unit id'},
        'extraction_partition_counts': partition_counts,
        'sealed_assessment': sealed_block,
        'role_assignment_code': code_files,
        'checks': checks,
        'status': 'PASS' if all(checks.values()) else 'CHECK_FAILED',
    }
    C.write_json_x(OUT, out)
    print(json.dumps({'status': out['status'], 'checks': checks, 'union': union,
                      'households': {r: v['households_distinct_units'] for r, v in per_role.items()}}, indent=1))
    return 0 if out['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
