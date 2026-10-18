"""Pre-outcome record (reader investigator; requested by the statistical reviewer, RULES.md section 11): the two recipient screens
for H3's FROZEN income view against each required comparator, written BEFORE any new candidate of the correction-transfer
campaign is scored with readers.

LABEL-FREE. Inputs: the source campaign's frozen development reader-grid outputs (admitted copy, read-only:
.private/pcrl_occupation_headroom_overnight_v1/runs/GRID/<bank>/S<s>/{GRID.json, BOOT.npz}; SCORE / VALIDATION AUCs, winners and
household-bootstrap draws) and the frozen release arrays only to hash released wires (cts.devreaders.load_member -> the admitted
copies). No label, feature or person-level array is printed or written. Public-use ACS 2019 1-Year PUMS Florida person records,
MODEL_SELECT development role (adaptively used); readers estimate the recorded SEX column from released outputs only.

Screens (imported, not re-defined): cts.rules.per_cell_gate (old per-cell gate, every cell <= +0.010) and
cts.rules.aggregate_paired_screen (AGG1, registered gross phi_prior; exact aliases pass by construction).
A cell difference = candidate (H3 income wire) SCORE AUC minus the comparator's; household-bootstrap SE of the paired difference
(the grid's B = 400 draws, same weights for both releases in a cell).

    <repository>/.venv/bin/python results/pcrl_correction_transfer_stability_v1/roles/reader_investigator/code/pre_outcome_h3_income.py
"""
from __future__ import annotations
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from cts import common as C          # noqa: E402
from cts import devreaders as DR     # noqa: E402  (committed; frozen-release loading through the admitted copies)
from cts import rules as RU          # noqa: E402

OUT = C.RESULTS / 'screen' / 'PRE_OUTCOME_H3_INCOME_VIEW.json'
GRID = C.OHN_RUNS / 'GRID'
BANKS = ('REF_BANK', 'WITNESS_BANK', 'INC_BANK', 'PROT_P1', 'PROT_P2', 'PROT_P2A', 'PROT_P2L', 'PROT_P2AL', 'PROT_P3', 'PROT_P3A',
         'RR_LC_WIT', 'RR_LC_PROT_P1', 'RR_LC_PROT_P2', 'RR_LC_PROT_P2B50', 'RR_LC_PROT_P2A')
SEEDS = (0, 1)
CELLS = tuple(f'S{s}R{r}k{k}' for s in SEEDS for r in (1, 2) for k in (0, 1))
VIEWS = ('income', 'occupation')
H3_REP = 'SUCC'                       # SUCC = H3 income + JOINT occupation: its income wire IS H3's frozen income wire


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_grid():
    cells, boot, specs, files, problems = {}, {}, {}, {}, []
    for b in BANKS:
        for s in SEEDS:
            d = GRID / b / f'S{s}'
            if not (d / 'GRID.json').is_file():
                continue
            files[f'{b}/S{s}/GRID.json'] = sha(d / 'GRID.json'); files[f'{b}/S{s}/BOOT.npz'] = sha(d / 'BOOT.npz')
            g = json.loads((d / 'GRID.json').read_text())
            for name, m in g['spec']['members'].items():
                specs.setdefault(name, m)
            with np.load(d / 'BOOT.npz') as z:
                bt = {k: z[k].copy() for k in z.files}
            for ck, cres in g['cells'].items():
                for name, ent in cres.items():
                    for v, e in ent.items():
                        key = (ck, name, v)
                        if key in cells:
                            if abs(cells[key]['score_auc'] - e['score_auc']) > 1e-12:
                                problems.append(f'{key}: {b} differs')
                            continue
                        cells[key] = e; boot[key] = bt[f'{ck}__{name}__{v}']
    return cells, boot, specs, files, problems


def comparison(cells, boot, cand, comp, view):
    d, se = {}, {}
    for c in CELLS:
        a, b = cells[(c, cand, view)], cells[(c, comp, view)]
        d[c] = float(a['score_auc'] - b['score_auc'])
        se[c] = float(np.std(boot[(c, cand, view)] - boot[(c, comp, view)], ddof=1))
    return d, se


def wires_of(specs, names):
    out = {}
    for n in names:
        out[n] = {}
        for s in SEEDS:
            try:
                _, prov = DR.load_member(specs[n], s)
            except FileNotFoundError:
                continue
            out[n][s] = prov['wire_sha256']
    return out


def alias(w, a, b, view):
    return all(s in w.get(a, {}) and s in w.get(b, {}) and w[a][s][view] == w[b][s][view] for s in SEEDS)


def screens(cells, boot, w, cand, comp, view, phi):
    d, se = comparison(cells, boot, cand, comp, view)
    al = alias(w, cand, comp, view)
    old = RU.per_cell_gate(d)
    agg = RU.aggregate_paired_screen(d, se, phi, alias=al)
    return d, se, al, old, agg


def what_if(rec, recipient_pass, name):
    """The source nomination order (ohn.rules.evaluate_config) with the recipient-gate result supplied."""
    lab = rec['label']
    if lab in ('NOT_ADMISSIBLE', 'NOT_CONSIDERED_ONE_SEED_ONLY') or lab.startswith('ALIAS_OF_'):
        return lab
    if rec.get('utility_point_failed'):
        return 'UTILITY_POINT_INELIGIBLE'
    if not recipient_pass:
        return 'RECIPIENT_POINT_FAILED' if name == 'OLD' else 'RECIPIENT_SCREEN_FAILED_AGG1'
    if (rec.get('outcomes') or {}).get('reserve') != 'RESERVE_MET':
        return 'RESERVE_NOT_MET'
    if rec.get('benefit_not_stable'):
        return 'BENEFIT_NOT_STABLE'
    if (rec.get('outcomes') or {}).get('matched_component_comparison') != 'PRESENT':
        return 'NO_MATCHED_COMPONENT_COMPARISON'
    return 'QUALIFYING'


def main():
    rules = RU.load_rules()
    phi = {v: RU.PHI_PRIOR[v] for v in VIEWS}
    cells, boot, specs, files, problems = load_grid()
    nom = json.loads((C.OHN_RESULTS / 'NOMINATION.json').read_text())
    two_seed = [r for r in nom['configurations'] if r.get('per_comparator')]
    w = wires_of(specs, sorted(set(RU.REQUIRED) | {r['config_id'] for r in two_seed}))
    # H3's frozen income wire = SUCC's income wire (checked against the anchor itself)
    from ohn import devrun as D
    h3_is_succ_income = all(C.SB.wire_sha(D.anchor_income('H3', s, 'MODEL_SELECT')) == w[H3_REP][s]['income'] for s in SEEDS)
    h3_record = {}
    for comp in RU.REQUIRED:
        d, se, al, old, agg = screens(cells, boot, w, H3_REP, comp, 'income', phi['income'])
        h3_record[comp] = {'exact_alias': al, 'cells_H3_minus_comparator': d, 'cell_se': se,
                           'old_gate': {'pass': old['pass'], 'cells_above_0.010': old.get('cells_above', []), 'max': old.get('max'),
                                        'mean': old.get('mean')},
                           'AGG1': {'pass': agg['pass'], 'label': agg['label'], 'U': agg.get('U'), 'dbar': agg.get('dbar'),
                                    'phi': agg.get('phi'), 'phi_prior': agg.get('phi_prior'), 'phi_emp': agg.get('phi_emp'),
                                    'se_total': agg.get('se_total'), 'per_seed': agg.get('per_seed'), 'flags': agg.get('flags')}}
    old_fail = [c for c, r in h3_record.items() if not r['old_gate']['pass']]
    agg_fail = [c for c, r in h3_record.items() if not r['AGG1']['pass']]

    def fmt(c):
        r = h3_record[c]['AGG1']
        return f"{c}, U = {r['U']:+.4f}"
    consequence = (
        f"Every H3-anchored release (any release whose income wire is bitwise H3's frozen income wire, e.g. every Track F candidate "
        f"that keeps H3's income) fails the configuration-level AGG1 screen through its income view alone "
        f"({'; '.join(fmt(c) for c in agg_fail)}), whatever its occupation output; it also fails the old per-cell gate through its "
        f"income view (against {', '.join(old_fail)}: "
        + '; '.join(f"{c} cells {', '.join(h3_record[c]['old_gate']['cells_above_0.010'])} up to {h3_record[c]['old_gate']['max']:+.4f}" for c in old_fail)
        + "). Neither screen can admit a candidate that keeps H3's frozen income. Known before any new outcome; computed label-free "
          "from frozen source grid outputs.")
    # source what-if (part A summary): the 18 two-seed source configurations, both screens over 6 comparators x 2 views
    wi = {}
    for r in two_seed:
        cid = r['config_id']
        per = {}
        for comp in RU.REQUIRED:
            for v in VIEWS:
                d, se, al, old, agg = screens(cells, boot, w, cid, comp, v, phi[v])
                per[f'{comp}|{v}'] = {'old': old['pass'], 'agg': agg['pass'], 'U': agg.get('U'), 'alias': al}
        old_failed = sorted(k for k, x in per.items() if not x['old'])
        agg_failed = sorted(k for k, x in per.items() if not x['agg'])
        wi[cid] = {'source_label': r['label'], 'old_failed': old_failed,
                   'old_reproduces_source': old_failed == sorted(r.get('recipient_point_failed') or []),
                   'old_reproduced_label': what_if(r, not old_failed, 'OLD'), 'agg1_failed': agg_failed,
                   'what_if_label_AGG1': what_if(r, not agg_failed, 'AGG1')}
    qual = [c for c, x in wi.items() if x['what_if_label_AGG1'] == 'QUALIFYING']
    # distinct occupation wires among the required comparators
    groups = {}
    for comp in RU.REQUIRED:
        groups.setdefault(tuple(w[comp][s]['occupation'] for s in SEEDS), []).append(comp)
    occ_groups = sorted(sorted(g) for g in groups.values())
    inc_groups = {}
    for comp in RU.REQUIRED:
        inc_groups.setdefault(tuple(w[comp][s]['income'] for s in SEEDS), []).append(comp)
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    rec = {
        'schema': 'CTS1-pre-outcome-h3-income-view-v1', 'utc': C.now(), 'labels_read': False, 'git_head': head,
        'purpose': 'pre-outcome record requested by the statistical reviewer (RULES.md section 11): both recipient screens for H3\'s '
                   'frozen income view against each required comparator, before any new candidate is scored with readers',
        'inputs': {'source_grid_files_sha256': files, 'source_nomination_sha256': sha(C.OHN_RESULTS / 'NOMINATION.json'),
                   'cts_rules_py_sha256': sha(ROOT / 'cts' / 'rules.py'), 'rules_json_sha256': rules['_sha256'],
                   'cts_devreaders_py_sha256': sha(ROOT / 'cts' / 'devreaders.py'), 'script_sha256': sha(Path(__file__)),
                   'merge_problems': problems,
                   'note': 'admitted read-only copies; identical (cell, release, view) entries from several banks come from the same cached readers'},
        'screens': {'old': 'cts.rules.per_cell_gate (every one of the 8 cells <= +0.010, inclusive)',
                    'AGG1': 'cts.rules.aggregate_paired_screen (U = dbar + z sqrt(phi v_samp) <= 0.010; phi = max(1, phi_prior, phi_emp); '
                            'EXACT_ALIAS passes by construction)',
                    'phi_prior_used': phi, 'z': RU.Z, 'allowance': RU.RECIPIENT_ALLOWANCE},
        'h3_income_wire': {'representative': H3_REP, 'h3_anchor_income_equals_SUCC_income_bitwise_both_seeds': h3_is_succ_income},
        'per_comparator': h3_record,
        'old_gate_fails_against': old_fail, 'AGG1_fails_against': agg_fail,
        'consequence': consequence,
        'source_what_if_part_A': {
            'configurations': wi, 'n_two_seed_configurations': len(wi),
            'old_gate_reproduces_source_recipient_failures_all': all(x['old_reproduces_source'] for x in wi.values()),
            'all_fail_AGG1': all(x['agg1_failed'] for x in wi.values()),
            'what_if_outcome': ('NO_QUALIFYING_CONFIGURATION (unchanged under AGG1)' if not qual else
                                f'descriptive what-if would qualify {qual}'),
            'note': 'descriptive what-if only: the source nomination order with the recipient gate replaced by AGG1, every other '
                    'component (utility point, reserve, benefit stability, matched component) as recorded in the source; the source '
                    'label NO_QUALIFYING_CONFIGURATION is never rewritten; a screen pass is permission for further development, not '
                    'evidence of protection'},
        'required_comparator_wires': {
            'occupation_wire_groups': occ_groups, 'n_distinct_occupation_wires': len(occ_groups),
            'income_wire_groups': sorted(sorted(g) for g in inc_groups.values()), 'n_distinct_income_wires': len(inc_groups),
            'note': f'{len(occ_groups)} distinct occupation wires among the 6 required comparators: the occupation-view screen has only '
                    f'{len(occ_groups)} distinct comparator wires; for an H3-anchored candidate the income-view screen has '
                    f'{len(inc_groups) - 1} distinct non-alias comparator wires (SUCC is an exact alias)'},
        'status': 'development evidence on adaptively used MODEL_SELECT outputs; reader-relative; not a protection claim',
    }
    C.write_json(OUT, rec)
    print(json.dumps({'old_gate_fails_against': old_fail, 'AGG1_fails_against': agg_fail,
                      'h3_is_succ_income': h3_is_succ_income, 'what_if': rec['source_what_if_part_A']['what_if_outcome'],
                      'old_reproduces_source': rec['source_what_if_part_A']['old_gate_reproduces_source_recipient_failures_all'],
                      'occupation_wire_groups': occ_groups, 'merge_problems': len(problems)}, indent=1))


if __name__ == '__main__':
    main()
