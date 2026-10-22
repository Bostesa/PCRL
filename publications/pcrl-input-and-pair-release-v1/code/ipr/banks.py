"""Reader-grid bank specifications of the PCRL input-and-pair release campaign (label-free; written to results/<campaign>/grid_specs/;
derived from nrm.banks: members of this campaign are type 'ipr'; 'NRM:F/C' names a saved release of the source campaign).

The nine point-eligible references of the frozen rules (NRM1-rules-v1 BENCHMARK, adopted by IPR1), with the exact source members:
    NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ   {'type': 'reference', 'name': <name>} (ohn.devrun.reference_release)
    W                                                     {'type': 'witness'} = A1_PO_16 S<v> occupation + H3 income S<v>
    TR100                                                 {'type': 'reference', 'name': 'H3+TASK_REFIT'} (bitwise the previous
                                                          campaign's S100 TASK_REFIT + H3, per its control identity checks)
    TT100                                                 {'type': 'reference', 'name': 'H3+T_TREE'}
plus the utility references U0 and Ucal as diagnostics ({'type': 'reference', 'name': 'U0' | 'Ucal'}).
Family members: {'type': 'ipr', 'family': F, 'config': C} (this campaign), {'type': 'nrm', 'family': F, 'config': C} (spec
'NRM:F/C': the new-release-mechanisms campaign's saved release, read-only) or {'type': 'ddr', ...} (spec 'DDR:F/C': the dual-design
release campaign's saved release, read-only; the beside references DDR_TWIN, DDR_PTC_R1, DDR_ZTILT_R1).

    python3 -m ipr.banks write --bank-id ID --purpose TEXT --member NAME=FAMILY/CONFIG[@OFFSET] ... --candidate NAME ...
                               [--comparator NAME ...] [--reps R1 R2 ...] [--reader-seeds 0 1 ...] [--fractions 1.0 ...]
Every reference and diagnostic is always a member and a comparator; every family member is a comparator too (a control can win).
"""
from __future__ import annotations
import argparse
import json
import sys

from . import common as C

REFERENCE_MEMBERS = {'NEURAL_RAW': {'type': 'reference', 'name': 'NEURAL_RAW'}, 'H_LJ': {'type': 'reference', 'name': 'H_LJ'},
                     'B2TWIN': {'type': 'reference', 'name': 'B2TWIN'},
                     'RETIRED_B3CTL': {'type': 'reference', 'name': 'RETIRED_B3CTL'},
                     'SUCC': {'type': 'reference', 'name': 'SUCC'}, 'TWJ': {'type': 'reference', 'name': 'TWJ'},
                     'W': {'type': 'witness'}, 'TR100': {'type': 'reference', 'name': 'H3+TASK_REFIT'},
                     'TT100': {'type': 'reference', 'name': 'H3+T_TREE'}}
DIAGNOSTIC_MEMBERS = {'U0': {'type': 'reference', 'name': 'U0'}, 'Ucal': {'type': 'reference', 'name': 'Ucal'}}
OUT = C.RESULTS / 'grid_specs'


def family_member(spec):
    """'FA/PRC_S2' (this campaign) or 'NRM:FA/PRC_TWIN' (the source campaign, read-only) -> member dict."""
    off = None
    if '@' in spec:
        spec, off = spec.split('@')
    typ = 'ipr'
    if spec.startswith('NRM:'):
        typ, spec = 'nrm', spec[4:]
    elif spec.startswith('DDR:'):
        typ, spec = 'ddr', spec[4:]
    fam, cfg = spec.split('/')
    m = {'type': typ, 'family': fam, 'config': cfg}
    if off is not None:
        m['draw_offset'] = int(off)
    return m


def write(bank_id, purpose, members, candidates, comparators=(), reps=None, reader_seeds=None, fractions=None):
    from . import devreaders as DR
    mem = {**REFERENCE_MEMBERS, **DIAGNOSTIC_MEMBERS}
    for name, spec in members.items():
        if name in mem:
            raise ValueError(f'member name {name} collides with a reference')
        mem[name] = family_member(spec) if isinstance(spec, str) else dict(spec)      # a dict: a ready member (ipr.roles)
    comps = list(dict.fromkeys(list(REFERENCE_MEMBERS) + list(DIAGNOSTIC_MEMBERS) + list(members) + list(comparators)))
    spec = {'bank_id': bank_id, 'purpose': purpose, 'members': mem, 'candidates': list(candidates), 'comparators': comps,
            'written_utc': C.now(), 'benchmark_references': list(REFERENCE_MEMBERS), 'diagnostics': list(DIAGNOSTIC_MEMBERS)}
    if reps:
        spec['reps'] = list(reps)
    if reader_seeds:
        spec['reader_seeds'] = [int(k) for k in reader_seeds]
    if fractions:
        spec['fractions'] = [float(f) for f in fractions]
    DR.validate_spec(spec)
    out = OUT / f'{bank_id}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (a bank spec is write-once; use a new bank id)')
    C.write_json(out, spec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('write'); w.add_argument('--bank-id', required=True); w.add_argument('--purpose', required=True)
    w.add_argument('--member', action='append', default=[]); w.add_argument('--candidate', action='append', default=[])
    w.add_argument('--comparator', action='append', default=[]); w.add_argument('--reps', nargs='*')
    w.add_argument('--reader-seeds', nargs='*'); w.add_argument('--fractions', nargs='*')
    a = ap.parse_args(argv)
    members = dict(m.split('=', 1) for m in a.member)
    out = write(a.bank_id, a.purpose, members, a.candidate, a.comparator, a.reps, a.reader_seeds, a.fractions)
    print(json.dumps({'spec': str(out.relative_to(C.ROOT)), 'members': len(json.loads(out.read_text())['members'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
