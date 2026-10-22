"""Write the robustness bank IPR1_ROB (label-free; coordinator), before any output of the registered grid: each family's CAND (the
strongest feasible nonzero release of its registered card is its only operation configuration), the shared twin, every saved point of
its gating control paths, its ablations and the references, on the robustness cells (R3, R4 with k0-k3; R1, R2 with k2, k3; the
registered cells R1, R2 x k0, k1 are cache hits). PROTOCOL 6.3 (D2R even if the registered interval crosses zero)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402
from ipr import banks as BK          # noqa: E402
from ipr import roles as RO          # noqa: E402


def main():
    members, cands = {}, []
    for fam in ('IA', 'IB'):
        p = json.loads((C.RESULTS / 'claims' / f'PLAN_{fam}_REG.json').read_text())
        if not p.get('cand'):
            continue
        cands.append(p['cand'])
        for name, spec in p['members'].items():
            members[name] = RO.array_member(name) if spec.startswith('ROLE:') else spec
        for a in p.get('ablations') or []:
            members[a] = f'{fam}/{a}'
            cands.append(a)
    out = BK.write('IPR1_ROB', 'robustness cells (R3, R4 x k0-k3; R1, R2 x k2, k3; registered cells as cache hits) for each family\'s '
                               'CAND, the shared twin, every saved control-path point, the ablations and the references; written '
                               'before any registered-grid output (PROTOCOL 6.3, D2R descriptive)', members, cands,
                   reps=['R1', 'R2', 'R3', 'R4'], reader_seeds=[0, 1, 2, 3])
    print(json.dumps({'spec': str(out.relative_to(C.ROOT)), 'members': sorted(members), 'candidates': cands}))


if __name__ == '__main__':
    main()
