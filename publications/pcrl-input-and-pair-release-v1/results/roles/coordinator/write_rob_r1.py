"""Write the robustness bank IPR1_ROB_R1 of IA revision R1 (label-free; coordinator), used only if IA_OSG is IA's strongest feasible
nonzero release (claims/STRONGEST.json; PROTOCOL 6.3: D2R even if the interval crosses zero). Members: the revision's CAND, the shared
twin, every saved point of its gating control paths, its ablation and the references, on the robustness cells of IPR1_ROB (R3, R4 x
k0-k3; R1, R2 x k2, k3); the registered cells and every member already read by IPR1_R1 / IPR1_ROB are reader-cache hits. Written
before any output of its grid.

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/write_rob_r1.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402
from ipr import banks as BK          # noqa: E402


def main():
    p = json.loads((C.RESULTS / 'claims' / 'PLAN_IA_R1.json').read_text())
    if not p.get('cand'):
        raise SystemExit('no candidate')
    members = dict(p['members'])
    cands = [p['cand']]
    for a in p.get('ablations') or []:
        members[a] = f'IA/{a}'
        cands.append(a)
    out = BK.write('IPR1_ROB_R1', "robustness cells (R3, R4 x k0-k3; R1, R2 x k2, k3; registered cells as cache hits) for IA revision R1's "
                                  'CAND (card IA_OSG_R1), the shared twin, every saved point of its control paths, its ablation and the '
                                  "references; written before any output of its grid (PROTOCOL 6.3, D2R descriptive)", members, cands,
                   reps=['R1', 'R2', 'R3', 'R4'], reader_seeds=[0, 1, 2, 3])
    print(json.dumps({'spec': str(out.relative_to(C.ROOT)), 'members': sorted(members), 'candidates': cands}))


if __name__ == '__main__':
    main()
