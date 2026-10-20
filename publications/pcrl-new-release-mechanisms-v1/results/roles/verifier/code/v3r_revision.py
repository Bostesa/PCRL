"""V3R revision order (label-free; git, the access log and the configuration ledger): a revision card and its written approvals
(the coordinator's proposal / decision and the statistical reviewer's ruling) are committed BEFORE the first label read under the
revision card and before its first saved configuration; the revision is disclosed in PROTOCOL_DEVIATIONS.md; optional identity
checks of a declared frozen base against the admitted reference wires.

    python3 .../v3r_revision.py --tag V1 --card cards/FB_FSC_v2.md --approval roles/coordinator/REVISION_FB_R1_PROPOSAL.md
            --approval roles/statistical_reviewer/RULING_FB_R1_FINAL.md [--deviation-text FB2] [--base FB/FB2_BASE=TWJ:income,W:occupation]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import vref  # noqa: E402


def first_commit(rel):
    rc, out, _ = V.git('log', '--format=%H %cI', '--', rel)
    lines = out.splitlines()
    if not lines:
        return None, None
    h, t = lines[-1].split()
    return h, V.parse_utc(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--card', required=True); ap.add_argument('--approval', action='append', default=[])
    ap.add_argument('--deviation-text', default=None); ap.add_argument('--base', action='append', default=[])
    a = ap.parse_args()
    card_rel = f'results/{V.CAMPAIGN}/{a.card}'
    c_hash, c_time = first_commit(card_rel)
    apps = {}
    for x in a.approval:
        h, t = first_commit(f'results/{V.CAMPAIGN}/{x}')
        apps[x] = {'commit': h, 'utc': t.isoformat() if t else None, 'before_card': bool(t and c_time and t < c_time)}
    reads = [e for e in V.jsonl(V.ACCESS_LOG) if e.get('card') == card_rel]
    first_read = min((V.parse_utc(e['utc']) for e in reads), default=None)
    saves = [e for e in V.jsonl(V.CONFIG_LEDGER) if e.get('card') == card_rel]
    first_save = min((V.parse_utc(e['utc']) for e in saves), default=None)
    dev = (V.RESULTS / 'PROTOCOL_DEVIATIONS.md').read_text()
    disclosed = (a.deviation_text in dev) if a.deviation_text else None
    bases = {}
    adm = json.loads(vref.NRM_ADMISSION.read_text())['reference_releases_MODEL_SELECT_wire_sha256']
    for spec in a.base:
        name, parts = spec.split('=')
        fam, cfg = name.split('/')
        want = dict(p.split(':')[::-1] for p in parts.split(','))      # task -> reference
        for v in (0, 1):
            arr, rec, prov = vref.release(fam, cfg, v)
            ms = arr['MODEL_SELECT']
            bases[f'{name}/V{v}'] = {t: V.wire_sha(ms[t]) == adm[f'{ref}/S{v}'][t] for t, ref in want.items()}
    ok = bool(c_time) and all(x['before_card'] for x in apps.values()) and apps and \
        (first_read is None or first_read > c_time) and (first_save is None or first_save > c_time) and disclosed is not False
    base_ok = all(all(x.values()) for x in bases.values())
    res = {'card': card_rel, 'card_commit': c_hash, 'card_commit_utc': c_time.isoformat() if c_time else None, 'approvals': apps,
           'first_read_under_card_utc': first_read.isoformat() if first_read else None, 'n_reads_under_card': len(reads),
           'first_save_under_card_utc': first_save.isoformat() if first_save else None, 'n_configurations_under_card': len(saves),
           'deviation_text': a.deviation_text, 'disclosed_in_PROTOCOL_DEVIATIONS': disclosed, 'base_identity': bases}
    chk = {'id': f'V3R.REVISION.{Path(a.card).stem}__{a.tag}',
           'what': f'revision order for {a.card}: approvals committed before the card, card before the first read and the first save; '
                   'disclosure; declared frozen base equals the admitted reference wires',
           'inputs': V.inputs_sha([V.ROOT / card_rel] + [V.RESULTS / x for x in a.approval] + [V.RESULTS / 'PROTOCOL_DEVIATIONS.md']),
           'result': res, 'numerical_status': ('PASS' if base_ok else 'FAIL') if bases else 'NOT_APPLICABLE',
           'procedural_status': 'PASS' if ok else 'FAIL'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps(res, indent=1, default=V._jd))
    return 0


if __name__ == '__main__':
    sys.exit(main())
