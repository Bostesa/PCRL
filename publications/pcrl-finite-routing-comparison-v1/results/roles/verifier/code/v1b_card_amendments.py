"""V1b card amendments after use (label-free; adjudication input, never a widening of rule R6 of v1_access.py).

For every 'fit' / 'calibrate' read of the access log whose governing card changed after the read (v1_access rule
R6_card_changed_after_use), establish with own git calls:
- the logged card bytes: the blob at the logged card commit, whose SHA-256 must equal the logged card_sha256;
- whether the change is APPEND-ONLY: the logged bytes are an exact byte prefix of the current bytes (nothing removed or edited);
- the commits that changed the card after the logged commit, with their commit times, and whether each precedes the reads that
  log it;
- the appended text (line count and its first heading line).
Status: procedural ATTENTION when every changed card is append-only and every logged version is recoverable from git (the text that
governed the read is unchanged); FAIL otherwise. Numerical NOT_APPLICABLE.

    python3 .../v1b_card_amendments.py --tag V2
    python3 .../v1b_card_amendments.py --selftest
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402


def blob_bytes(commit, relpath):
    r = subprocess.run(['git', '-C', str(V.ROOT), 'show', f'{commit}:{relpath}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def append_only(old, new):
    return old is not None and new is not None and len(new) >= len(old) and new[:len(old)] == old


def analyse(events):
    groups = {}
    for i, e in enumerate(events):
        if e.get('operation') not in ('fit', 'calibrate') or not e.get('card'):
            continue
        key = (e['card'], e.get('card_commit'), e.get('card_sha256'))
        groups.setdefault(key, []).append({'event': i, 'utc': e['utc'], 'unit_id': e.get('unit_id')})
    out, ok = [], True
    for (card, commit, sha), reads in groups.items():
        now_p = V.ROOT / card
        now = now_p.read_bytes() if now_p.is_file() else None
        old = blob_bytes(commit, card) if commit else None
        rec = {'card': card, 'logged_commit': commit, 'logged_sha256': sha, 'reads': reads,
               'logged_bytes_recoverable': old is not None and V.bytes_sha(old) == sha,
               'changed_after_use': now is None or V.bytes_sha(now) != sha}
        if rec['changed_after_use']:
            rec['append_only'] = append_only(old, now)
            if rec['append_only']:
                tail = now[len(old):].decode('utf-8', 'replace')
                lines = [x for x in tail.splitlines() if x.strip()]
                rec['appended_lines'] = len(lines)
                rec['appended_first_line'] = lines[0][:160] if lines else ''
            rc, log, _ = V.git('log', '--format=%H %cI', f'{commit}..HEAD', '--', card)
            rec['later_card_commits'] = [{'commit': c.split()[0], 'commit_utc': c.split()[1]} for c in log.splitlines() if c.strip()]
            first_read = min(V.parse_utc(r['utc']) for r in reads)
            rec['every_later_commit_after_these_reads'] = all(V.parse_utc(c['commit_utc']) > first_read for c in rec['later_card_commits'])
            ok &= bool(rec['append_only'] and rec['logged_bytes_recoverable'])
        else:
            ok &= rec['logged_bytes_recoverable']
        out.append(rec)
    return out, ok


def run(tag):
    events = V.jsonl(V.ACCESS_LOG)
    out, ok = analyse(events)
    changed = [r for r in out if r['changed_after_use']]
    chk = {'id': f'V1B.CARD_AMENDMENTS_AFTER_USE__{tag}',
           'what': 'cards changed after a fitting read that they governed (v1_access R6_card_changed_after_use): logged bytes '
                   'recoverable from git, append-only test (logged bytes an exact prefix of the current bytes), later card commits '
                   'and their times; adjudication input, the R6 FAIL of V1.ACCESS_LOG stands as recorded',
           'inputs': V.inputs_sha([V.ACCESS_LOG] + sorted({V.ROOT / r['card'] for r in out})),
           'result': {'groups': out, 'n_reads_with_changed_card': sum(len(r['reads']) for r in changed)},
           'numerical_status': 'NOT_APPLICABLE',
           'procedural_status': ('PASS' if not changed else ('ATTENTION' if ok else 'FAIL'))}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps(chk['result'], indent=1)[:4000])
    return 0


def selftest():
    ok = {'append_only_prefix': append_only(b'abc\n', b'abc\n## A1\nx\n'), 'edit_not_append_only': not append_only(b'abc\n', b'abd\n## A1\n'),
          'shorter_not_append_only': not append_only(b'abc\n', b'ab'), 'missing_not_append_only': not append_only(None, b'x')}
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
