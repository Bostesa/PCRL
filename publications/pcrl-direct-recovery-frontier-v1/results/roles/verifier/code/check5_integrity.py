"""Verifier check 5 (DRF1): integrity of the order of work and of the label access, from the append-only logs and git history.
Label-free (reads no label; it reads the access log's metadata, the queue ledger, the configuration ledger, the receipts and git).

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check5_integrity.py [--out FILE]

Checks:
  1. the registration receipt was verified (remote SHA = receipt commit) before the first label read of the campaign;
  2. every fitting read names a committed card whose commit time precedes the read, and the card matches the fit's revision;
     every saved release record names the same card (path and SHA-256) as the committed card file;
  3. each reading plan was committed before the first MODEL_SELECT read of the campaign and is unchanged since its commit;
     dr/BUDGETS.json was committed before the first fitting read;
  4. specification count: distinct (family, config, version) in the configuration ledger <= 32 (and the count per card);
  5. no MODEL_SELECT label read by a fitting entry point (drf.dr anywhere in the module chain); no closed-role or CALIBRATION
     label read; every read's fields allowed by its operation; every read inside a queue unit whose start precedes the read;
  6. disclosure: the uncommitted code paths and code drift recorded by the queue for every unit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
POLICY_FIELDS = {'fit': {'roles': {'MODEL_FIT'}, 'fields': {'income', 'occupation', 'sex'}},
                 'benchmark': {'roles': {'MODEL_SELECT'}, 'fields': {'income', 'occupation', 'sex'}},
                 'readers': {'roles': {'MODEL_SELECT'}, 'fields': {'sex'}},
                 'verify': {'roles': {'MODEL_FIT', 'MODEL_SELECT'}, 'fields': {'income', 'occupation', 'sex'}}}
CLOSED = {'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'CALIBRATION', 'SEALED_ASSESSMENT'}


def t(s):
    return dt.datetime.fromisoformat(str(s).replace('Z', '+00:00')).astimezone(dt.timezone.utc)


def git(*a):
    r = subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def first_commit(rel):
    out = git('log', '--diff-filter=A', '--format=%H %cI', '--', rel)
    if not out:
        return None, None
    h, when = out.splitlines()[-1].split()
    return h, t(when)


def last_commit(rel):
    out = git('log', '-1', '--format=%H %cI', '--', rel)
    if not out:
        return None, None
    h, when = out.split()
    return h, t(when)


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).is_file() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items, info = [], {}
    access = jl(PRIV / 'access' / 'LABEL_ACCESS.jsonl')
    queue = jl(PRIV / 'compute' / 'ledger.jsonl')
    cfg = jl(PRIV / 'runs' / 'CONFIG_LEDGER.jsonl')
    starts = {e['unit_id']: e for e in queue if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: e for e in queue if e.get('event') == 'unit_end'}
    info['label_reads'] = len(access)
    # 1. registration receipt before the first label read
    rec = json.loads((PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json').read_text())
    first_read = min(t(e['utc']) for e in access) if access else None
    info['first_label_read_utc'] = first_read.isoformat() if first_read else None
    info['receipt'] = {k: rec.get(k) for k in ('receipt_commit', 'remote_sha', 'remote_sha_verified', 'verified_utc', 'research_commit')}
    items.append(V.cmp_eq('1:receipt_remote_sha_equals_receipt_commit', bool(rec.get('remote_sha_verified') and
                                                                             rec.get('remote_sha') == rec.get('receipt_commit')), True))
    items.append(V.cmp_eq('1:receipt_verified_before_first_label_read', bool(first_read and t(rec['verified_utc']) < first_read), True))
    items.append(V.cmp_eq('1:receipt_commit_in_local_history', git('cat-file', '-t', rec.get('receipt_commit', '')), 'commit'))
    # 2. cards before their fits
    fit_reads = [e for e in access if e.get('operation') == 'fit']
    for e in fit_reads:
        card = e.get('card')
        h, when = first_commit(card) if card else (None, None)
        nm = f'2:fit_read:{e.get("unit_id")}:{e["utc"]}'
        items.append(V.cmp_eq(nm + ':card_commit_recorded_equals_git_first_commit', (e.get('card_commit') or '')[:40], h))
        items.append(V.cmp_eq(nm + ':card_committed_before_read', bool(when and when < t(e['utc'])), True))
        items.append(V.cmp_eq(nm + ':card_unchanged_since_commit', last_commit(card)[0], h))
        items.append(V.cmp_eq(nm + ':card_sha256_equals_file', e.get('card_sha256'), V.sha256_file(REPO / card)))
        argv = list(starts.get(e.get('unit_id'), {}).get('argv', []))
        rev = argv[argv.index('--revision') + 1] if '--revision' in argv else 'v1'     # drf.dr's default revision is v1
        # (correction 04:08Z: the first version mapped every non-R1 revision to v1 and failed the R2 reads; kept in out/ as recorded)
        items.append(V.cmp_eq(nm + ':card_matches_revision', Path(card).name, f'DR_DRF_{rev}.md'))
    for e in cfg:
        if e.get('release_sha256') is None:
            continue
        d = PRIV / 'runs' / e['family'] / e['config_id'] / f'V{e["version"]}'
        rr = json.loads((d / 'RELEASE.json').read_text())
        nm = f'2:release:{e["config_id"]}:V{e["version"]}'
        items.append(V.cmp_eq(nm + ':card_sha256_equals_committed_card', rr.get('card_sha256'), V.sha256_file(REPO / rr['mechanism_card'])))
        items.append(V.cmp_eq(nm + ':release_file_sha256', V.sha256_file(d / 'RELEASE.npz'), rr.get('release_sha256')))
        sfx = e['config_id'].rsplit('_', 1)[-1]
        expect = f'DR_DRF_{sfx}.md' if sfx in ('R1', 'R2') else 'DR_DRF_v1.md'
        items.append(V.cmp_eq(nm + ':governing_card', Path(rr['mechanism_card']).name, expect))
    # 3. plans before the first MODEL_SELECT read; budgets before the first fit
    ms_reads = sorted(t(e['utc']) for e in access if e.get('role') == 'MODEL_SELECT')
    first_ms = ms_reads[0] if ms_reads else None
    info['first_model_select_read_utc'] = first_ms.isoformat() if first_ms else None
    for pid in ('PLAN_DRF_V1', 'PLAN_DRF_R1'):
        rel = f'results/{CAMPAIGN}/claims/{pid}.json'
        h, when = first_commit(rel)
        info[f'{pid}_commit'] = [h, when.isoformat() if when else None]
        items.append(V.cmp_eq(f'3:{pid}:committed_before_first_MODEL_SELECT_read', bool(when and first_ms and when < first_ms), True))
        items.append(V.cmp_eq(f'3:{pid}:unchanged_since_first_commit', last_commit(rel)[0], h))
        items.append(V.cmp_eq(f'3:{pid}:working_tree_equals_commit', git('diff', '--quiet', 'HEAD', '--', rel) is not None, True))
    h, when = first_commit(f'results/{CAMPAIGN}/dr/BUDGETS.json')
    first_fit = min((t(e['utc']) for e in fit_reads), default=None)
    items.append(V.cmp_eq('3:budgets_committed_before_first_fit_read', bool(when and first_fit and when < first_fit), True))
    items.append(V.cmp_eq('3:budgets_unchanged_since_commit', last_commit(f'results/{CAMPAIGN}/dr/BUDGETS.json')[0], h))
    # 3b. registration status of every recorded reading (RULES.md 7): registered iff its card was committed before any MODEL_SELECT
    # outcome of the campaign (the first MODEL_SELECT read); otherwise adaptive
    for d in sorted((RES / 'claims').glob('*/DECISIONS.json')):
        dec = json.loads(d.read_text())['decisions']
        card = dec.get('card')
        h, when = first_commit(card) if card else (None, None)
        mine_reg = bool(when and first_ms and when < first_ms)
        rid = d.parent.name
        info.setdefault('readings', {})[rid] = {'card': card, 'card_first_commit_utc': when.isoformat() if when else None,
                                                'registered_verifier': mine_reg, 'registered_record': dec.get('registered'),
                                                'adaptive_tag_record': dec.get('adaptive_tag')}
        items.append(V.cmp_eq(f'3b:{rid}:registered', mine_reg, dec.get('registered')))
        items.append(V.cmp_eq(f'3b:{rid}:adaptive_tag_absent_iff_registered', dec.get('adaptive_tag') is None, mine_reg))
    # 4. specification count
    keys = {(e['family'], e['config_id'], int(e['version'])) for e in cfg}
    by_card = {}
    for e in cfg:
        by_card[Path(e['card']).name] = by_card.get(Path(e['card']).name, 0) + 1
    info['specifications'] = {'distinct': len(keys), 'entries': len(cfg), 'by_card': by_card,
                              'kinds': sorted({e['kind'] for e in cfg})}
    items.append(V.cmp_eq('4:no_duplicate_specification_entries', len(keys) == len(cfg), True))
    items.append(V.cmp_eq('4:specifications_within_32', len(keys) <= 32, True))
    items.append(V.cmp_eq('4:configuration_numbers_consecutive', [int(e['configuration_number']) for e in cfg],
                          list(range(1, len(cfg) + 1))))
    # 5. label access policy
    for e in access:
        nm = f'5:read:{e.get("unit_id")}:{e["utc"]}:{e["role"]}'
        op = e.get('operation')
        pol = POLICY_FIELDS.get(op)
        items.append(V.cmp_eq(nm + ':role_and_fields_allowed', bool(pol and e['role'] in pol['roles'] and
                                                                    set(e['fields']) <= pol['fields']), True))
        items.append(V.cmp_eq(nm + ':not_closed_role', e['role'] in CLOSED, False))
        chain = [str(m) for m in e.get('module_chain', [])] + [str(e.get('entry_point', '')).split(':')[0]]
        if e['role'] == 'MODEL_SELECT':
            items.append(V.cmp_eq(nm + ':no_fitting_module_in_chain', any(m == 'drf.dr' or m.startswith('drf.dr.') for m in chain), False))
        u = e.get('unit_id')
        st = starts.get(u)
        ok = bool(u and st and t(st['utc']) <= t(e['utc']) and (u not in ends or t(e['utc']) <= t(ends[u]['utc'])))
        items.append(V.cmp_eq(nm + ':inside_its_queue_unit', ok, True))
    # 6. disclosure of uncommitted code and drift per unit
    disc = {}
    for u, st in starts.items():
        d = {'uncommitted_code_paths': st.get('uncommitted_code_paths'),
             'digest_registration_equals_launch': st.get('code_digest_at_registration') == st.get('code_digest_at_launch'),
             'code_changed_during_run': ends.get(u, {}).get('code_changed_during_run'), 'status': ends.get(u, {}).get('status'),
             'git_head': (st.get('git_head') or '')[:9]}
        launch = PRIV / 'compute' / 'units' / u / 'CODE_AT_LAUNCH.json'
        if launch.is_file():
            d['files_changed_between_registration_and_launch'] = json.loads(launch.read_text()).get('files_changed')
        disc[u] = d
    out = {'schema': 'DRF1-verifier-check5-v1', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'info': info,
           'disclosure_units': disc, 'summary': {}, 'overall': V.summarize(items), 'items': items}
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    out['summary'] = {g: V.summarize(x) for g, x in groups.items()}
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'summary': out['summary'], 'info': info}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
