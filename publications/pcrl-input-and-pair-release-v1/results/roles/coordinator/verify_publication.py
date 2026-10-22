"""Verify a pushed publication commit of the PCRL input-and-pair release campaign on the GitHub remote and write
PUBLICATION_RECEIPT.json (coordinator; label-free; read-only against the remote).

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/verify_publication.py --commit <publication commit> \
        --research-commit <research commit the package was built from> --build-json <build summary JSON>

Checks: the remote branch tip (git ls-remote) equals the commit; a fresh fetch of the branch has the same tree; the GitHub API
reports the same commit and a non-truncated tree with the same file list; every gated registration file in the published tree has
the same blob as in the registration receipt e909fb8c0; every published ipr/**/*.py blob equals the research commit's file or its
sanitized copy is recorded by the build; the documents' raw URLs answer HTTP 200.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402

REMOTE = 'https://github.com/Bostesa/PCRL.git'
BRANCH = 'publication/pcrl-input-and-pair-release-v1'
RECEIPT = 'e909fb8c01e3e05b884b41fe9524a29d6e7cafa4'
PREFIX = 'publications/pcrl-input-and-pair-release-v1'
DOCS = ['START_HERE.md', 'RESEARCH_DECISION.md', 'ADVISOR_BRIEF.md', 'TABLES.md', 'VERIFICATION.md', 'HANDOFF.md', 'QUICKSTART.md']


def git(*a):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'git {a}: {r.stderr.strip()}')
    return r.stdout.strip()


def gh(path):
    r = subprocess.run(['gh', 'api', path], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'gh api {path}: {r.stderr.strip()}')
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--commit', required=True)
    ap.add_argument('--research-commit', required=True)
    ap.add_argument('--build-json', required=True)
    ap.add_argument('--pushed-utc', required=True)
    a = ap.parse_args()
    commit = git('rev-parse', a.commit)
    build = json.loads(Path(a.build_json).read_text())
    tip = git('ls-remote', REMOTE, f'refs/heads/{BRANCH}').split()[0]
    git('fetch', '-q', REMOTE, BRANCH)
    fetched = git('rev-parse', 'FETCH_HEAD')
    tree_local, tree_fetched = git('rev-parse', f'{commit}^{{tree}}'), git('rev-parse', 'FETCH_HEAD^{tree}')
    api_commit = gh(f'repos/Bostesa/PCRL/commits/{commit}')
    api_tree = gh(f'repos/Bostesa/PCRL/git/trees/{tree_local}?recursive=1')
    api_files = sorted(e['path'] for e in api_tree['tree'] if e['type'] == 'blob')
    local_files = sorted(git('ls-tree', '-r', '--name-only', commit).splitlines())
    pub_files = [f for f in local_files if f.startswith(PREFIX + '/')]
    gated = {}
    for f in C.REGISTRATION_FILES:
        rel = str(Path(f).relative_to(C.RESULTS)) if Path(f).is_absolute() else str(f).split('pcrl_input_and_pair_release_v1/', 1)[-1]
        path = f'{PREFIX}/results/{rel}'
        gated[rel] = {'receipt_blob': git('rev-parse', f'{RECEIPT}:{path}'), 'published_blob': git('rev-parse', f'{commit}:{path}')}
    code = {}
    from mechanism_design.publish_build import sanitise     # the build's sanitizer (ipr.publish_build applies it to every code file)
    for f in pub_files:
        if f.startswith(f'{PREFIX}/code/') and f.endswith('.py'):
            src = 'ipr/' + f[len(f'{PREFIX}/code/ipr/'):] if f.startswith(f'{PREFIX}/code/ipr/') else None
            if src is None:
                continue
            same = git('rev-parse', f'{commit}:{f}') == git('rev-parse', f'{a.research_commit}:{src}')
            if same:
                code[src] = 'blob_identical'
                continue
            pub_text = subprocess.run(['git', '-C', str(C.ROOT), 'show', f'{commit}:{f}'], capture_output=True, text=True).stdout
            res_text = subprocess.run(['git', '-C', str(C.ROOT), 'show', f'{a.research_commit}:{src}'], capture_output=True,
                                      text=True).stdout
            code[src] = 'sanitized' if pub_text == sanitise(res_text) else 'DIFFERS'
    links = {}
    for d in DOCS:
        url = f'https://raw.githubusercontent.com/Bostesa/PCRL/{commit}/{PREFIX}/results/{d}'
        r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', url], capture_output=True, text=True)
        links[d] = r.stdout.strip()
    rec = {'schema': 'IPR1-publication-receipt-v1', 'remote_url': REMOTE, 'branch': BRANCH, 'publication_commit': commit,
           'parent_registration_receipt': RECEIPT, 'built_from_research_commit': git('rev-parse', a.research_commit),
           'research_branch_pushed': False, 'pushed_utc': a.pushed_utc, 'verified_utc': C.now(), 'build': build,
           'remote_checks': {'ls_remote_tip': tip, 'ls_remote_equals_commit': tip == commit, 'fetched_head': fetched,
                             'fetched_tree_equals_local_tree': tree_fetched == tree_local,
                             'github_api_commit_sha': api_commit.get('sha'), 'github_api_commit_equals': api_commit.get('sha') == commit,
                             'github_api_tree_truncated': bool(api_tree.get('truncated')), 'published_files': len(pub_files),
                             'github_api_files': len([f for f in api_files if f.startswith(PREFIX + '/')]),
                             'github_api_tree_equals_local': api_files == local_files,
                             'gated_registration_files_equal_receipt': all(v['receipt_blob'] == v['published_blob'] for v in gated.values()),
                             'code_files': {'blob_identical': sum(1 for v in code.values() if v == 'blob_identical'),
                                            'sanitized': sorted(k for k, v in code.items() if v == 'sanitized'),
                                            'differs': sorted(k for k, v in code.items() if v == 'DIFFERS')},
                             'links': links, 'links_all_200': all(v == '200' for v in links.values())},
           'gated': gated}
    ok = (rec['remote_checks']['ls_remote_equals_commit'] and rec['remote_checks']['fetched_tree_equals_local_tree'] and
          rec['remote_checks']['github_api_commit_equals'] and not rec['remote_checks']['github_api_tree_truncated'] and
          rec['remote_checks']['github_api_tree_equals_local'] and rec['remote_checks']['gated_registration_files_equal_receipt'] and
          not rec['remote_checks']['code_files']['differs'] and rec['remote_checks']['links_all_200'])
    rec['all_ok'] = bool(ok)
    C.write_json(C.RESULTS / 'PUBLICATION_RECEIPT.json', rec)
    print(json.dumps({'all_ok': ok, **{k: v for k, v in rec['remote_checks'].items() if k not in ('code_files', 'links')},
                      'code_differs': rec['remote_checks']['code_files']['differs'], 'links': links}, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
