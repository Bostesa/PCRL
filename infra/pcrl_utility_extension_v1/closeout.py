"""Cloud closeout: archive this run's results directory into verified chunks (upload, then an
independent read-back + extraction + per-file sha256 verification), and write a summary."""
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import archive

cfg = json.loads(Path(os.environ['PCRL_UX_RUNCFG']).read_text())
out = Path(os.environ['PCRL_UX_OUT'])
work = Path(cfg['scratch']) / 'closeout'; work.mkdir(parents=True, exist_ok=True)
groups = [{'label': 'results_run', 'root': str(out.parents[1]), 'subdirs': [str(out.relative_to(out.parents[1]))],
           'note': 'pcrl_utility_extension_v1 cloud run outputs (releases, fits, audits incl. fitted attackers)',
           'readers': ['program.point / screen', 'report generation']}]
(work / 'groups.json').write_text(json.dumps(groups))
archive.plan(groups, str(work / 'plan.json'))
archive.upload(str(work / 'plan.json'), set(), cfg['bucket'], cfg['prefix'], str(work / 'manifests'), threads=4)
mans = sorted((work / 'manifests').glob('*.json'))
archive.verify(cfg['bucket'], cfg['prefix'], [f"{cfg['prefix']}/manifests/{m.name}" for m in mans],
               str(Path(cfg['scratch'])), str(work / 'verify'))
ver = [json.loads(v.read_text()) for v in (work / 'verify').glob('*.verify.json')]
summary = {'chunks': len(mans), 'verified': sum(v['verified'] for v in ver), 'all_verified': len(ver) == len(mans) and all(v['verified'] for v in ver)}
(out / '_scheduler' / 'RESULTS_ARCHIVE.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
