"""Background verifier on the cloud host: every 5 minutes, independently read back and verify any
laptop-uploaded archive chunk (manifests/) that has no verification record yet. Stops at deadline."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import archive

cfg = json.loads(Path(os.environ['PCRL_UX_RUNCFG']).read_text())
b, p = cfg['bucket'], cfg['prefix']
vdir = Path(cfg['verify_dir']); scratch = Path(cfg['scratch']); scratch.mkdir(parents=True, exist_ok=True)
while time.time() < cfg['deadline_epoch'] - 1800:
    ls = subprocess.run(['aws', 's3', 'ls', f's3://{b}/{p}/manifests/'], capture_output=True, text=True).stdout
    names = [l.split()[-1] for l in ls.splitlines() if l.strip().endswith('.json')]
    todo = [n for n in names if not n.startswith('exec_') and not n.startswith('results_')
            and not (vdir / n.replace('.json', '.verify.json')).exists()]
    for n in todo:
        archive.verify(b, p, [f'{p}/manifests/{n}'], str(scratch), str(vdir))
    if not todo:
        time.sleep(300)
