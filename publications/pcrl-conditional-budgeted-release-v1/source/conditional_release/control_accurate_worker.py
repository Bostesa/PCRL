"""Registered fixed-accuracy correction, never a numerical tolerance search."""
import argparse,json
from pathlib import Path
from . import accurate_memory_lp
from .control_wide_worker import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if cfg.get('fixed_numerical_accuracy')!='LP1E-10_MIP1E-9':
        raise PermissionError('fixed engineering correction must be registered')
    print(json.dumps(run(args.config,numerical_backend=accurate_memory_lp)))
