"""Queue entry point for the confirmation-runner rehearsals (synthetic and development only)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def run(spec_path):
    from .confirmation_runner import _sha
    from .runner_rehearsal import rehearse
    cfg = json.loads(Path(spec_path).read_text())
    if cfg['mode'] not in ('synthetic-rehearsal', 'development-rehearsal'):
        raise PermissionError('only rehearsal modes run in the queue; execute-fresh is hard-disabled')
    if _sha(cfg['protocol']) != cfg['protocol_sha256']:
        raise PermissionError('protocol sha256 differs')
    if cfg['mode'] == 'synthetic-rehearsal':
        from .evaluation_queue import queue_config
        queue_config(spec_path, 'verification')   # binds the unit to the queue (no role data is read)
    return rehearse(cfg['mode'], json.loads(Path(cfg['protocol']).read_text()), spec_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
