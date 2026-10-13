"""Queued custody-only off-device copy of an existing versioned snapshot.

The coordinator supplies an admitted custody spec. This command never acquires
an independent heavy slot or reopens the scientific queue by itself.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--spec',required=True);a=p.parse_args()
    # Metadata check only; custody_worker admits every actual copied byte.
    cfg=json.loads(Path(a.spec).read_text())
    if cfg.get('operation')!='off_device':raise PermissionError('explicit queued off-device custody spec required')
    from .custody_worker import run
    result=run(a.spec);print(json.dumps(result))

if __name__=='__main__':main()
