"""Queue for the 2026-10-16 correction pass on the mechanism-design campaign.

It reuses the campaign queue (mechanism_design/jobqueue.py) unchanged and only swaps its budget constants and its
ledger directory before running the same CLI:
- ledger under .private/mechanism_design/correction_v1/compute (separate from the campaign's own ledger);
- 2 CPU-hours (failed attempts and agent allowances included), 2 elapsed hours from the correction pass's first
  action (2026-10-16T12:36:49Z), at most two numerical jobs at once, one thread per job, 2 GiB combined RSS.
The correction pass reads no real-person data; every unit is a small data-free exact check.
"""
from __future__ import annotations

import sys
from pathlib import Path

from mechanism_design import jobqueue as q

q.ROOT = Path(__file__).resolve().parent.parent.parent / '.private' / 'mechanism_design' / 'correction_v1' / 'compute'
q.CPU_LIMIT_S = 2 * 3600
q.ELAPSED_LIMIT_H = 2
q.RSS_LIMIT_GIB = 2.0
q.SLOTS = 2
q.CAMPAIGN_START_UTC = '2026-10-16T12:36:49+00:00'   # first action of the correction pass (worktree inspection)

if __name__ == '__main__':
    sys.exit(q.main())
