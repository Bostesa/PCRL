"""Bind every byte a numerical worker reads to the active frozen-tree control-completion queue."""
import hashlib, json, os
from pathlib import Path


def queue_config(spec_path, stage, allow_inner_truth=False):
    """Bind every byte read by a numerical worker to the new durable queue."""
    from .scheduler import assert_active_from_env
    active = assert_active_from_env(stage=stage)
    state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    unit = state['units'][active['unit_id']]
    registered = {str(Path(i['path']).resolve()): i['sha256']
                  for i in unit['spec']['sources']}
    def verify(path, expected):
        p = Path(path)
        if p.is_symlink() or not expected or registered.get(str(p.resolve())) != expected:
            raise PermissionError('source absent from active frozen-tree control-completion queue')
        # INNER truth must not be hashed before all reporting dependencies freeze.
        low = p.as_posix().lower()
        from .admission import inner_truth_path
        if inner_truth_path(p):
            if not allow_inner_truth:
                raise PermissionError('INNER truth requires reporting-specific admission')
            from .admission import reporting_admitted_path
            reporting_admitted_path(p)
        with p.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise PermissionError('active queue source hash differs')
        return p
    p = Path(spec_path)
    verify(p, registered.get(str(p.resolve())))
    cfg = json.loads(p.read_text())
    if cfg['unit_id'] != active['unit_id']:
        raise PermissionError('worker and queue unit differ')
    def guard():
        assert_active_from_env(unit_id=active['unit_id'], stage=stage)
        verify(p, registered[str(p.resolve())])
        for name in ("science_lock", "training_complete_lock", "selection_lock", "reporting_lock"):
            if name in cfg:
                verify(cfg[name], cfg[name+'_sha256'])
    guard()
    return active, cfg, verify, guard


