"""Single resumable scientific job pool for PCRL task-aligned cuts.

This runner never creates cloud resources.  It runs a frozen, hash-pinned JSON
queue with immutable unit IDs, output ownership, and dependency edges.  Each
scientific command is an argv list; shell interpretation is forbidden.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

import psutil


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(temp, path)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_queue(queue: dict[str, Any], root: Path) -> None:
    if queue.get('schema') != 'pcrl-task-aligned-queue-v1' or not queue.get('frozen_before_comparative_outcomes'):
        raise ValueError('queue is not the frozen study queue')
    protocol = root / 'results/pcrl_task_aligned_cuts_v1/PROTOCOL.md'
    if not protocol.is_file() or _sha(protocol) != queue.get('protocol_sha256'):
        raise ValueError('protocol hash mismatch; scientific jobs cannot start')
    lock_path = root / 'results/pcrl_task_aligned_cuts_v1/PROTOCOL_LOCK.json'
    if not lock_path.is_file():
        raise ValueError('PROTOCOL_LOCK.json is required before scientific jobs')
    lock = json.loads(lock_path.read_text())
    if lock.get('protocol_sha256') != queue['protocol_sha256']:
        raise ValueError('protocol lock does not match frozen queue/protocol')
    if not isinstance(queue.get('units'), list) or not queue['units']:
        raise ValueError('empty or invalid frozen queue')
    ids: set[str] = set()
    outputs: set[Path] = set()
    for unit in queue['units']:
        ident = unit.get('id')
        argv = unit.get('argv')
        output = unit.get('output_dir')
        marker = unit.get('complete_marker', 'COMPLETE.json')
        if not isinstance(ident, str) or not ident or '/' in ident or ident in ids:
            raise ValueError(f'bad/duplicate immutable unit ID: {ident}')
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
            raise ValueError(f'unit {ident} requires a literal argv array')
        if not isinstance(output, str) or not output or Path(output).is_absolute() or '..' in Path(output).parts:
            raise ValueError(f'unsafe output directory in {ident}')
        full_output = (root / output).resolve()
        if not full_output.is_relative_to(root.resolve()) or full_output in outputs:
            raise ValueError(f'duplicate/nonlocal output owner: {ident}')
        if not isinstance(marker, str) or Path(marker).name != marker:
            raise ValueError(f'invalid completion marker: {ident}')
        ids.add(ident); outputs.add(full_output)
    for unit in queue['units']:
        deps = unit.get('dependencies', [])
        if not isinstance(deps, list) or len(set(deps)) != len(deps) or any(dep not in ids or dep == unit['id'] for dep in deps):
            raise ValueError(f'invalid dependency: {unit["id"]}')


def _complete_marker(root: Path, unit: dict[str, Any], queue_sha: str) -> dict[str, Any] | None:
    path = root / unit['output_dir'] / unit.get('complete_marker', 'COMPLETE.json')
    if not path.is_file():
        return None
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get('unit_id') != unit['id'] or marker.get('queue_sha256') != queue_sha:
        return None
    artifacts = marker.get('artifacts')
    if not isinstance(artifacts, dict) or not artifacts:
        return None
    base = path.parent.resolve()
    for name, expected in artifacts.items():
        file = (base / name).resolve()
        if not file.is_relative_to(base) or not file.is_file() or _sha(file) != expected:
            return None
    return {'marker': str(path.relative_to(root)), 'marker_sha256': _sha(path),
            'artifacts': artifacts}


def write_completion(output_dir: Path, unit_id: str, queue_sha256: str,
                     artifact_names: list[str]) -> Path:
    """Write a verified unit receipt after scientific outputs are closed.

    A caller should pass the immutable unit ID and queue hash provided in
    ``PCRL_UNIT_ID`` and ``PCRL_QUEUE_SHA256`` by this scheduler.  Artifact
    paths must be files inside the unit's output directory.
    """
    output_dir = output_dir.resolve()
    if not unit_id or not queue_sha256 or not artifact_names:
        raise ValueError('incomplete unit receipt')
    artifacts: dict[str, str] = {}
    for name in artifact_names:
        path = (output_dir / name).resolve()
        if (Path(name).is_absolute() or not path.is_relative_to(output_dir)
                or not path.is_file() or name == 'COMPLETE.json' or name in artifacts):
            raise ValueError(f'invalid unit artifact: {name}')
        artifacts[name] = _sha(path)
    marker = output_dir / 'COMPLETE.json'
    if marker.exists():
        raise FileExistsError(marker)
    _atomic(marker, {'unit_id': unit_id, 'queue_sha256': queue_sha256,
                     'artifacts': artifacts, 'completed_utc': _now()})
    return marker


def _execute(root: Path, unit: dict[str, Any], queue_sha: str,
             deadline: dt.datetime, log_dir: Path) -> dict[str, Any]:
    ident = unit['id']
    output = root / unit['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{ident}.log'
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                'MKL_NUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1',
                'VECLIB_MAXIMUM_THREADS':'1', 'PYTHONUNBUFFERED':'1',
                'PCRL_QUEUE_SHA256':queue_sha, 'PCRL_UNIT_ID':ident})
    started = _now(); timer = time.perf_counter()
    with log_path.open('a') as stream:
        process = subprocess.Popen(unit['argv'], cwd=root, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            if dt.datetime.now(dt.timezone.utc) >= deadline:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(1)
    marker = _complete_marker(root, unit, queue_sha)
    status = 'complete' if process.returncode == 0 and marker else ('deadline_stop' if dt.datetime.now(dt.timezone.utc) >= deadline else 'incident')
    return {'id': ident, 'status': status, 'started_utc': started,
            'completed_utc': _now(), 'seconds': time.perf_counter() - timer,
            'pid': process.pid, 'returncode': process.returncode,
            'log': str(log_path.resolve().relative_to(root)), 'completion': marker}


def run_queue(root: Path, queue_path: Path, *, state_path: Path,
              workers: int, deadline: dt.datetime,
              only_ids: tuple[str, ...] = (), tiers: tuple[str, ...] = (),
              retry_incident: bool = False,
              minimum_free_memory_bytes: int = 4 * 1024**3,
              minimum_free_disk_bytes: int = 8 * 1024**3) -> dict[str, Any]:
    root = root.resolve(); queue_path = queue_path.resolve()
    queue = json.loads(queue_path.read_text())
    validate_queue(queue, root)
    if not only_ids and not tiers:
        raise ValueError('select --only-id or --tier; the full queue never launches implicitly')
    all_ids = {unit['id'] for unit in queue['units']}
    unknown = set(only_ids) - all_ids
    if unknown:
        raise ValueError(f'unknown immutable unit IDs: {sorted(unknown)}')
    selected = [unit for unit in queue['units']
                if unit['id'] in only_ids or unit.get('tier') in tiers]
    if not selected:
        raise ValueError('selector matched no frozen units')
    queue_sha = _sha(queue_path)
    lock = json.loads((root / 'results/pcrl_task_aligned_cuts_v1/PROTOCOL_LOCK.json').read_text())
    if lock.get('run_queue_sha256') != queue_sha:
        raise ValueError('protocol lock does not pin the exact run queue bytes')
    if workers < 1 or workers > (psutil.cpu_count(logical=False) or 1):
        raise ValueError('workers exceed physical-core cap')
    if deadline.tzinfo is None:
        raise ValueError('deadline must have time zone')
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + '.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get('queue_sha256') != queue_sha:
                raise ValueError('queue changed after execution; preserve old state')
        else:
            state = {'queue_sha256': queue_sha, 'created_utc': _now(),
                     'units': {}, 'attempts': [], 'status': 'registered'}
            _atomic(state_path, state)
        units = {u['id']:u for u in queue['units']}
        for ident, unit in units.items():
            marker = _complete_marker(root, unit, queue_sha)
            old = state['units'].get(ident)
            if marker:
                state['units'][ident] = {'status':'complete', 'completion':marker}
            elif old and old.get('status') == 'complete':
                raise RuntimeError(f'previously accepted unit {ident} lost its verified artifact')
            elif (root / unit['output_dir'] / unit.get('complete_marker','COMPLETE.json')).exists():
                raise RuntimeError(f'invalid completion marker for {ident}; quarantine before retry')
        for unit in selected:
            ident = unit['id']
            if state['units'].get(ident, {}).get('status') == 'complete':
                continue
            attempts = sum(record.get('id') == ident for record in state['attempts'])
            if attempts >= 2:
                raise RuntimeError(f'{ident} exhausted the two-attempt technical retry limit')
            previous = state['units'].get(ident, {}).get('status')
            if attempts and not retry_incident:
                raise RuntimeError(f'{ident} has previous {previous}; diagnose and pass --retry-incident once')
            output_dir = root / unit['output_dir']
            if output_dir.is_dir() and any(output_dir.iterdir()):
                raise RuntimeError(f'{ident} has partial output; preserve/quarantine it before retry')
        _atomic(state_path, state)
        pending = [u for u in selected if state['units'].get(u['id'], {}).get('status') != 'complete']
        active = {}
        logs = state_path.parent / 'logs'
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending or active:
                if _sha(queue_path) != queue_sha or _sha(root / 'results/pcrl_task_aligned_cuts_v1/PROTOCOL.md') != queue['protocol_sha256']:
                    raise RuntimeError('frozen queue/protocol changed during execution')
                now = dt.datetime.now(dt.timezone.utc)
                if now >= deadline:
                    for unit in pending:
                        state['units'][unit['id']] = {'status':'not_started_deadline'}
                    pending = []
                while pending and len(active) < workers and now < deadline:
                    memory = psutil.virtual_memory().available
                    disk = shutil.disk_usage(root).free
                    if disk < minimum_free_disk_bytes:
                        state['status'] = 'disk_safety_stop'; _atomic(state_path, state)
                        raise RuntimeError('free disk below registered reserve')
                    if memory < minimum_free_memory_bytes:
                        break
                    ready = next((i for i,u in enumerate(pending) if all(
                        state['units'].get(d, {}).get('status') == 'complete'
                        for d in u.get('dependencies', []))), None)
                    if ready is None:
                        break
                    unit = pending.pop(ready)
                    active[pool.submit(_execute, root, unit, queue_sha, deadline, logs)] = unit
                    state['units'][unit['id']] = {'status':'running', 'launched_utc':_now()}
                    _atomic(state_path, state)
                if active:
                    done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        unit = active.pop(future)
                        try:
                            outcome = future.result()
                        except Exception as error:
                            outcome = {'id':unit['id'], 'status':'scheduler_error',
                                       'error':f'{type(error).__name__}: {error}',
                                       'completed_utc':_now()}
                        state['attempts'].append(outcome)
                        state['units'][unit['id']] = outcome
                        _atomic(state_path, state)
                elif pending:
                    if psutil.virtual_memory().available < minimum_free_memory_bytes:
                        time.sleep(5)
                    else:
                        for unit in pending:
                            state['units'][unit['id']] = {'status':'dependency_blocked'}
                        pending = []
                state['updated_utc'] = _now()
                _atomic(state_path, state)
        selected_complete = all(state['units'].get(u['id'], {}).get('status') == 'complete'
                                for u in selected)
        full_complete = all(state['units'].get(u['id'], {}).get('status') == 'complete'
                            for u in queue['units'])
        state['status'] = 'complete' if full_complete else ('selected_complete' if selected_complete else 'incomplete')
        state['last_selected_ids'] = [u['id'] for u in selected]
        state['finished_utc'] = _now(); _atomic(state_path, state)
        return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--only-id', action='append', default=[])
    parser.add_argument('--tier', action='append', default=[])
    parser.add_argument('--retry-incident', action='store_true',
                        help='allow one identical retry after explicit diagnosis and artifact quarantine')
    args = parser.parse_args(argv)
    deadline = dt.datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    outcome = run_queue(args.root, args.queue, state_path=args.state,
                        workers=args.workers, deadline=deadline,
                        only_ids=tuple(args.only_id), tiers=tuple(args.tier),
                        retry_incident=args.retry_incident)
    print(json.dumps({'status':outcome['status'], 'completed':sum(v.get('status') == 'complete' for v in outcome['units'].values()),
                      'units':len(outcome['units'])}, sort_keys=True))


if __name__ == '__main__':
    main()
