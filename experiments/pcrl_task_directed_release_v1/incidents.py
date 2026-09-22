"""Preserve failed in-memory arrays privately without repairing or accepting them."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .run import atomic,sha


def preserve_arrays(error,directory,*,max_array_bytes=16*1024**2,max_total_bytes=128*1024**2):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    frames=[];tb=error.__traceback__
    while tb is not None:
        frame=tb.tb_frame
        if 'experiments/pcrl_task_directed_release_v1/' in frame.f_code.co_filename:
            frames.append((frame,tb.tb_lineno))
        tb=tb.tb_next
    arrays=[];seen=set();total=0;skipped=[]
    # The deepest validator frame first preserves the rejected probability row.
    for frame,line in reversed(frames):
        for name,value in sorted(list(frame.f_locals.items())):
            if not isinstance(value,np.ndarray) or id(value) in seen:continue
            seen.add(id(value))
            row={'module':Path(frame.f_code.co_filename).name,'function':frame.f_code.co_name,
                 'line':line,'variable':name,'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.nbytes}
            if value.dtype.kind not in 'biufcSU' or value.nbytes>max_array_bytes or total+value.nbytes>max_total_bytes:
                skipped.append({**row,'reason':'object dtype or fixed incident size bound'});continue
            path=directory/f'array_{len(arrays):04d}.npy'
            with path.open('xb') as handle:np.save(handle,value,allow_pickle=False)
            arrays.append({**row,'file':path.name,'sha256':sha(path),'accepted_for_scoring':False})
            total+=value.nbytes
    record={'arrays':arrays,'skipped':skipped,'total_array_bytes':total,
            'max_array_bytes':max_array_bytes,'max_total_bytes':max_total_bytes,
            'purpose':'private forensic snapshot only; no repair, renormalization, scoring or retry selection'}
    atomic(directory/'MANIFEST.json',record)
    return {'directory':str(directory),'manifest_sha256':sha(directory/'MANIFEST.json'),
            'arrays_preserved':len(arrays),'arrays_skipped':len(skipped),'bytes':total}
