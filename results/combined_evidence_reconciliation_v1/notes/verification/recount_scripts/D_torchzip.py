"""Minimal numpy reader for torch zip-format .pt files containing plain tensors
(lists/tuples/dicts of tensors). No torch import. Used only to read stored
LEACE-on-raw erased feature tensors from the drive archive.
"""
from __future__ import annotations

import pickle
import zipfile

import numpy as np

_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
       "IntStorage": np.int32, "HalfStorage": np.float16, "BoolStorage": np.bool_,
       "ByteStorage": np.uint8}


class _Storage:
    def __init__(self, dtype, key):
        self.dtype, self.key = dtype, key


def load_pt(path: str):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    root = names[0].split("/")[0]
    cache = {}

    def storage(key, dtype):
        if key not in cache:
            cache[key] = np.frombuffer(zf.read(f"{root}/data/{key}"), dtype=dtype)
        return cache[key]

    def rebuild_tensor_v2(st, offset, size, stride, *args):
        flat = storage(st.key, st.dtype)
        if not size:
            return flat[offset].copy()
        itemsize = flat.itemsize
        return np.lib.stride_tricks.as_strided(
            flat[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild_tensor_v2
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if name in _DT:
                return name
            raise pickle.UnpicklingError(f"refusing {mod}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _loc, _n = pid
            dtype = _DT[stype] if isinstance(stype, str) else _DT[getattr(stype, "__name__", str(stype))]
            return _Storage(dtype, key)

    import io
    return U(io.BytesIO(zf.read(f"{root}/data.pkl"))).load()
