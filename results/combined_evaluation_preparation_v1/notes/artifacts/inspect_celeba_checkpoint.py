#!/usr/bin/env python3
"""Metadata-only inspection of a PCRL checkpoint (.pt), CPU only, no forward pass.

Written for the PENDING inspection of the durable-guarantees CelebA encoder
    fl-PCRL-main-checkpoints.tar::checkpoints/celeba_v2/final.pt
    sha256 b0df3fb7fb3f5fa36e7a847d1fd2d16481f08389e76acdd81300648be4fd08fb, 35,605,887 bytes
but generic: it was tested on the PCRL v2 Round-4 tabular checkpoints.

Loading policy (never executes code embedded in the pickle):
  1. torch.load(path, map_location="cpu", weights_only=True).
  2. If that refuses (non-allowlisted globals, e.g. numpy scalars or custom classes
     inside "history"/"config"), fall back to a restricted unpickler that reads the
     torch zip container directly. Its find_class allowlists only tensor-rebuild
     helpers, storage types, OrderedDict and a few builtins; every other global is
     replaced by an inert RecordingStub that records the (module, name), its
     constructor args and its state. Tensor storages are never materialised: tensors
     become TensorInfo(shape, dtype) records.

Output: a JSON report (stdout, or --out FILE) with
  file size/sha256 (optional --expect-sha256 check), loader used, top-level keys and
  types, embedded config/args (non-tensor scalars only), state/epoch, tensor names +
  shapes + dtypes per state dict and parameter counts, architecture fingerprint
  (FiLM CNNEncoder / CNNPurposeProjectionEncoder / v2 LoRA backbone ...), purpose /
  task / attribute names found in keys or strings, normalisation / preprocessing
  fields, split / seed / row-index fields (integer arrays are reported by shape only,
  never by value), history key names with lengths, and non-allowlisted globals seen.

Usage:
  python inspect_celeba_checkpoint.py CKPT [--expect-sha256 HEX] [--out report.json]
         [--force-restricted] [--max-tensors N]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import pickle
import re
import sys
import zipfile

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
    "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]
CELEBA_PURPOSES = ["smile_detection", "age_estimation", "expression_analysis",
                   "attractiveness_prediction", "gender_analysis"]
NORM_PAT = re.compile(r"(mean|std|normali[sz]|transform|resize|image_size|img_size|"
                      r"crop|flip|pixel|scale)", re.I)
SPLIT_PAT = re.compile(r"(split|partition|train_idx|val_idx|test_idx|indices|index|"
                       r"row_id|rows|n_train|n_val|n_test|max_train|max_val|max_test|"
                       r"seed|subsample|holdout|fold)", re.I)
CONFIG_KEYS = ("config", "cfg", "args", "hparams", "hyper_parameters", "settings",
               "trainer_config", "run_config", "meta", "metadata", "state")


# --------------------------------------------------------------------------- #
# Inert records                                                                #
# --------------------------------------------------------------------------- #
class TensorInfo:
    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = str(dtype)

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n


class StorageInfo:
    __slots__ = ("dtype", "key", "numel")

    def __init__(self, dtype, key, numel):
        self.dtype, self.key, self.numel = dtype, key, numel


_SEEN_STUBS: collections.Counter = collections.Counter()


class RecordingStub:
    """Stand-in for any non-allowlisted global. Records, never executes."""
    _qualname = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__dict__["_stub_args"] = _plain(args)
        obj.__dict__["_stub_state"] = None
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_stub_state"] = _plain(state)

    def __setitem__(self, k, v):  # for SETITEMS on dict subclasses
        self.__dict__.setdefault("_stub_items", {})[str(k)] = _plain(v)

    def append(self, v):  # for APPENDS on list subclasses
        self.__dict__.setdefault("_stub_list", []).append(_plain(v))

    def extend(self, vs):
        for v in vs:
            self.append(v)

    def describe(self):
        return {"__stub__": self._qualname, "args": self.__dict__.get("_stub_args"),
                "state": self.__dict__.get("_stub_state"),
                "items": self.__dict__.get("_stub_items"),
                "list_len": len(self.__dict__.get("_stub_list", []))}


def _make_stub(module, name):
    q = f"{module}.{name}"
    _SEEN_STUBS[q] += 1
    return type(f"Stub_{name}", (RecordingStub,), {"_qualname": q})


def _plain(x, depth=0):
    """Reduce to JSON-friendly scalars without touching tensor contents."""
    if depth > 4:
        return "<deep>"
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, TensorInfo):
        return {"tensor": list(x.shape), "dtype": x.dtype}
    if isinstance(x, RecordingStub):
        return {"__stub__": x._qualname}
    if isinstance(x, (list, tuple)):
        return [_plain(v, depth + 1) for v in list(x)[:20]]
    if isinstance(x, dict):
        return {str(k): _plain(v, depth + 1) for k, v in list(x.items())[:50]}
    return f"<{type(x).__name__}>"


# --------------------------------------------------------------------------- #
# Restricted unpickler over the torch zip container                            #
# --------------------------------------------------------------------------- #
def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    return TensorInfo(size, storage.dtype if isinstance(storage, StorageInfo) else "?")


def _rebuild_parameter(data, requires_grad=None, backward_hooks=None, *rest):
    return data


def _rebuild_from_type_v2(func, new_type, args, state):
    return func(*args) if callable(func) else TensorInfo((), "?")


def _odict(*args):
    return collections.OrderedDict(*args)


_STORAGE_DTYPES = {
    "FloatStorage": "float32", "DoubleStorage": "float64", "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
    "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8",
    "BoolStorage": "bool", "ComplexFloatStorage": "complex64",
    "ComplexDoubleStorage": "complex128", "UntypedStorage": "uint8",
}
_SAFE_BUILTINS = {("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
                  ("builtins", "slice"): slice, ("builtins", "complex"): complex}


class _DType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self._zf, self._prefix = zf, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor",
                                                 "_rebuild_tensor_v3"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch._utils" and name == "_rebuild_parameter_with_state":
            return _rebuild_parameter
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return _rebuild_from_type_v2
        if module == "torch" and name.endswith("Storage"):
            return _DType(_STORAGE_DTYPES.get(name, name))
        if module == "torch" and name in ("float32", "float64", "float16", "bfloat16",
                                          "int64", "int32", "int16", "int8", "uint8",
                                          "bool"):
            return _DType(name)
        if module == "collections" and name == "OrderedDict":
            return _odict
        if (module, name) in _SAFE_BUILTINS:
            return _SAFE_BUILTINS[(module, name)]
        return _make_stub(module, name)

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            stype, key = pid[1], pid[2]
            numel = pid[4] if len(pid) > 4 else None
            return StorageInfo(str(stype), key, numel)
        return StorageInfo("?", str(pid), None)


def restricted_load(path):
    if not zipfile.is_zipfile(path):
        raise RuntimeError("legacy (non-zip) torch format: restricted loader not "
                           "implemented; inspect with weights_only=True only")
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl:
            raise RuntimeError("no data.pkl in archive")
        prefix = pkl[0][: -len("data.pkl")]
        data = zf.read(pkl[0])
        return RestrictedUnpickler(io.BytesIO(data), zf, prefix).load()


# --------------------------------------------------------------------------- #
# Summaries                                                                     #
# --------------------------------------------------------------------------- #
def _is_tensor(v):
    if isinstance(v, TensorInfo):
        return True
    try:
        import torch
        return isinstance(v, torch.Tensor)
    except Exception:  # pragma: no cover
        return False


def _tinfo(v):
    if isinstance(v, TensorInfo):
        return list(v.shape), v.dtype, v.numel()
    return list(v.shape), str(v.dtype).replace("torch.", ""), int(v.numel())


def _scalarise(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, RecordingStub):
        return v.describe()
    if _is_tensor(v):
        shape, dtype, n = _tinfo(v)
        if n == 1:
            try:
                return float(v.item())  # real tensor scalar
            except Exception:
                return {"tensor": shape, "dtype": dtype}
        return {"tensor": shape, "dtype": dtype}
    if isinstance(v, (list, tuple)):
        if len(v) > 12 and all(isinstance(x, (int, float)) for x in v):
            return {"list_len": len(v), "first": v[0], "last": v[-1]}
        return [_scalarise(x) for x in list(v)[:12]] + (["..."] if len(v) > 12 else [])
    if isinstance(v, dict):
        return {str(k): _scalarise(x) for k, x in v.items()}
    if hasattr(v, "dtype") and hasattr(v, "shape"):  # numpy array
        return {"ndarray": list(v.shape), "dtype": str(v.dtype)}
    return f"<{type(v).__name__}>"


def _walk_tensors(obj, prefix, out, depth=0):
    if depth > 6:
        return
    if _is_tensor(obj):
        out.append((prefix, *_tinfo(obj)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_tensors(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_tensors(v, f"{prefix}[{i}]", out, depth + 1)


def _walk_strings(obj, out, depth=0):
    if depth > 6:
        return
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _walk_strings(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_strings(v, out, depth + 1)
    elif isinstance(obj, RecordingStub):
        _walk_strings(obj.__dict__.get("_stub_args"), out, depth + 1)
        _walk_strings(obj.__dict__.get("_stub_state"), out, depth + 1)


def _walk_matching_keys(obj, pat, prefix, out, depth=0):
    if depth > 6 or not isinstance(obj, dict):
        return
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if pat.search(str(k)):
            out[path] = _scalarise(v)
        if isinstance(v, dict) and not all(_is_tensor(x) for x in v.values()):
            _walk_matching_keys(v, pat, path, out, depth + 1)


def fingerprint(tensors):
    names = [t[0] for t in tensors]
    joined = "\n".join(names)
    fp = []
    if re.search(r"film_gamma", joined):
        fp.append("FiLM-conditioned CNNEncoder (pcrl.models.cnn_encoder.CNNEncoder)")
    if re.search(r"purpose_projections\.\d", joined):
        fp.append("CNNPurposeProjectionEncoder (per-purpose MLP projections)")
    if re.search(r"purpose_embedding", joined):
        fp.append("learned purpose embedding present")
    if re.search(r"lora_adapters|\.A\.weight", joined):
        fp.append("v2 LoRA adapters (pcrl v2 trainer)")
    if re.search(r"conv_layers\.\d", joined):
        fp.append("conv backbone")
    dims = {}
    for n, shape, dtype, _ in tensors:
        if re.search(r"(repr_proj(\.\d+)?|purpose_projections\.\d+\.\d+)\.weight$", n) \
                and len(shape) == 2:
            dims[n] = shape
        if re.search(r"purpose_embedding\.weight$", n):
            dims[n] = shape
        if re.search(r"film_gamma\.\d+\.weight$", n):
            dims[n] = shape
    return fp, dims


def int_array_candidates(tensors):
    """Integer tensors with >1000 elements: possible row-index arrays (shape only)."""
    return [{"name": n, "shape": s, "dtype": d} for n, s, d, k in tensors
            if ("int" in d and "num_batches_tracked" not in n and k > 1000)]


def summarise(ck, loader, max_tensors):
    rep = {"loader": loader}
    if not isinstance(ck, dict):
        rep["top_level_type"] = type(ck).__name__
        ck = {"<root>": ck}
    rep["top_level_keys"] = {str(k): type(v).__name__ for k, v in ck.items()}

    tensors = []
    _walk_tensors(ck, "", tensors)
    by_top = collections.OrderedDict()
    for n, s, d, k in tensors:
        top = n.split(".")[0]
        e = by_top.setdefault(top, {"n_tensors": 0, "numel": 0, "dtypes": set()})
        e["n_tensors"] += 1
        e["numel"] += k
        e["dtypes"].add(d)
    rep["tensor_groups"] = {k: {"n_tensors": v["n_tensors"], "numel": v["numel"],
                                "dtypes": sorted(v["dtypes"])} for k, v in by_top.items()}
    rep["tensor_index"] = [{"name": n, "shape": s, "dtype": d}
                           for n, s, d, _ in tensors[:max_tensors]]
    rep["tensor_index_truncated"] = len(tensors) > max_tensors
    fp, dims = fingerprint(tensors)
    rep["architecture_fingerprint"] = fp
    rep["key_projection_shapes"] = dims

    # embedded config / args / state (non-tensor values)
    emb = {}
    for k, v in ck.items():
        if str(k).lower() in CONFIG_KEYS or any(c in str(k).lower() for c in ("config", "args")):
            emb[str(k)] = _scalarise(v)
    rep["embedded_config"] = emb
    st = ck.get("state") if isinstance(ck.get("state"), dict) else {}
    rep["epoch"] = _scalarise(st.get("epoch")) if st else _scalarise(ck.get("epoch"))

    # history: names and lengths only (plus nested key names)
    hist = ck.get("history")
    if isinstance(hist, dict):
        h = {}
        for k, v in hist.items():
            if isinstance(v, list):
                h[str(k)] = {"len": len(v),
                             "elem_type": type(v[0]).__name__ if v else None,
                             "elem_keys": (sorted(map(str, v[0].keys()))[:40]
                                           if v and isinstance(v[0], dict) else None)}
            else:
                h[str(k)] = type(v).__name__
        rep["history_keys"] = h

    strings = set()
    _walk_strings({k: v for k, v in ck.items()}, strings)
    names_blob = "\n".join(strings) + "\n" + "\n".join(t[0] for t in tensors)
    rep["celeba_attribute_names_found"] = sorted(a for a in CELEBA_ATTRS
                                                 if re.search(rf"(?<![A-Za-z]){a}(?![A-Za-z])",
                                                              names_blob))
    rep["celeba_purpose_names_found"] = sorted(p for p in CELEBA_PURPOSES if p in names_blob)
    # generic purpose/attr pairs as written by the v2 trainer ("purpose__attr")
    pairs = sorted({s for s in strings if re.fullmatch(r"[A-Za-z0-9_]+__[A-Za-z0-9_]+", s)})
    rep["purpose__attribute_pairs"] = pairs[:200]
    heads = sorted({n.split(".")[1] for n, *_ in tensors
                    if n.startswith("task_heads.") and len(n.split(".")) > 2})
    rep["task_head_names"] = heads

    norm, split = {}, {}
    _walk_matching_keys(ck, NORM_PAT, "", norm)
    _walk_matching_keys(ck, SPLIT_PAT, "", split)
    rep["normalisation_or_preprocessing_fields"] = norm
    rep["split_seed_or_row_fields"] = split
    rep["integer_array_candidates_for_row_indices"] = int_array_candidates(tensors)
    rep["non_allowlisted_globals"] = dict(_SEEN_STUBS)
    return rep


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force-restricted", action="store_true",
                    help="skip torch.load and use the restricted unpickler only")
    ap.add_argument("--max-tensors", type=int, default=400)
    a = ap.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    rep = {"file": os.path.basename(a.ckpt), "size_bytes": os.path.getsize(a.ckpt)}
    rep["sha256"] = sha256_file(a.ckpt)
    if a.expect_sha256:
        ok = rep["sha256"].lower().startswith(a.expect_sha256.lower())
        rep["sha256_matches_expected"] = ok
        if not ok:
            print(json.dumps(rep, indent=2))
            print("ERROR: sha256 mismatch; refusing to inspect", file=sys.stderr)
            return 2

    ck, loader, err = None, None, None
    if not a.force_restricted:
        try:
            import torch
            ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
            loader = f"torch.load(weights_only=True) torch=={torch.__version__}"
        except Exception as e:  # weights_only refused, or torch missing
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
    if ck is None:
        ck = restricted_load(a.ckpt)
        loader = "restricted_unpickler (no code execution; tensors as shape records)"
    rep["weights_only_error"] = err
    rep.update(summarise(ck, loader, a.max_tensors))

    text = json.dumps(rep, indent=2, default=str)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
