"""Hash-bound private selected-policy serialization and deployment loading."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .r2_space import Space
from .r2_policy import Policy

SPACE_KEYS=("probability1","probability2","actions1","actions2","valid1","valid2")


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def write_json(path,value,private=True):
    mode=0o600 if private else 0o644
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode),"w") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return sha(path)


def write_npz(path,arrays):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    return sha(path)


def read_exact(path,keys):
    with ZipFile(path) as z:
        if len(z.infolist())!=len(keys) or {i.filename for i in z.infolist()}!={k+".npy" for k in keys}:
            raise PermissionError("private archive member allowlist differs")
    with np.load(path,allow_pickle=False) as a:return {k:np.array(a[k],copy=True) for k in keys}


def from_bank(bank):
    values=[]
    probs=[];actions=[];valid=[]
    for r in (1,2):
        item=bank["recipients"][str(r)];p=np.asarray(item["probability"],np.float64)
        a=np.repeat(np.arange(len(p),dtype=np.int64)[:,None],4,axis=1);v=np.zeros(a.shape,bool)
        for i,row in enumerate(item["support"]):
            a[i,:len(row)]=row;v[i,:len(row)]=True
        probs.append(p);actions.append(a);valid.append(v)
    return Space(*probs,*actions,*valid)


def save_policy(directory,policy,*,arm_id,source_seed,source_unit_id,bank_sha256,source_binding_sha256,lock_sha256,selection,crossfit_manifest_sha256,exact_p0_alias):
    directory=Path(directory)
    if directory.exists() or ".private" not in directory.parts:raise PermissionError("new private policy directory required")
    directory.mkdir(parents=True,mode=0o700)
    arrays={key:getattr(policy.space,key) for key in SPACE_KEYS}
    for i,w in enumerate(policy.weights):
        for j,a in enumerate(w):arrays[f"gate{i}_{j}"]=a
    package=directory/"policy.npz";package_sha=write_npz(package,arrays)
    receipt={"schema":"IVR1-selected-policy-v1","arm_id":arm_id,"source_seed":source_seed,"source_unit_id":source_unit_id,
             "policy_arm":policy.arm,"law_kind":policy.law_kind,"eta1":policy.eta1,"eta2":policy.eta2,
             "parameter_count":policy.parameter_count,"gate_count":len(policy.weights),
             "policy_package_sha256":package_sha,"public_bank_sha256":bank_sha256,
             "source_binding_sha256":source_binding_sha256,"source_lock_sha256":lock_sha256,
             "selection":selection,"crossfit_manifest_sha256":crossfit_manifest_sha256,
             "exact_p0_alias":bool(exact_p0_alias),"array_sha256":{key:array_sha(a) for key,a in arrays.items()}}
    receipt_sha=write_json(directory/"POLICY_RECEIPT.json",receipt)
    return receipt,receipt_sha


def load_policy(directory,*,expected_receipt_sha256):
    directory=Path(directory);receipt_path=directory/"POLICY_RECEIPT.json"
    if sha(receipt_path)!=expected_receipt_sha256:raise PermissionError("selected-policy receipt hash differs")
    receipt=json.loads(receipt_path.read_text())
    if receipt["schema"]!="IVR1-selected-policy-v1" or receipt["source_seed"] not in (0,1):raise PermissionError("policy receipt schema/seed differs")
    package=directory/"policy.npz"
    if sha(package)!=receipt["policy_package_sha256"]:raise PermissionError("selected-policy package hash differs")
    arrays=read_exact(package,set(receipt["array_sha256"]))
    if any(array_sha(a)!=receipt["array_sha256"][k] for k,a in arrays.items()):raise PermissionError("policy array binding differs")
    space=Space(*(arrays[k] for k in SPACE_KEYS))
    weights=tuple(tuple(arrays[f"gate{i}_{j}"] for j in range(4)) for i in range(receipt["gate_count"]))
    policy=Policy(space,receipt["policy_arm"],weights,receipt["eta1"],receipt["eta2"])
    if policy.parameter_count!=receipt["parameter_count"] or policy.law_kind!=receipt["law_kind"]:raise PermissionError("policy deployment family differs")
    return policy
