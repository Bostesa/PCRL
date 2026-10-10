"""Label-free cached actual-wire releases on admitted role-only P0 inputs."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .r2_package import write_json,write_npz,array_sha
from .r1_bank import wires_for_codes,wire_hash
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import offline_uniform
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import offline_factor_uniforms


def materialize(directory,policy,source,bank,*,arm_id,source_seed,source_unit_id,role,realization,private_root,policy_receipt_sha256):
    if role not in ("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION") or realization not in (0,1,2):raise ValueError("release role/draw differs")
    row=np.asarray(source["row_id"]);unit=np.asarray(source["unit"])
    c1=np.asarray(source["code_r1"]);c2=np.asarray(source["code_r2"])
    n=len(row)
    if any(a.dtype!=np.int64 or a.shape!=(n,) for a in (row,unit,c1,c2)):raise ValueError("role P0 codes/people not aligned int64")
    if len(np.unique(row))!=n:raise ValueError("row IDs repeated")
    for r,c in ((1,c1),(2,c2)):
        if not np.array_equal(source[f"prob_r{r}"],getattr(policy.space,f"probability{r}")[c]) or not np.array_equal(source[f"decision_r{r}"],getattr(policy.space,f"probability{r}")[c].argmax(1)):
            raise ValueError("P0 source code/probability/decision binding differs")
    group={}
    for i,u in enumerate(unit):
        pair=(int(c1[i]),int(c2[i]))
        if int(u) in group and group[int(u)]!=pair:raise ValueError("duplicate provider group input differs")
        group[int(u)]=pair
    p=policy.space.prepare(c1,c2);q=policy.law_prepared(p)
    law={"row_id":row,"unit":unit,**{k:p[k] for k in ("code_r1","code_r2","valid","b0_code_r1","b0_code_r2")},"mass":q}
    if policy.arm in ("P0","V_D"):
        coins=None
    elif policy.arm in ("V_P","V_L","V_S12","V_S21"):
        coins=np.asarray([offline_factor_uniforms(private_root,source_seed=source_seed,role=role,realization=realization,unit=int(u)) for u in unit],np.float64)
    else:
        coins=np.asarray([offline_uniform(private_root,source_seed=source_seed,role=role,realization=realization,unit=int(u)) for u in unit],np.float64)
    a,b=policy.sample_codes(c1,c2,coins)
    positive=((p["code_r1"]==a[:,None])&(p["code_r2"]==b[:,None])&(q>0)).any(1)
    if not positive.all():raise ValueError("sampled code outside positive actual law")
    sampled={"row_id":row,"unit":unit,"code_r1":a,"code_r2":b}
    directory=Path(directory)
    if directory.exists() or ".private" not in directory.parts:raise PermissionError("new private release custody required")
    directory.mkdir(parents=True,mode=0o700)
    law_sha=write_npz(directory/"law.npz",law);codes_sha=write_npz(directory/"release_codes.npz",sampled)
    content=hashlib.sha256("".join(k+array_sha(law[k]) for k in sorted(law)).encode()).hexdigest()
    receipt={"schema":"IVR1-selected-role-release-v1","arm_id":arm_id,"source_seed":source_seed,"source_unit_id":source_unit_id,"role":role,"realization":realization,
             "rows":n,"groups":len(group),"policy_receipt_sha256":policy_receipt_sha256,
             "law_file_sha256":law_sha,"law_content_sha256":content,"codes_file_sha256":codes_sha,
             "row_unit_pair_sha256":array_sha(np.c_[row,unit]),"source_binding_sha256":bank["source_binding_sha256"],
             "wire_sha256_by_recipient":{"r1":wire_hash(wires_for_codes(bank,1,a)),"r2":wire_hash(wires_for_codes(bank,2,b))},
             "private_stream_root_sha256":hashlib.sha256(private_root).hexdigest(),
             "sampling_contract":"cached_single_release_per_group; three separate realizations; no repeated query averaging",
             "draw_kind":"none" if coins is None else ("independent_domain_two_coins" if coins.ndim==2 else "one_joint_coin")}
    return receipt,write_json(directory/"RELEASE_RECEIPT.json",receipt)
