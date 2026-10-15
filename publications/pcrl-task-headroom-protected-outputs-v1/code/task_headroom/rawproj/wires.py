"""Canonical literal releases, inherited byte-identically (contract, variants, arithmetic) from the separate-pool trial; queue guard rebound to this campaign.

Imports are metadata only. Numerical entrypoints require the active new queue.
Role forwards always use the same fixed batching, including unchanged anchors.
"""
from __future__ import annotations
import copy
import hashlib
import json

# The inherited contract and per-seed geometry bindings are byte-identical to
# the admitted corrected safeguard trial; this campaign never rebinds them.
CAMPAIGN = 'pcrl_compressed_income_safeguard_v1'
NEW_CAMPAIGN = 'pcrl_separate_pool_utility_guard_v1'
ROLE_FORWARD_BATCH_SIZE = 1024
WIRE_CONTRACT = {
    'schema': 'CIS1-literal-hybrid-wire-v1',
    'campaign': CAMPAIGN,
    'income': 'factorized_float64_z_times_B_transpose_times_B',
    'occupation': 'actual_learned_full_z16',
    'outputs': 'unchanged_own_head_recomputed_from_released_representation',
    'class_order': [[0, 1], list(range(6))],
    'widths': [21, 29, 50],
    'income_effective_feature_rank': 1,
    'decision_rule': 'argmax probabilities; first class in fixed source order',
    'role_forward_batch_size': ROLE_FORWARD_BATCH_SIZE,
    'extra_raw_z_or_q_release': False,
}
VARIANTS = {name: ('head_row_space', 'full') for name in
            ('RAW_HYBRID','SEPARATE_GUARD','GLOBAL_UPDATE_GUARD','OLD_BATCH_GUARD','ORDINARY','TASK_ONLY')}
VARIANTS.update({name: ('full','full') for name in ('RAW_FULL','U0_FULL')})
VARIANTS.update({name: ('head_row_space','head_row_space') for name in ('RAW_PROJ','U0_PROJ')})


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _queued():
    from task_headroom.scheduler import assert_active_from_env
    return assert_active_from_env()


def tensor_sha(value):
    """Byte identity includes dtype, shape, signed zero and all finite coordinates."""
    a=value.detach().cpu().contiguous().numpy()
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def head_sha(head):
    return _sha({'weight':tensor_sha(head.weight),'bias':tensor_sha(head.bias)})


def bind_geometry(model, *, source_seed, checkpoint_sha256):
    _queued()
    from .geometry import build_geometry, public_geometry
    if source_seed not in (0,1) or not isinstance(checkpoint_sha256,str) or len(checkpoint_sha256)!=64:
        raise PermissionError('admitted source seed and checkpoint SHA256 required')
    heads=[public_geometry(build_geometry(head.weight.detach().cpu().numpy(),head.bias.detach().cpu().numpy(),model.class_orders[i]))
           for i,head in enumerate(model.head)]
    binding={'schema':'CIS1-per-seed-geometry-v1','campaign':CAMPAIGN,'source_seed':source_seed,
             'checkpoint_sha256':checkpoint_sha256,'wire_contract_sha256':_sha(WIRE_CONTRACT),
             'heads':heads,'head_tensor_sha256':[head_sha(h) for h in model.head]}
    binding['binding_sha256']=_sha(binding)
    return binding


class PreparedGeometry:
    def __init__(self,binding):
        import torch
        self.binding=copy.deepcopy(binding)
        self.bases=[torch.tensor(g['B'],dtype=torch.float64).reshape(g['rank'],g['source_dimension']) for g in binding['heads']]
        self.basis_hashes=[tensor_sha(b) for b in self.bases]
    @property
    def binding_sha256(self):
        return self.binding['binding_sha256']


def prepare_geometry(binding, *, expected_sha256=None, source_seed=None, checkpoint_sha256=None):
    _queued()
    b=copy.deepcopy(binding); digest=b.pop('binding_sha256',None)
    if (b.get('schema')!='CIS1-per-seed-geometry-v1' or b.get('campaign')!=CAMPAIGN
        or b.get('wire_contract_sha256')!=_sha(WIRE_CONTRACT) or digest!=_sha(b)
        or (expected_sha256 is not None and digest!=expected_sha256)
        or (source_seed is not None and b.get('source_seed')!=source_seed)
        or (checkpoint_sha256 is not None and b.get('checkpoint_sha256')!=checkpoint_sha256)):
        raise PermissionError('geometry identity/hash differs from corrected trial')
    if len(b.get('heads',[]))!=2 or len(b.get('head_tensor_sha256',[]))!=2:
        raise PermissionError('both own-head geometry bindings required')
    b['binding_sha256']=digest
    return PreparedGeometry(b)


def _prepared(geometry):
    return geometry if isinstance(geometry,PreparedGeometry) else prepare_geometry(geometry)


def corrected_trial_wires(model,x,geometry,*,variant='RAW_HYBRID'):
    """Return literal own-wire dicts; no full-income fallback is available."""
    import torch
    from task_conditioned_representation.models import own_wire
    if variant not in VARIANTS:raise PermissionError('undeclared service wire variant')
    geometry=_prepared(geometry)
    if model.class_orders!=WIRE_CONTRACT['class_order']:raise PermissionError('own-head class order differs')
    wires=[]
    for i,rule in enumerate(VARIANTS[variant]):
        z=model.encode(i,x)
        if rule=='head_row_space':
            if head_sha(model.head[i])!=geometry.binding['head_tensor_sha256'][i]:
                raise PermissionError('projected service own head differs from bound basis head')
            basis=geometry.bases[i]
            if tensor_sha(basis)!=geometry.basis_hashes[i]:raise PermissionError('fixed projection basis changed')
            basis=basis.to(device=z.device)
            z=(z.to(torch.float64)@basis.T)@basis
        wires.append(own_wire(z,model.head[i],model.class_orders[i]))
    return wires


class FrozenIncomeGuard:
    """Verify frozen income and basis after each optimizer update and draw."""
    def __init__(self,model,geometry):
        self.model=model;self.geometry=_prepared(geometry)
        self.state={name:tensor_sha(value) for name,value in model.state_dict().items() if name.startswith(('enc.0.','head.0.'))}
        self.binding_sha=_sha(self.geometry.binding)
        self.basis_hashes=[tensor_sha(b) for b in self.geometry.bases]
    def verify(self):
        state=self.model.state_dict()
        if (any(tensor_sha(state[name])!=digest for name,digest in self.state.items())
            or _sha(self.geometry.binding)!=self.binding_sha
            or [tensor_sha(b) for b in self.geometry.bases]!=self.basis_hashes):
            raise AssertionError('frozen income tensor or immutable geometry bytes changed')
        return True


def forward_role(model,raw,names,geometry,*,variant='RAW_HYBRID'):
    """Raw83 forward with deterministic batching shared by every role alias.

    Same wire schema as UsefulModel.route_raw, with 'probability' singular.
    Full role arrays are label free and cannot nominate a checkpoint.
    """
    _queued()
    import numpy as np
    import torch
    if tuple(map(str,names))!=tuple(model.names):raise ValueError('wrong or reordered raw83 schema')
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=83 or not np.issubdtype(raw.dtype,np.floating) or not np.isfinite(raw).all():
        raise ValueError('finite raw83 matrix required')
    geometry=_prepared(geometry)
    collected=[{},{}]
    with torch.no_grad():
        for start in range(0,len(raw),ROLE_FORWARD_BATCH_SIZE):
            x=torch.from_numpy(np.ascontiguousarray(raw[start:start+ROLE_FORWARD_BATCH_SIZE],dtype=np.float32))
            wires=corrected_trial_wires(model,x,geometry,variant=variant)
            for i,w in enumerate(wires):
                for target,key in (('representation','representation'),('centered_logits','centered_logits'),('probability','probabilities'),('decision','decision'),('wire','full')):
                    collected[i].setdefault(target,[]).append(w[key].detach().cpu().numpy())
    if not len(raw):raise ValueError('nonempty role forward required')
    return [{key:np.concatenate(values,axis=0) for key,values in c.items()} for c in collected]


PROBE_SALT = 'SPG1|UPDATE-wire-probe|20261013|'


def probe_key(row_id):
    """Inherited label-independent probe order (same salt as the separate-pool trial)."""
    return hashlib.sha256((PROBE_SALT + str(row_id)).encode()).hexdigest()
