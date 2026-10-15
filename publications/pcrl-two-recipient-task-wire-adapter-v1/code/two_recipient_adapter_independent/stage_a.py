"""Stage A: zero-residual parity, frozen source/basis identity, attacked INNER wires vs stored releases, wire contract."""
from __future__ import annotations
import argparse
import hashlib
import json

from .common import bind, Source, bases, role_forward, npz, arr_sha, tsha, write_out, maxabs, WIDTHS, KS


def head_sha(w, b):
    v = {'weight': tsha(w), 'bias': tsha(b)}
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def contract(src, B, wires):
    import numpy as np
    import torch
    fails = []
    for i, w in enumerate(wires):
        K = KS[i]
        if w.shape[1] != WIDTHS[i]:
            fails.append(f'r{i+1} width {w.shape[1]}'); continue
        z = torch.from_numpy(w[:, :16].copy())
        off = z - (z @ B[i].T) @ B[i]
        if float(off.abs().max()) > 1e-12 * max(1.0, float(z.abs().max())):
            fails.append(f'r{i+1} rep outside row space ({float(off.abs().max()):.3e})')
        e = src.wire(i, z)
        if maxabs(e['logits'].numpy(), w[:, 16:16 + K]) > 1e-12:
            fails.append(f'r{i+1} logits not head(rep)')
        if maxabs(e['p'].numpy(), w[:, 16 + K:16 + 2 * K]) > 1e-12:
            fails.append(f'r{i+1} probabilities not softmax(head(rep))')
        if not np.array_equal(e['dec'].numpy().astype(np.float64), w[:, -1]):
            fails.append(f'r{i+1} decision not argmax')
        rank = int(np.linalg.matrix_rank(w[:, :16], tol=1e-9))
        if rank > B[i].shape[0]:
            fails.append(f'r{i+1} representation rank {rank} > basis rank')
    return fails


def run(spec):
    import numpy as np
    import torch
    active, cfg, get, guard = bind(spec)
    res = {'schema': 'TRA1-independent-stage-A-v1', 'unit_id': active['unit_id'], 'seeds': {}}
    tl = json.loads(get(cfg['training_complete_lock_item']).read_text())
    sel = json.loads(get(cfg['selection_lock_item']).read_text())
    ok_all = True
    for seed in ('0', '1'):
        guard()
        S = cfg['seeds'][seed]; rec = {'failures': []}
        src = Source(get(S['checkpoint']))
        B, geo = bases(get(S['geometry']))
        binding = json.loads(get(S['adapter_binding']).read_text())
        scales = [float(x) for x in binding['scales']]
        rec['scales'] = scales; rec['ranks'] = [int(b.shape[0]) for b in B]
        # source heads bound by geometry
        hs = [head_sha(*src.head[i]) for i in (0, 1)]
        rec['heads_match_geometry_binding'] = hs == geo['head_tensor_sha256']
        rec['basis_orthonormal_max_abs'] = [float((b @ b.T - torch.eye(len(b), dtype=torch.float64)).abs().max()) for b in B]
        rec['basis_sha_match_binding'] = [tsha(b) for b in B] == binding['basis_sha256']
        if not rec['heads_match_geometry_binding'] or not rec['basis_sha_match_binding']:
            rec['failures'].append('source head or basis identity differs')
        inner = npz(get(cfg['inner_features'])); fit2 = npz(get(cfg['fit2_features']))
        Xs = {'INNER': inner, 'FIT2': fit2}
        rawp = {k: role_forward(src, B, v['X'], v['feature_names'], mode='raw_proj') for k, v in Xs.items()}
        zero = {k: role_forward(src, B, v['X'], v['feature_names'], states=None, scales=scales, mode='adapter') for k, v in Xs.items()}
        rec['zero_state_formula_bitwise_raw_proj'] = {k: [bool(np.array_equal(rawp[k][i], zero[k][i])) for i in (0, 1)] for k in Xs}
        # every checkpoint: identity; epoch-0 parity
        ck = {}
        nck = 0
        for arm, labels in cfg['checkpoint_paths'][seed].items():
            for label, item in labels.items():
                guard()
                if item['sha256'] != tl['checkpoints'][seed][arm][label]['sha256']:
                    rec['failures'].append(f'{arm} {label} sha differs from TRAINING_COMPLETE_LOCK')
                p = torch.load(get(item), map_location='cpu', weights_only=False)
                nck += 1
                idok = (p['source_checkpoint_sha256'] == S['checkpoint']['sha256'] and p['seed'] == int(seed) and p['arm'] == arm
                        and p['label'] == label and all(torch.equal(p['bases'][i], B[i]) for i in (0, 1))
                        and [float(x) for x in p['scales']] == scales
                        and [tsha(p['states'][i]['R']) for i in (0, 1)] == tl['checkpoints'][seed][arm][label]['R_sha256']
                        and [tsha(p['states'][i]['b']) for i in (0, 1)] == tl['checkpoints'][seed][arm][label]['b_sha256'])
                if not idok:
                    rec['failures'].append(f'{arm} {label} identity/bases/scales/state sha differ')
                ck[(arm, label)] = p['states']
                if label in ('E0', 'P1E0'):
                    zs = all(bool(torch.all(p['states'][i]['R'] == 0) and torch.all(p['states'][i]['b'] == 0)) for i in (0, 1))
                    par = {k: [bool(np.array_equal(role_forward(src, B, v['X'], v['feature_names'], states=p['states'], scales=scales)[i], rawp[k][i])) for i in (0, 1)] for k, v in Xs.items()}
                    rec.setdefault('epoch0_parity', {})[f'{arm}|{label}'] = {'states_exactly_zero': zs, 'bitwise_raw_proj': par}
                    if not zs or not all(all(v) for v in par.values()):
                        rec['failures'].append(f'{arm} {label} zero-residual parity fails')
        rec['checkpoints_checked'] = nck
        # attacked services vs stored INNER releases
        svc_rec = {}
        attacked = {a: sel['selection'][a]['attacked_label'] for a in sel['selection']}
        for name, svc in cfg['services'][seed].items():
            guard()
            if svc['kind'] == 'RAW_PROJ':
                wires = rawp['INNER']
            else:
                states = [None, None]
                for i, n in enumerate(('income', 'occupation')):
                    s = svc.get(n)
                    if s is None:
                        continue
                    if s['source'] == 'TRA1':
                        st = ck[(s['arm'], s['label'])][i]
                        states[i] = {'R': st['R'], 'b': st['b']}
                        if s['arm'] in attacked and s['label'] != attacked[s['arm']]:
                            rec['failures'].append(f'{name} uses {s["label"]} but SELECTION_LOCK attacks {attacked[s["arm"]]}')
                    else:
                        pp = torch.load(get(s['checkpoint']), map_location='cpu', weights_only=False)
                        if not (pp.get('arm') == 'J_TRUST' and pp.get('epoch') == 10 and pp.get('seed') == int(seed)
                                and pp.get('source_checkpoint_sha256') == S['checkpoint']['sha256'] and float(pp['scale']) == scales[1] and i == 1):
                            rec['failures'].append('O_PREVIOUS identity differs')
                        states[i] = {'R': pp['R'], 'b': pp['b_res']}
                wires = role_forward(src, B, inner['X'], inner['feature_names'], states=states, scales=scales)
            bank = cfg['banks'][seed][name]
            rel = npz(get(bank['release']))
            receipt = json.loads(get(bank['receipt']).read_text())
            same = [bool(np.array_equal(wires[i], rel[f'r{i+1}'])) for i in (0, 1)]
            order = bool(np.array_equal(rel['row_id'], inner['row_id']) and np.array_equal(rel['unit'], inner['unit']))
            hashes = [arr_sha(wires[0]) == receipt['wire_hashes']['INNER_r1'], arr_sha(wires[1]) == receipt['wire_hashes']['INNER_r2']]
            cf = contract(src, B, wires)
            dup = True
            for w in wires:
                for g in np.unique(inner['unit']):
                    ix = np.flatnonzero(inner['unit'] == g)
                    if len(ix) > 1 and not all(np.array_equal(w[ix[0]], w[j]) for j in ix[1:]):
                        dup = False
            change = [maxabs(wires[i], rawp['INNER'][i]) for i in (0, 1)]
            svc_rec[name] = {'bitwise_equal_stored_INNER_RELEASE': same, 'row_order_equal': order, 'receipt_wire_hash_match': hashes,
                             'contract_failures': cf, 'duplicate_groups_identical': dup, 'max_abs_change_vs_RAW_PROJ': change,
                             'widths': [int(w.shape[1]) for w in wires]}
            if not (all(same) and order and all(hashes) and not cf and dup):
                rec['failures'].append(f'service {name} wire check fails')
        # controls bank release must be RAW_PROJ
        relc = npz(get(cfg['banks'][seed]['CONTROLS']['release']))
        rec['controls_release_is_raw_proj'] = [bool(np.array_equal(relc[f'r{i+1}'], rawp['INNER'][i])) for i in (0, 1)]
        if not all(rec['controls_release_is_raw_proj']):
            rec['failures'].append('controls release differs from RAW_PROJ')
        rec['services'] = svc_rec
        rec['n_services'] = len(svc_rec)
        rec['pass'] = not rec['failures']
        ok_all &= rec['pass']
        res['seeds'][seed] = rec
    res['pass'] = bool(ok_all)
    return write_out(cfg, res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    print(json.dumps(run(ap.parse_args().spec)['pass']))
