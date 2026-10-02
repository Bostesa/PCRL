import json,subprocess,sys
ref=sys.argv[1]
def show(p):
    try: return json.loads(subprocess.check_output(['git','-C','/Users/nathansamson/PCRL','show',f'{ref}:{p}'],stderr=subprocess.DEVNULL))
    except Exception: return None
for d in sys.argv[2:]:
    ps=show(f'results/{d}/per_seed_results.json'); da=show(f'results/{d}/dominant_axis_audit.json')
    out=[d]
    if ps:
        rows=[(s['seed'],r['purpose'],r['attribute'],r['linear_r2']) for s in ps['per_seed'] for r in s['attribute_results']]
        out.append(f"per_seed: n={len(rows)} lin_r2<=.05:{sum(r[3]<=0.05 for r in rows)} mean={sum(r[3] for r in rows)/len(rows):.4f} seeds={sorted(set(r[0] for r in rows))} last_epochs={[s.get('last_epoch') for s in ps['per_seed']]} best_epochs={[s.get('best_epoch') for s in ps['per_seed']]}")
    if da:
        rr=[(k,r['purpose'],r['attribute'],r['r2_onehot']) for k,v in da['per_seed'].items() for r in v['rows']]
        out.append(f"DA onehot: n={len(rr)} <=.05:{sum(r[3]<=0.05 for r in rr)} mean={sum(r[3] for r in rr)/len(rr):.4f} epochs={[v.get('epoch') for v in da['per_seed'].values()]}")
    print(' | '.join(out))
