"""Stage R: the registered representation screen, both families, k=64 then the 256 fallback."""
import json
import os
import pickle
import sys
import time
from pathlib import Path

WT = Path('/Users/nathansamson/PCRL-terminal-1-replacement')
sys.path.insert(0, str(WT))
os.environ.setdefault('PCRL_NLR_FALLBACK', '/Users/nathansamson/PCRL-terminal-1-stochastic')

from experiments.pcrl_stochastic_replacement_overnight_v1 import ledger as L      # noqa: E402
from experiments.pcrl_stochastic_replacement_overnight_v1 import replacement as R  # noqa: E402

RES = WT / 'results' / 'pcrl_stochastic_replacement_overnight_v1'
SEEDS = (0, 1, 2)
SCRATCH = Path('/private/tmp/claude-501/-Users-nathansamson-PCRL/'
               '96f5d9cc-f85d-4834-a6ad-b988bf8c3108/scratchpad')


def main():
    led = L.Ledger(RES / 'ledger')
    with L.RunLock(RES / 'run.lock', 'stage_R'):
        t0 = time.perf_counter()
        cache = {}
        for s in SEEDS:
            cache[s] = R.seed_inputs(s)
            print(f'inputs seed {s} ready', flush=True)

        out, paired = {}, {}
        for family in R.FAMILIES:
            out[family] = {}
            for k in (R.K_PRIMARY, R.K_FALLBACK):
                cfg = {'stage': 'R', 'family': family, 'k': k, 'anchors': list(SEEDS),
                       'slate': R.probe_schedule(), 'rows': 'all_eligible'}
                cid = L.config_id(cfg)
                if not led.claim(cid, cfg):
                    print(f'{family} k={k} reused {cid}', flush=True)
                    out[family][k] = L.read_json(RES / 'stage_R' / f'{family}_k{k}.json')
                    continue
                tick = time.perf_counter()
                rec = R.screen_family(SEEDS, family, k, cache)
                paired[(family, k)] = rec.pop('_per_anchor_paired')
                L.write_json_atomic(RES / 'stage_R' / f'{family}_k{k}.json', rec)
                led.put(cid, cfg, 'fitted', seconds=time.perf_counter() - tick,
                        passed=rec['pass'])
                out[family][k] = rec
                print(f"{family} k={k} pass={rec['pass']} "
                      f"deficit_unw={rec['mean_deficit_T_vs_J']['unweighted']:+.5f} "
                      f"deficit_pw={rec['mean_deficit_T_vs_J']['person_weighted']:+.5f} "
                      f"anchors_gain={rec['anchors_with_gain_both_weightings']} "
                      f"({time.perf_counter()-tick:.0f}s)", flush=True)
                if rec['pass']:
                    break          # first passing resolution per family; failures retained

        with open(SCRATCH / 'stage_r_paired.pkl', 'wb') as fh:
            pickle.dump(paired, fh)
        L.write_json_atomic(RES / 'stage_R' / 'SUMMARY.json',
                            {'families': out, 'seconds': time.perf_counter() - t0,
                             'seeds': list(SEEDS)})
        print(f'\nstage R done in {time.perf_counter()-t0:.0f}s', flush=True)
        for family, ks in out.items():
            for k, rec in ks.items():
                if rec:
                    print(f"  {family:6s} k={k:3d} pass={rec['pass']}", flush=True)


if __name__ == '__main__':
    main()
