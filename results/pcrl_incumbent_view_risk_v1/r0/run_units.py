"""Run registered units serially while refreshing the host receipt each time."""
from __future__ import annotations
import argparse
import json
from .compute_queue import Queue
from .host_occupancy import main as inventory

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=1)
    args=parser.parse_args()
    q=Queue()
    for _ in range(args.count):
        inventory()
        path=q.root/'host_occupancy.json'
        q.occupancy(path,json.loads(path.read_text())['conservative_external_heavy_reserved'])
        result=q.run_once()
        print(json.dumps(result,sort_keys=True),flush=True)
        if result.get('status')!='done': break

if __name__=='__main__':main()
