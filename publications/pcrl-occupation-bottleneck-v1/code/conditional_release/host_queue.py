"""Refresh a process-name-only inventory and run registered queue units."""
import os,json,datetime,argparse,psutil
from .scheduler import Queue,atomic

def inventory(q):
    rows=[]
    for p in psutil.process_iter(["pid","ppid","name","status","memory_info"]):
        try:
            d=p.info
            if d["pid"]==os.getpid() or d["status"]==psutil.STATUS_ZOMBIE: continue
            if d["name"].lower().startswith(("python","julia","rscript","matlab")):
                rows.append({"pid":d["pid"],"ppid":d["ppid"],"name":d["name"],"rss_bytes":d["memory_info"].rss})
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    evidence={"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"inventory_method":"psutil names/PIDs only, zombies excluded","numerical_process_candidates":rows,"external_heavy":min(2,len(rows)),"host_available_memory_bytes":psutil.virtual_memory().available}
    path=q.root/"host_occupancy.json";atomic(path,evidence)
    q.occupancy(path,evidence["external_heavy"])
    return evidence

def main():
    a=argparse.ArgumentParser();a.add_argument("--count",type=int,default=1);args=a.parse_args();q=Queue()
    for i in range(args.count):
        inventory(q);r=q.run_once();print(json.dumps(r),flush=True)
        if r["status"]!="done":break
if __name__=="__main__":main()
