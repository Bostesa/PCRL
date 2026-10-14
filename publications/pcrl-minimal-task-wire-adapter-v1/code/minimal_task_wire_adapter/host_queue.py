"""Refresh name-only machine occupancy; one campaign worker at a time."""
import argparse,datetime,json,os,time
import psutil
from .scheduler import Queue,atomic

def inventory(q):
    rows=[];idle=[];candidates=[]
    for p in psutil.process_iter(['pid','ppid','name','status','memory_info','create_time']):
        try:
            d=p.info
            if d['pid']==os.getpid() or d['status']==psutil.STATUS_ZOMBIE:continue
            name=(d['name'] or '').lower()
            if name.startswith(('python','julia','rscript','matlab')) or 'virtualmachine' in name:
                row={'pid':d['pid'],'ppid':d['ppid'],'name':d['name'],'rss_bytes':d['memory_info'].rss,'created_epoch':d['create_time']}
                if 'virtualmachine' in name:rows.append(row)
                else:candidates.append((p,row,sum(p.cpu_times()[:2])))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if candidates:time.sleep(.25)
    for proc,row,before in candidates:
        try:
            after=sum(proc.cpu_times()[:2]);row['cpu_delta_s']=max(0,after-before)
            if row['rss_bytes']<64*1024**2 and row['cpu_delta_s']<=.001:idle.append(row)
            else:rows.append(row)
        except (psutil.NoSuchProcess,psutil.AccessDenied):rows.append(row)
    # The local VM reserves a slot. Tiny sleeping Python IPC processes are
    # recorded, not charged as fits; they are remeasured before every unit.
    evidence={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inventory_method':'names/PIDs only; existing VM reserved; numerical candidates sampled .25s, <64MiB and <=.001 CPU seconds marked idle','numerical_process_candidates':rows,'measured_idle_candidates':idle,'external_heavy':min(2,len(rows)),'host_available_memory_bytes':psutil.virtual_memory().available}
    p=q.root/'host_occupancy.json';atomic(p,evidence);q.occupancy(p,evidence['external_heavy']);return evidence

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=1);a=p.parse_args();q=Queue()
    for _ in range(a.count):
        inventory(q);result=q.run_once();print(json.dumps(result),flush=True)
        if result['status']!='done':break
if __name__=='__main__':main()
