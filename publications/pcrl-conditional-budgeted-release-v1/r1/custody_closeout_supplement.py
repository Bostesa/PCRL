"""Versioned metadata-only closeout copies under the global custody queue."""
from pathlib import Path
import argparse,json
from conditional_release.scheduler import assert_active_from_env
from conditional_release.custody import uncached_hash,copy_pinned,safe_alias
from conditional_release.source import write_once

def run(config):
    active=assert_active_from_env(stage="custody");c=json.loads(Path(config).read_text())
    if c["unit_id"]!=active["unit_id"]:raise PermissionError("unit differs")
    target=Path(c["target"])
    if target.exists():raise FileExistsError("new versioned supplement required")
    if c.get("require_separate_filesystem"):
        ancestor=target.parent
        while not ancestor.exists():ancestor=ancestor.parent
        if Path(c["source_root"]).stat().st_dev==ancestor.stat().st_dev:raise PermissionError("separate filesystem required")
    target.mkdir(parents=True,mode=0o700);entries=[]
    for e in c["entries"]:
        source=Path(e["path"]);alias=safe_alias(e["alias"])
        if source.suffix not in {".json",".py",".md",".txt",".log"}:raise PermissionError("person-data recapture refused")
        copy_pinned(source,target/alias,e["sha256"],e["size_bytes"])
        entries.append({"alias":str(alias),"sha256":uncached_hash(target/alias),"size_bytes":e["size_bytes"]})
    receipt={"schema":"CBR1-metadata-closeout-supplement-v1","status":"PASS","unit_id":active["unit_id"],"scope":"Metadata and restore helpers only; no person data or scientific source recapture.","scientific_snapshot_commit":c["scientific_snapshot_commit"],"scientific_plan_sha256":c["scientific_plan_sha256"],"separate_filesystem_verified":bool(c.get("require_separate_filesystem")),"uncached_full_readback":True,"file_count":len(entries),"bytes":sum(e["size_bytes"] for e in entries),"entries":entries}
    write_once(target/"SUPPLEMENT_RECEIPT.json",receipt)
    write_once(c["public_output"],receipt,False)
    print(json.dumps({"status":"PASS","metadata_files":len(entries)}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();run(a.config)
