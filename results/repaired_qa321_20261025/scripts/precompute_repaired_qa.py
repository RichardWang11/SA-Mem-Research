#!/usr/bin/env python3
"""Compute queued independent Conversations without interrupting live workers."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

RUNNER=Path(__file__).with_name('run_repaired_qa_comparison.py')


def write(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    convs=['conv-44','conv-47','conv-48','conv-49','conv-50']
    qids=json.loads(Path('/workspace/research/tsm_temporal_first/runs/memory_repair_verified_20261025/locomo_explicit_temporal_qids.json').read_text())['selected_qids']
    for conv in convs:
        if (args.run/'conversations'/conv).exists():
            raise RuntimeError(f'Coordinator has already started {conv}; refuse duplicate computation')
    for conv in convs:
        p=args.run/'precomputed'/conv;p.mkdir(parents=True,exist_ok=False)
        write(p/'qids.json',[q for q in qids if q.startswith(conv+'_')])
        write(p/'state.json',{'status':'reserved'})
    p=args.run/'manifest.json';m=json.loads(p.read_text())
    m['additional_precomputed_conversation_workers']=5
    m['peak_answer_api_parallelism']=8
    m['concurrency_note']='Five queued independent Conversations computed ahead and reused by coordinator; each Conversation remains sequential; no model request intentionally duplicated.'
    write(p,m)
    def launch(conv):
        p=args.run/'precomputed'/conv
        write(p/'state.json',{'status':'running'})
        with (p/'console.log').open('w') as log:
            code=subprocess.run([sys.executable,str(RUNNER),'--worker','--conversation',conv,
                                 '--allowlist',str(p/'qids.json'),'--output',str(p)],stdout=log,stderr=subprocess.STDOUT).returncode
        write(p/'state.json',{'status':'completed' if code==0 else 'failed','returncode':code})
        print(conv,code,flush=True)
        if code:raise RuntimeError(f'{conv} failed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(launch,convs))


if __name__=='__main__':main()
