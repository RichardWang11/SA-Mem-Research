#!/usr/bin/env python3
"""Observed-config QA/LLM-J rerun against verified complete, read-only memories."""
from __future__ import annotations
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RESEARCH = Path('/workspace/research/tsm_temporal_first')
CODE = RESEARCH / 'TSM'
MEMORY = RESEARCH / 'runs/memory_repair_verified_20261025'
OLD = RESEARCH / 'runs/siliconflow_qwen_full_tmux_20261021'
JUDGE = Path('/workspace/SA-mem/tsm_acceptance_20261023/scripts/evaluate_llm_judge.py')


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def credentials():
    # Same selectors as the observed historical launcher. No secret persists.
    lines = [x.strip() for x in Path('/home/jql/4omini.txt').read_text().splitlines()]
    os.environ['OPENAI_EMBEDDING_API_KEY'] = lines[1]
    os.environ['TSM_JUDGE_API_KEY'] = lines[1]
    section = next(i for i, x in enumerate(lines) if x.lower().startswith('#siliconflow'))
    vals = {}
    for line in lines[section+1:]:
        if line.startswith('#'):
            break
        if '=' in line:
            k,v = line.split('=',1)
            vals[k.strip()] = v.strip().strip('\"').strip("'")
    if vals['OPENAI_BASE_URL'].rstrip('/') != 'https://api.siliconflow.cn/v1':
        raise RuntimeError('Historical SiliconFlow endpoint differs')
    if lines[0].rstrip('/') != 'https://api.relayrouter.ai/v1':
        raise RuntimeError('Historical RelayRouter endpoint differs')
    os.environ['OPENAI_API_KEY'] = vals['OPENAI_API_KEY']


def worker(args):
    # Work may be computed ahead in an isolated Conversation process. This
    # handoff lets the original coordinator consume it without a duplicate
    # model request, while its already-running workers continue unchanged.
    if args.output.parent.name == 'conversations':
        import shutil
        ahead = args.output.parent.parent/'precomputed'/args.conversation
        state = ahead/'state.json'
        if state.exists():
            while not (ahead/'verified_answers.json').exists():
                try:
                    current = json.loads(state.read_text())
                except json.JSONDecodeError:
                    time.sleep(1)
                    continue
                if current.get('status') == 'failed':
                    raise RuntimeError(f'Precomputed worker failed: {ahead}/console.log')
                time.sleep(1)
            assert set(json.loads(args.allowlist.read_text())) == set(json.loads((ahead/'qids.json').read_text()))
            for p in ahead.iterdir():
                if p.name == 'console.log':
                    shutil.copy2(p,args.output/'precomputed_console.log')
                elif p.is_dir():
                    shutil.copytree(p,args.output/p.name,dirs_exist_ok=True)
                elif p.name != 'state.json':
                    shutil.copy2(p,args.output/p.name)
            print(f'Validated Conversation outputs reused without another API call: {args.conversation}')
            return
    import asyncio
    import logging
    import yaml
    credentials()
    sys.path.insert(0, str(CODE))
    import main_time
    cfg = yaml.safe_load((MEMORY/'query_config.yaml').read_text())
    cfg['paths'] = {k: str(args.output/v) for k,v in
                    [('cache_path','cache'),('answer_path','answers'),('log_path','logs')]}
    cfg['conversation_filter'] = [args.conversation]
    cfg['query_filter_path'] = str(args.allowlist)
    cfg['query_only'] = True
    cfg['force_rebuild'] = cfg['rebuild_snapshots'] = False
    cfg['promote_cache'] = False
    cfg['graphiti']['reset_group'] = False
    cfg['llm_parallel']['max_parallel_llm'] = 1
    cfg['parallel_api_calls'] = 1
    os.environ['TSM_LLM_TRACE_DIR'] = str(args.output/'request_trace')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    main_time.get_cache_folder = lambda cache,data,date,piece: str(
        MEMORY/'cache'/'locomo10'/'20261025'/str(piece))
    asyncio.run(main_time.run_pipeline(cfg, logging.getLogger('E2GraphRAG')))
    files = list((args.output/'answers/Locomo').glob('*results.json'))
    if len(files) != 1:
        raise RuntimeError('Expected one Conversation answer file')
    rows = [q for v in json.loads(files[0].read_text()).values() if isinstance(v,list) for q in v]
    allowed = json.loads(args.allowlist.read_text())
    assert {r['question_id'] for r in rows} == set(allowed)
    for r in rows:
        if not r.get('retrieval_trace') or not r.get('retrieved_evidences'):
            raise RuntimeError(f'Empty retrieval trace/context: {r["question_id"]}')
        if not r.get('hypothesis') or r['hypothesis'] in ['API调用失败','无法生成答案'] or r['hypothesis'].startswith('处理失败:'):
            raise RuntimeError(f'Failed answer: {r["question_id"]}')
        assert len(r['retrieval_trace']['dense_top20_ids']) == 20
    for p in (args.output/'request_trace').glob('*tsm_chat_completion.json'):
        trace = json.loads(p.read_text())
        assert trace['status'] == 'completed'
        assert all(c.get('finish_reason') == 'stop' for c in trace['response']['choices'])
    write(args.output/'verified_answers.json', rows)


def launch(root, conv, qids):
    root.mkdir(parents=True, exist_ok=False)
    allow = root/'qids.json'
    write(allow, qids)
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker',
           '--conversation',conv,'--allowlist',str(allow),'--output',str(root)]
    with (root/'console.log').open('w') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'Conversation {conv} failed: {root}/console.log')
    return json.loads((root/'verified_answers.json').read_text())


def main(args):
    credentials()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = json.loads((MEMORY/'locomo10.json').read_text())
    allow = json.loads((MEMORY/'locomo_explicit_temporal_qids.json').read_text())['selected_qids']
    assert len(set(allow)) == len(allow) == 321
    groups = {conv['sample_id']:[q for q in allow if q.startswith(conv['sample_id']+'_')] for conv in dataset}
    identity = {'created_at':datetime.now(timezone.utc).isoformat(),
                'memory_manifest_sha256':hashlib.sha256((MEMORY/'manifest.json').read_bytes()).hexdigest(),
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
                'conversation_workers':args.workers,'within_conversation_workers':1,
                'embedding':'text-embedding-3-small','answer_model':'Qwen/Qwen3-30B-A3B-Instruct-2507',
                'top_k':20,'max_tokens':8192,'judge_model':'gpt-4o-mini','judge_prompt':'Mem0 paper Appendix A',
                'questions':321,'memory':str(MEMORY),'reextract_graph':False}
    write(args.output/'manifest.json',identity)
    write(args.output/'status.json',{'stage':'smoke','completed':0})
    # Fixed, known historical cases; success gates runtime correctness, not
    # answer correctness, so the smoke cannot bias question inclusion.
    for qid in ['conv-26_11','conv-48_105']:
        conv = qid.rsplit('_',1)[0]
        launch(args.output/'smoke'/qid,conv,[qid])
        print(f'[smoke] {qid}: retrieval/context/answer/finish_reason PASS',flush=True)
    write(args.output/'status.json',{'stage':'qa','completed':0,'expected':321})
    combined = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch,args.output/'conversations'/conv,conv,qids):conv for conv,qids in groups.items()}
        for future in concurrent.futures.as_completed(futures):
            conv = futures[future]
            combined[conv] = future.result()
            done = sum(len(v) for v in combined.values())
            write(args.output/'status.json',{'stage':'qa','completed':done,'expected':321,'conversations':len(combined)})
            print(f'[qa {done}/321] {conv}: {len(combined[conv])} answers PASS',flush=True)
    rows = [q for v in combined.values() for q in v]
    assert len(rows) == len({q['question_id'] for q in rows}) == 321
    assert {q['question_id'] for q in rows} == set(allow)
    results = args.output/'answers/results.json'
    write(results,combined)
    write(args.output/'status.json',{'stage':'llm_j','completed':0,'expected':321})
    cmd = [sys.executable,str(JUDGE),'--results',str(results),'--mode','all',
           '--workers','4','--output',str(args.output/'judge/mem0_gpt4omini_all'),'--execute']
    with (args.output/'judge_console.log').open('w') as log:
        proc = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'Judge incomplete; see {args.output}/judge_console.log')
    # Old records can lack subset fields; reaggregate unchanged raw labels
    # against the canonical CSV instead of trusting a zero-sized old summary.
    subprocess.run([sys.executable,str(Path(__file__).with_name('compare_repaired_judges.py')),
                    '--run',str(args.output)],check=True)
    comparison = json.loads((args.output/'comparison.json').read_text())
    old,new = comparison['before'],comparison['after']
    print(json.dumps({'main_before':old['main_244'],'main_after':new['main_244'],
                      'delta':comparison['main_accuracy_delta'],
                      'rescued':comparison['main_rescued'],'damaged':comparison['main_damaged']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--worker',action='store_true')
    ap.add_argument('--conversation')
    ap.add_argument('--allowlist',type=Path)
    args=ap.parse_args()
    if args.worker:
        worker(args)
    else:
        main(args)
