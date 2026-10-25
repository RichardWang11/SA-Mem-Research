#!/usr/bin/env python3
"""Portable TSM smoke/full launcher; dry-run by default, credentials via env."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['official-default', 'historical-qwen-batch1'], default='official-default')
    p.add_argument('--scope', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--dataset', type=Path, default=ROOT / 'TSM/data/locomo/locomo10.json')
    p.add_argument('--output', type=Path, help='Must not already exist; default is a fresh outputs/ directory.')
    p.add_argument('--execute', action='store_true', help='Actually build memory and query; incurs API cost.')
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT / 'configs' / (args.profile + '.yaml')).read_text())
    run_id = f"{args.scope}_{args.profile}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    out = (args.output or ROOT / 'outputs' / run_id).resolve()
    if out.exists():
        p.error('Output exists; choose a fresh directory. No existing output will be deleted.')
    cfg['experiment']['entrypoint'] = str(ROOT / 'TSM/main_time.py')
    cfg['dataset']['dataset_path'] = str(args.dataset.resolve())
    cfg['query_filter_path'] = str(ROOT / 'selection' / ('smoke_qids.json' if args.scope == 'smoke' else 'explicit321_qids.json'))
    cfg['paths'] = {f'{k}_path': str(out / v) for k, v in [('log','logs'), ('answer','answers'), ('cache','cache')]}
    cfg['graphiti']['group_id_prefix'] = 'tsm_handoff_' + uuid.uuid4().hex + '_'
    cfg['graphiti']['reset_group'] = False
    if args.scope == 'smoke':
        cfg['conversation_filter'] = ['conv-26']
    cfg['llm']['base_url'] = os.getenv('TSM_LLM_BASE_URL', cfg['llm']['base_url'])
    cfg['embedding']['base_url'] = os.getenv('TSM_EMBEDDING_BASE_URL', cfg['embedding']['base_url'])
    command = [sys.executable, str(ROOT/'TSM/main_time.py'), '--config', str(out/'config.resolved.yaml'), '--force-rebuild']
    summary = {'profile':args.profile, 'scope':args.scope, 'model':cfg['llm']['api_model'], 'batch_size':cfg['batch_size'], 'parallel_api_calls':cfg['parallel_api_calls'], 'max_parallel_llm':cfg['llm_parallel']['max_parallel_llm'], 'top_k':cfg['retriever']['kwargs']['max_chunk_setting'], 'graphiti_max_tokens':cfg['llm'].get('graphiti_max_tokens', cfg['llm']['max_tokens']), 'output':str(out), 'complete_history':True, 'selected_queries':1 if args.scope=='smoke' else 321, 'dataset_exists':args.dataset.is_file(), 'command':command, 'execute':args.execute}
    print(json.dumps(summary, indent=2), flush=True)
    if not args.execute:
        return 0
    if not args.dataset.is_file() or hashlib.sha256(args.dataset.read_bytes()).hexdigest()!=DATA_SHA:
        p.error('Dataset missing or checksum differs. Run TSM/scripts/download_data.py --help.')
    env = os.environ.copy()
    if not env.get('OPENAI_EMBEDDING_API_KEY') and env.get('OPENAI_API_KEY') and cfg['embedding']['base_url']==cfg['llm']['base_url']:
        env['OPENAI_EMBEDDING_API_KEY'] = env['OPENAI_API_KEY']
    for key in ['OPENAI_API_KEY','OPENAI_EMBEDDING_API_KEY','NEO4J_PASSWORD']:
        if not env.get(key):
            p.error(f'Missing environment variable: {key}')
    # Historical run used a global Graphiti semaphore of 1; official profile
    # leaves Graphiti library defaults intact rather than silently serializing it.
    if args.profile == 'historical-qwen-batch1':
        env['SEMAPHORE_LIMIT'] = '1'
    elif env.get('SEMAPHORE_LIMIT'):
        p.error('Unset SEMAPHORE_LIMIT to use official Graphiti concurrency defaults.')
    env['TSM_LLM_TRACE_DIR'] = str(out/'request_trace')
    out.mkdir(parents=True)
    (out/'config.resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out/'launcher_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'console.log').open('w') as log:
        proc = subprocess.Popen(command,cwd=ROOT/'TSM',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code = proc.wait()
    summary['exit_code']=code
    (out/'launcher_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
