#!/usr/bin/env python3
"""Run TSM query-only on restored caches with date-independent cache paths."""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    args = ap.parse_args()
    sys.path.insert(0,str(ROOT/'code'))
    import main_time
    config = yaml.safe_load(args.config.read_text())
    if config.get('conversation_filter'):
        raise RuntimeError('Handoff full evaluation does not accept a Conversation filter')
    config['query_only'] = True
    config['force_rebuild'] = False
    config['rebuild_snapshots'] = False
    root = Path(config['paths']['cache_path']) / 'locomo10'
    for i in range(10):
        path = root/f'item_{i}_all_sessions'
        if not (path/'dense_index.faiss').is_file():
            raise RuntimeError(f'Missing recovered index: {path}')
    # Only the filesystem resolver changes; retrieval/answer behavior stays
    # in the included main_time implementation. Avoid its YYYYMMDD cache trap.
    main_time.get_cache_folder = lambda cache,data,date,piece: str(Path(cache)/data/str(piece))
    work = Path(config['paths']['answer_path']).parent
    attempt = len(list((work/'request_trace').glob('attempt_*'))) + 1
    os.environ['TSM_LLM_TRACE_DIR'] = str(work/'request_trace'/f'attempt_{attempt:03d}')
    asyncio.run(main_time.run_pipeline(config,logging.getLogger('E2GraphRAG')))
    files = list(Path(config['paths']['answer_path']).glob('Locomo/*results.json'))
    if len(files) != 1:
        raise RuntimeError('Expected one QA result file')
    rows = [q for v in json.loads(files[0].read_text()).values() if isinstance(v,list) for q in v]
    expected = 321 if config.get('query_filter_path') else 1986
    assert len(rows)==len({q['question_id'] for q in rows})==expected
    errors = [q['question_id'] for q in rows if not q.get('hypothesis') or
              q.get('hypothesis') in ['API调用失败','无法生成答案','配置错误：未启用API']]
    if errors:
        raise RuntimeError(f'{len(errors)} API/empty answers; see result file')
    print(f'Completed {expected} QA answers: {files[0]}')


if __name__ == '__main__':
    main()
