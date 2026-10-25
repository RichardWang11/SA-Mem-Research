#!/usr/bin/env python3
"""Check recorded smoke artifacts without network, Neo4j, models or extraction."""
import json
import tarfile
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'results/smoke/recorded_conv26_smoke.tar.gz'
    with tarfile.open(p,'r:gz') as tar:
        files=[m for m in tar.getmembers() if m.isfile()]
        def get(member):
            return json.load(tar.extractfile(member))
        traces=[m for m in files if m.name.startswith('request_trace/') and m.name.endswith('.json')]
        statuses=Counter();stages=Counter();errors=[]
        for m in traces:
            d=get(m);statuses[d['status']]+=1;stages[d['stage']]+=1
            if d['status']!='completed':errors.append(m.name)
        result_member=next(m for m in files if m.name.endswith('_results.json'))
        results=get(result_member)
        rows=[r for v in results.values() if isinstance(v,list) for r in v]
        assert len(rows)==1 and rows[0]['question_id']=='conv-26_11'
        q=rows[0];assert q['output_text'] and q['retrieval_trace']['dense_top20_ids']
        tree=get(next(m for m in files if m.name.endswith('/tree.json')))
        mapping=get(next(m for m in files if m.name.endswith('/dense_index_mapping.json')))
        episodes=get(next(m for m in files if m.name.endswith('/text_edge_mapping.json')))
        assert mapping['num_docs']==len(tree)
        snapshots=get(next(m for m in files if m.name.endswith('/snapshots.json')))
        assert snapshots
        report={'scope':'recorded_artifact_checks_only_not_live_or_quality_acceptance','query_id':q['question_id'],'trace_files':len(traces),'request_status':dict(statuses),'stages':dict(stages),'non_completed_files':errors,'tree_leaves':len(tree),'input_round_records':len(episodes),'snapshots':len(snapshots),'prediction':q['output_text'],'known_issue':'Dense Tree does not cover complete input rounds; smoke completion did not validate memory coverage.','paper_llm_judge_acc':None}
        print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
