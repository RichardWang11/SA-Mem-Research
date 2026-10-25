#!/usr/bin/env python3
"""Print recorded candidate evidence and verify the existing-memory rank projection."""
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads(gzip.decompress((ROOT/'results/cases/deeper_candidate_conv48_105.json.gz').read_bytes()))
    q=data['query'];trace=q['retrieval_trace']
    order=list(dict.fromkeys(trace['dense_top100_ids']))
    # Archived source texts were checked against leaf content during the
    # original read-only audit. Do not infer evidence truth from matching names.
    names={edge['episode_name'] for edge in data['time_edges']}
    filtered=[lid for lid in order if lid.replace('leaf_','chunk_',1) in names]
    assert order.index('leaf_34')+1==31
    assert filtered.index('leaf_34')+1==6
    print(json.dumps({'qid':q['question_id'],'question':q['question'],'gold_answer':q['answer'],'dense_rank':31,'time_filtered_rank':6,'official_gold_dialogues':q['golden_evidence']['evidence'],'extra_support_dialogues':['D4:5','D4:6'],'temporal_candidates':filtered,'baseline_output':q['output_text'],'scope':'Existing incomplete memory only. Additional supporting evidence; not a verified wrong-to-correct QA rescue.'},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
