#!/usr/bin/env python3
"""Evidence coverage audit for repaired QA; mapping_unknown never counts as miss."""
import argparse
import csv
import json
from pathlib import Path
import re

ROOT=Path('/workspace/research/tsm_temporal_first')
MEMORY=ROOT/'runs/memory_repair_verified_20261025'
OLD=ROOT/'runs/siliconflow_qwen_full_tmux_20261021/answers/Locomo/locomo10_split_round_based_c6dd0749_results.json'


def flatten(path):
    return {r['question_id']:r for v in json.loads(path.read_text()).values() if isinstance(v,list) for r in v}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    data=json.loads((MEMORY/'locomo10.json').read_text())
    ann={r['qid']:r for r in csv.DictReader(open('/workspace/SA-mem/annotations-locomo.csv'))
         if r['dataset']=='locomo' and r['jev_label']=='explicit_temporal_filter'}
    old=flatten(OLD)
    if args.partial:
        new={}
        for p in list((args.run/'precomputed').glob('*/answers/Locomo/*results.json')) + list((args.run/'conversations').glob('*/answers/Locomo/*results.json')):
            try:new.update(flatten(p))
            except json.JSONDecodeError:pass
    else:
        new=flatten(args.run/'answers/results.json')
    samples=[]
    for i,conv in enumerate(data):
        sources=json.loads((MEMORY/'cache/locomo10/20261025'/f'item_{i}_all_sessions/source_mapping.json').read_text())
        by_dialogue={did:r['leaf_id'] for r in sources for did in [r['user_dialogue_id'],r['assistant_dialogue_id']] if did}
        for j,gold in enumerate(conv['qa']):
            qid=f"{conv['sample_id']}_{j}"
            if qid not in ann or qid not in new:continue
            evidence=set(re.findall(r'D\d+:\d+',json.dumps(gold.get('evidence',[]))))
            unknown=sorted(evidence-set(by_dialogue))
            leaves={by_dialogue[d] for d in evidence if d in by_dialogue}
            status='no_gold_evidence' if not evidence else 'mapping_unknown' if unknown else 'mapped'
            row={'qid':qid,'main_subset':ann[qid]['main_subset']=='True','category':gold['category'],
                 'gold_dialogue_ids':sorted(evidence),'gold_leaf_ids':sorted(leaves),
                 'mapping_status':status,'unmapped_dialogue_ids':unknown}
            for name,outputs in [('before',old),('after',new)]:
                trace=outputs[qid].get('retrieval_trace',{})
                row[name]={}
                for field in ['dense_top20_ids','dense_top100_ids','final_context_ids']:
                    ids=set(trace.get(field,[]))
                    row[name][field]={'matched_gold_leaves':len(leaves&ids),'gold_leaves':len(leaves),
                                      'any_gold':bool(leaves&ids),'all_gold':bool(leaves) and leaves<=ids}
                row[name]['parser_hit']=trace.get('tsm_parser_hit',False)
                row[name]['tkg_active']=bool(trace.get('temporal_matched_chunk_ids'))
            samples.append(row)
    assert len(samples)==(len(new) if args.partial else 321)
    summary={}
    for subset in ['main','adversarial','all']:
        rows=[r for r in samples if subset=='all' or r['main_subset']==(subset=='main')]
        mapped=[r for r in rows if r['mapping_status']=='mapped']
        metrics={'samples':len(rows),'mapped':len(mapped),'mapping_unknown':sum(r['mapping_status']=='mapping_unknown' for r in rows),
                 'no_gold_evidence':sum(r['mapping_status']=='no_gold_evidence' for r in rows)}
        for period in ['before','after']:
            m={'parser_hits':sum(r[period]['parser_hit'] for r in rows),'tkg_active':sum(r[period]['tkg_active'] for r in rows)}
            for field in ['dense_top20_ids','dense_top100_ids','final_context_ids']:
                hits=sum(r[period][field]['matched_gold_leaves'] for r in mapped)
                denom=sum(r[period][field]['gold_leaves'] for r in mapped)
                m[field]={'sample_any_gold':sum(r[period][field]['any_gold'] for r in mapped),
                          'sample_all_gold':sum(r[period][field]['all_gold'] for r in mapped),
                          'sample_denominator':len(mapped),'gold_leaf_hits':hits,'gold_leaf_denominator':denom,
                          'gold_leaf_recall':hits/denom if denom else None}
            metrics[period]=m
        summary[subset]=metrics
    summary['note']='Same global leaf IDs are auditable via source_mapping. Historical missing-index gold is distinct from within-index Dense Miss. Unknown or absent gold excluded from evidence recall denominators.'
    prefix='partial_' if args.partial else ''
    summary['partial']=args.partial
    (args.run/(prefix+'evidence_samples.json')).write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    (args.run/(prefix+'evidence_summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary['main'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
