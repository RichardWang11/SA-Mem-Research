#!/usr/bin/env python3
"""Reaggregate original judge labels using canonical CSV subsets; no API calls."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path

BASE=Path('/workspace/SA-mem/tsm_acceptance_20261023/judge/mem0_gpt4omini_all_20261025')
OLD_RESULTS=Path('/workspace/research/tsm_temporal_first/runs/siliconflow_qwen_full_tmux_20261021/answers/Locomo/locomo10_split_round_based_c6dd0749_results.json')
ANN=Path('/workspace/SA-mem/annotations-locomo.csv')


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def audited(directory,results,annotations):
    original=json.loads((directory/'summary.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['identity']['results_sha256']==hashlib.sha256(results.read_bytes()).hexdigest()
    payload=json.loads(results.read_text())
    answers={r['question_id']:r for v in payload.values() if isinstance(v,list) for r in v}
    labels={qid:json.loads((directory/'results'/f'{qid}.json').read_text()) for qid in annotations}
    assert set(answers)==set(labels)==set(annotations)
    assert all(r['status']=='completed' and r['label'] in ('CORRECT','WRONG') for r in labels.values())
    # Ensure the persisted judge inputs actually belong to these generated answers.
    tasks={r['qid']:r for r in json.loads((directory/'inputs.json').read_text())}
    for qid in labels:
        assert tasks[qid]['generated_answer']==answers[qid]['hypothesis']
        assert tasks[qid]['question']==answers[qid]['question']
        assert labels[qid]['request_sha256']==tasks[qid]['request_sha256']
    def metric(qids):
        correct=sum(labels[q]['label']=='CORRECT' for q in qids)
        return {'expected':len(qids),'scored':len(qids),'errors':0,'correct':correct,'wrong':len(qids)-correct,
                'accuracy_on_scored':correct/len(qids),'complete_accuracy':correct/len(qids)}
    main=[q for q,a in annotations.items() if a['main_subset']=='True']
    adv=[q for q,a in annotations.items() if a['main_subset']!='True']
    assert len(main)==244 and len(adv)==77
    summary=dict(original)
    summary['real_questions']=metric(list(labels));summary['main_244']=metric(main)
    summary['adversarial_77_llm_j_synthetic_reference']=metric(adv)
    refusal=sum(any(s in answers[q]['hypothesis'].lower() for s in ['no information available','not mentioned']) for q in adv)
    summary['adversarial_77_official_refusal_heuristic']={'expected':77,'refusal_count':refusal,'refusal_rate':refusal/77}
    summary['reaggregation_note']='Subset membership taken from canonical annotation CSV, not optional subset fields in old per-question result records. Raw completed judge labels reused unchanged; no model calls.'
    return summary,labels,manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    ann={r['qid']:r for r in csv.DictReader(ANN.open()) if r['dataset']=='locomo' and r['jev_label']=='explicit_temporal_filter'}
    before,old,om=audited(BASE,OLD_RESULTS,ann)
    after,new,nm=audited(args.run/'judge/mem0_gpt4omini_all',args.run/'answers/results.json',ann)
    for key in ['prompt_sha256','model','base_url','temperature','max_tokens','response_format']:
        assert om['identity'][key]==nm['identity'][key],key
    changes=[{'qid':q,'subset':'main' if ann[q]['main_subset']=='True' else 'adversarial',
              'before':old[q]['label'],'after':new[q]['label']} for q in ann]
    comparison={'before':before,'after':after,
        'main_accuracy_delta':after['main_244']['complete_accuracy']-before['main_244']['complete_accuracy'],
        'main_rescued':sum(x['subset']=='main' and x['before']=='WRONG' and x['after']=='CORRECT' for x in changes),
        'main_damaged':sum(x['subset']=='main' and x['before']=='CORRECT' and x['after']=='WRONG' for x in changes),
        'judge_settings_verified_equal':True,'baseline_summary_reaggregated':True}
    write(args.run/'baseline_judge_summary_reaudited.json',before)
    write(args.run/'repaired_judge_summary_reaudited.json',after)
    write(args.run/'comparison.json',comparison)
    write(args.run/'label_changes.json',changes)
    write(args.run/'status.json',{'stage':'completed','qa':321,'judge':321,'paired_comparison_verified':True})
    print(json.dumps({k:v for k,v in comparison.items() if k not in ['before','after']},indent=2))


if __name__=='__main__':main()
