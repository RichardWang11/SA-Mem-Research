#!/usr/bin/env python3
"""Write a paired construction-repair QA report from completed real outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    r=args.run
    comparison=json.loads((r/'comparison.json').read_text())
    evidence=json.loads((r/'evidence_summary.json').read_text())
    samples={x['qid']:x for x in json.loads((r/'evidence_samples.json').read_text())}
    labels={x['qid']:x for x in json.loads((r/'label_changes.json').read_text())}
    answers={q['question_id']:q for v in json.loads((r/'answers/results.json').read_text()).values() if isinstance(v,list) for q in v}
    before=comparison['before'];after=comparison['after']
    assert after['main_244']['scored']==244 and after['adversarial_77_llm_j_synthetic_reference']['scored']==77
    def pct(x):return f'{100*x:.2f}%'
    lines=['# 修复完整 Dense Memory 后的 LoCoMo 显式时间 QA 对照','',
           '本报告比较建库修复前后的相同 321 个 QID；244 道主评测与 77 道 Adversarial 分开。',
           '两者均使用 Qwen/Qwen3-30B-A3B-Instruct-2507、Top-20、相同 Answer Prompt，',
           'Judge 均为 RelayRouter gpt-4o-mini + Mem0 论文附录 A 提示词。',
           '这不是 TSM 论文 GPT-4.1-mini Judge 口径，也不是 LoCoMo 全量 1986 题。','',
           '| 指标 | 修复前 | 修复后 |','|---|---:|---:|',
           f"| 主评测 LLM-J | {before['main_244']['correct']}/244 ({pct(before['main_244']['complete_accuracy'])}) | {after['main_244']['correct']}/244 ({pct(after['main_244']['complete_accuracy'])}) |",
           f"| Adversarial 合成拒答参考 LLM-J | {before['adversarial_77_llm_j_synthetic_reference']['correct']}/77 | {after['adversarial_77_llm_j_synthetic_reference']['correct']}/77 |",
           f"| Adversarial 官方拒答短语规则 | {before['adversarial_77_official_refusal_heuristic']['refusal_count']}/77 | {after['adversarial_77_official_refusal_heuristic']['refusal_count']}/77 |",'',
           f"主评测净变化：{comparison['main_accuracy_delta']*100:+.2f} 个百分点；错→对 {comparison['main_rescued']}，对→错 {comparison['main_damaged']}。",'',
           'Adversarial 的普通 gold answer 为空，合成参考被明确标注，不能混算为主评测标准 Accuracy。','',
           '## 证据覆盖','',
           'Gold Dialogue ID 通过实际 source_mapping 对齐 leaf；仅在全部 Gold 均可可靠映射的题目上统计召回。',
           f"主评测可映射 {evidence['main']['mapped']}/244，mapping_unknown {evidence['main']['mapping_unknown']}，无 Gold {evidence['main']['no_gold_evidence']}。",'',
           '| 主评测证据指标 | 修复前 | 修复后 |','|---|---:|---:|']
    for key,name in [('dense_top20_ids','Gold leaf Recall@20'),('dense_top100_ids','Gold leaf Recall@100'),('final_context_ids','最终上下文 Gold leaf 覆盖率')]:
        a=evidence['main']['before'][key];b=evidence['main']['after'][key]
        lines.append(f"| {name} | {a['gold_leaf_hits']}/{a['gold_leaf_denominator']} ({pct(a['gold_leaf_recall'])}) | {b['gold_leaf_hits']}/{b['gold_leaf_denominator']} ({pct(b['gold_leaf_recall'])}) |")
    lines+=['','## 修复后仍错误的可追踪样例','',
            '以下只列主评测，按 QID 排序。未召回与已召回仍答错分开记录；Gold 标注不一定穷尽全部可用证据。','']
    failures=[]
    for qid,change in sorted(labels.items()):
        if change['subset']!='main' or change['after']!='WRONG':continue
        s=samples[qid];q=answers[qid]
        category='mapping_unknown' if s['mapping_status']!='mapped' else (
            'gold_in_final_context_but_answer_wrong' if s['after']['final_context_ids']['all_gold'] else
            'dense_top20_missing_some_or_all_gold' if not s['after']['dense_top20_ids']['all_gold'] else 'context_error')
        failures.append({'qid':qid,'category':category,'question':q['question'],'gold_answer':q['answer'],
                         'answer':q['hypothesis'],'evidence':s})
    for f in failures[:10]:
        lines += [f"### {f['qid']} — {f['category']}",'',f"问题：{f['question']}",f"Gold：{f['gold_answer']}",
                  f"生成答案：{f['answer'][:600]}",f"Gold leaf：{', '.join(f['evidence']['gold_leaf_ids'])}",'']
    lexical=r/'lexical/qa_scoring_summary.json'
    if lexical.exists():
        scores=json.loads(lexical.read_text())
        lines+=['## 补充词面指标','',
                f"修复后主评测官方类别规则词级 F1 为 {pct(scores['main']['upstream_qa_score'])}，历史值为 2.9568%。",
                f"主评测生成答案平均 {scores['main']['mean_prediction_words']:.1f} 词，未进行事后简答提取。",
                '该 F1 与二元 LLM-J 不是同一指标，不能相互替代。','']
    lines += ['## 结论边界','',
              '这次比较修复了全量源文本进入 Dense Index 的前提。得分变化反映完整索引恢复后的结果，',
              '不能直接证明 Temporal-first 优于 Dense-first。后续需要在同一完整 Memory 上筛查真实 Dense Miss，',
              '再核验 Gold 对应事件边的时间和 Episode 映射。本轮没有实施 Temporal-first。',
              '并发只发生在不同 Conversation 的只读 QA；每个 Conversation 内逐题执行。',
              '外层初始 3 路，随后将 5 个排队 Conversation 提前独立执行，峰值 8 路；提前结果复用，未重复调用。',
              '历史 Judge 汇总的子集字段缺失，本次使用 CSV 与原始逐题标签重新分组，未修改标签或重调模型。',
              'API 响应 Trace 及 Judge 显式重试均保留；SDK 内部重试可能只有 console 日志，',
              '因此 API Token 合计是已记录返回响应的用量，不保证覆盖提供者全部计费。','']
    usage=Counter();models=Counter();statuses=Counter()
    for p in list((r/'conversations').glob('*/request_trace/*.json'))+list((r/'smoke').glob('*/request_trace/*.json')):
        t=json.loads(p.read_text());resp=t.get('response') or {};statuses[t.get('status')]+=1
        models[resp.get('model',t.get('model','unknown'))]+=1
        for k,v in (resp.get('usage') or {}).items():
            if k in ['prompt_tokens','completion_tokens','total_tokens'] and isinstance(v,int):usage[k]+=v
    write={'qa_and_smoke_usage':dict(usage),'qa_and_smoke_requested_or_returned_models':dict(models),
           'qa_and_smoke_trace_statuses':dict(statuses),'judge_usage':after['usage_including_retries_and_controls'],
           'failure_categories_main':dict(Counter(f['category'] for f in failures))}
    (r/'usage_and_failures.json').write_text(json.dumps(write,ensure_ascii=False,indent=2)+'\n')
    (r/'failure_cases.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
    (r/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(r/'REPORT.md')


if __name__=='__main__':main()
