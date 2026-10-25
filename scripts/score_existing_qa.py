"""Score unchanged TSM answers with pinned upstream LoCoMo QA functions, offline."""
import sys,ast,json,csv,hashlib,importlib.util,contextlib,io
from pathlib import Path
from collections import Counter,defaultdict
root=Path(__file__).resolve().parents[1]

import numpy as np
import regex,string
from nltk.stem import PorterStemmer
source=root/'results/full/locomo_official_evaluation.py'
# Execute the exact upstream QA functions. Avoid importing the unused BERTScore
# dependency; no change is made to the scorer functions themselves.
names={'normalize_answer','exact_match_score','f1_score','f1','eval_question_answering'}
tree=ast.parse(source.read_text()); funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in funcs}==names
ns={'np':np,'regex':regex,'string':string,'Counter':Counter,'ps':PorterStemmer()}
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),ns)
import gzip,argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--results',type=Path,help='Optional new TSM QA JSON or gzip; default is archived historical QA');args=p.parse_args()
if args.out.exists():raise SystemExit('Output exists')
args.out.mkdir(parents=True)
respath=args.results or root/'results/full/qa_results.json.gz'
result_bytes=gzip.decompress(respath.read_bytes()) if respath.suffix=='.gz' else respath.read_bytes()
data=json.loads(result_bytes); rows=[r for v in data.values() if isinstance(v,list) for r in v]
ann={r['qid']:r for r in csv.DictReader(open(root/'selection/annotations_locomo1986.csv')) if r['dataset']=='locomo'}
gold={r['question_id']:{'question':r['question'],'answer':r['answer'],'category':int(ann[r['question_id']]['official_type'])} for r in rows}
qas=[]
for r in rows:
 q=gold[r['question_id']]
 assert str(r['answer'])==str(q.get('answer',''))
 assert r['question']==q['question']
 qas.append({**q,'answer':q.get('answer',''),'prediction':r['hypothesis'] if 'hypothesis' in r else r['output_text']})
with contextlib.redirect_stdout(io.StringIO()):scores,_,unused_recall=ns['eval_question_answering'](qas,eval_key='prediction')
# The upstream returned recall defaults to 1 when no context IDs are provided;
# deliberately do not report this as evidence recall.
spec=importlib.util.spec_from_file_location('tsm_diag',str(root/'TSM/scripts/evaluate_predictions.py'));diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
per=[]
for r,q,s in zip(rows,qas,scores):
 pred=q['prediction'];a=ann[r['question_id']]
 per.append({'qid':r['question_id'],'scope':a['scope'],'category':q['category'],'main_subset':a['main_subset']=='True','upstream_qa_score':float(s),'diagnostic_exact_match':float(diag.normalize(pred)==diag.normalize(str(q.get('answer','')))),'diagnostic_token_f1':diag.token_f1(pred,str(q.get('answer',''))),'prediction_words':len(pred.split()),'parser_hit':bool(r['retrieval_trace']['tsm_parser_hit']),'tkg_chunk_match':bool(r['retrieval_trace']['temporal_matched_chunk_ids'])})
def group(xs):
 n=len(xs)
 return {'n':n,'upstream_qa_score':sum(x['upstream_qa_score'] for x in xs)/n,'diagnostic_exact_match':sum(x['diagnostic_exact_match'] for x in xs)/n,'diagnostic_token_f1':sum(x['diagnostic_token_f1'] for x in xs)/n,'mean_prediction_words':sum(x['prediction_words'] for x in xs)/n,'parser_hit':sum(x['parser_hit'] for x in xs),'tkg_chunk_match':sum(x['tkg_chunk_match'] for x in xs)}
summary={'evaluator_repository':'https://github.com/snap-research/locomo','evaluator_commit':'3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376','evaluator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'result_sha256':hashlib.sha256(result_bytes).hexdigest(),'execution':'Unchanged upstream QA function AST executed; unused BERTScore imports excluded. Full original generated answers scored without answer extraction.','metrics_scope':'Main: category-specific lexical F1. Adversarial: upstream refusal-phrase heuristic. No semantic judge accuracy; no retrieval recall derived from evaluator.','overall_mixed_not_accuracy':group(per),'main':group([x for x in per if x['main_subset']]),'adversarial':group([x for x in per if not x['main_subset']]),'by_category':{str(c):group([x for x in per if x['category']==c]) for c in sorted({x['category'] for x in per})},'by_conversation':{c:group([x for x in per if x['scope']==c and x['main_subset']]) for c in sorted({x['scope'] for x in per})},'adversarial_heuristic_pass_count':sum(x['upstream_qa_score']==1 for x in per if not x['main_subset'])}
(args.out/'qa_scoring_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (args.out/'qa_per_question_scores.jsonl').open('w') as f:
 for x in per:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
