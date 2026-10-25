#!/usr/bin/env python3
"""Mem0 LoCoMo LLM judge. Reads existing answers; never rebuilds memory."""
import argparse,collections,concurrent.futures,hashlib,json,os,re,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=Path('/workspace/research/tsm_temporal_first/runs/siliconflow_qwen_full_tmux_20261021')
def sha(x):return hashlib.sha256(x).hexdigest()
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def label(text):
 # Accept only a whole JSON object (optionally a whole fenced JSON block),
 # never classify malformed output or exceptions as WRONG.
 s=text.strip()
 if s.startswith('```'):
  m=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',s,re.S)
  if not m:raise ValueError('Invalid JSON fence')
  s=m.group(1)
 d=json.loads(s)
 if not isinstance(d,dict) or d.get('label') not in ('CORRECT','WRONG'):raise ValueError('Missing exact CORRECT/WRONG label')
 return d['label']
def samples(result,annotations,dataset,subset):
 import csv
 with annotations.open(newline='',encoding='utf-8') as handle:
  ann={r['qid']:r for r in csv.DictReader(handle) if r['dataset']=='locomo' and r['jev_label']=='explicit_temporal_filter'}
 gold={f"{c['sample_id']}_{i}":q for c in json.loads(dataset.read_text()) for i,q in enumerate(c['qa'])}
 rows=[q for v in json.loads(result.read_text()).values() if isinstance(v,list) for q in v]
 assert len(rows)==321 and len({q['question_id'] for q in rows})==321
 assert {q['question_id'] for q in rows}==set(ann)
 out=[];excluded=[]
 for r in rows:
  qid=r['question_id'];a=ann[qid];g=gold[qid]
  assert r['question']==g['question']
  is_main=a['main_subset']=='True'
  if subset=='main' and not is_main:excluded.append(qid);continue
  if subset=='adversarial' and is_main:excluded.append(qid);continue
  if is_main:
   assert str(r['answer'])==str(g.get('answer',''))
   assert int(g['category'])!=5
   gold_answer=str(g['answer'])
  else:
   assert str(r['answer']) in ('','None') and int(g['category'])==5 and g.get('answer') in (None,'')
   gold_answer='No information available.'
  pred=r.get('hypothesis',r.get('output_text',''))
  assert isinstance(pred,str) and pred.strip()
  if is_main: assert 'answer' in g and str(g['answer']).strip()
  out.append({'qid':qid,'question':g['question'],'gold_answer':gold_answer,'generated_answer':pred,'category':int(g['category']),'kind':'real','subset':'main' if is_main else 'adversarial','gold_source':'locomo.answer' if is_main else 'synthetic_abstention_reference'})
 assert len(out)=={'main':244,'adversarial':77,'all':321}[subset]
 return out,excluded
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--results',type=Path,default=DEFAULT_RUN/'answers/Locomo/locomo10_split_round_based_c6dd0749_results.json')
 ap.add_argument('--annotations',type=Path,default=Path('/workspace/SA-mem/annotations-locomo.csv'))
 ap.add_argument('--dataset',type=Path,default=Path('/workspace/research/tsm_temporal_first/TSM/data/locomo/locomo10.json'))
 ap.add_argument('--prompt',type=Path,default=ROOT/'judge/prompts/mem0_locomo.txt')
 ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--mode',choices=['smoke','main','adversarial','all'],default='smoke')
 ap.add_argument('--model',default='gpt-4o-mini')
 ap.add_argument('--base-url',default='https://api.relayrouter.ai/v1')
 ap.add_argument('--credentials-file',type=Path,default=Path('/home/jql/4omini.txt'))
 ap.add_argument('--workers',type=int,default=1)
 ap.add_argument('--attempts',type=int,default=3)
 ap.add_argument('--timeout',type=float,default=90)
 ap.add_argument('--execute',action='store_true')
 ap.add_argument('--resume',action='store_true')
 args=ap.parse_args()
 assert 1<=args.workers<=8 and args.attempts>=1
 subset={'main':'main','adversarial':'adversarial','all':'all'}.get(args.mode,'main')
 rows,excluded=samples(args.results,args.annotations,args.dataset,subset)
 if args.mode=='smoke':
  chosen={'conv-26_11','conv-26_31','conv-48_105'}
  rows=[r for r in rows if r['qid'] in chosen];assert len(rows)==3
  for lab,pred in [('CORRECT','A shell necklace.'),('WRONG','A red bicycle.')]:
   rows.append({'qid':'control_'+lab.lower(),'question':'What did I get on my last trip to Hawaii?','gold_answer':'A shell necklace','generated_answer':pred,'expected_label':lab,'kind':'control','category':None})
  refusal=next(r for r in rows if r['qid']=='conv-26_11')
  rows.append({**refusal,'qid':'control_real_refusal','expected_label':'WRONG','kind':'control','category':None})
  rows.append({'qid':'control_short_refusal','question':'Where did Caroline move from?','gold_answer':'Sweden','generated_answer':'There is no information about where Caroline moved from.','expected_label':'WRONG','kind':'control','category':None})
  rows.append({'qid':'control_wrong_date','question':'On what date did Caroline attend the meeting?','gold_answer':'7 May 2023','generated_answer':'She attended on 7 May 2022.','expected_label':'WRONG','kind':'control','category':None})
 template=args.prompt.read_text()
 identity={'results_sha256':sha(args.results.read_bytes()),'annotations_sha256':sha(args.annotations.read_bytes()),'dataset_sha256':sha(args.dataset.read_bytes()),'prompt_sha256':sha(template.encode()),'runner_sha256':sha(Path(__file__).read_bytes()),'model':args.model,'base_url':args.base_url,'mode':args.mode,'subset':subset,'qids':[r['qid'] for r in rows],'temperature':0,'max_tokens':512,'response_format':'json_object','timeout_seconds':args.timeout,'max_attempts':args.attempts,'workers':args.workers}
 fingerprint=sha(json.dumps(identity,sort_keys=True).encode())
 tasks=[]
 for r in rows:
  prompt=template.format(question=r['question'],gold_answer=r['gold_answer'],generated_answer=r['generated_answer'])
  tasks.append({**r,'request_sha256':sha(prompt.encode()),'messages':[{'role':'user','content':prompt}]})
 if not args.execute:
  print(json.dumps({'dry_run':True,'identity':identity,'fingerprint':fingerprint,'real_questions':sum(r['kind']=='real' for r in rows),'controls':sum(r['kind']=='control' for r in rows),'excluded_adversarial':len(excluded),'input_chars':sum(len(t['messages'][0]['content']) for t in tasks),'output_directory':str(args.output)},ensure_ascii=False,indent=2));return
 # Credentials are only read during execution. Never persisted in artifacts.
 vals=[x.strip() for x in args.credentials_file.read_text().splitlines()]
 key=os.getenv('TSM_JUDGE_API_KEY') or vals[1]
 if not key or not args.base_url.startswith('https://'):raise ValueError('Missing credentials or invalid endpoint')
 if args.output.exists() and any(args.output.iterdir()) and not args.resume:raise ValueError('Output exists: use a new directory or --resume')
 args.output.mkdir(parents=True,exist_ok=True)
 mp=args.output/'manifest.json'
 if mp.exists():
  old=json.loads(mp.read_text())
  if old['fingerprint']!=fingerprint:raise ValueError('Resume fingerprint differs: use a new output directory')
 else:dump(mp,{'fingerprint':fingerprint,'identity':identity,'created_at':datetime.now(timezone.utc).isoformat(),'prompt_source':'Mem0 paper Appendix A; https://arxiv.org/pdf/2504.19413v1#page=18','official_locomo_judge_runner':False,'excluded_qids':excluded,'adversarial_metric_note':'Category 5 has no ordinary LoCoMo gold answer; adversarial rows use the explicitly labelled synthetic abstention reference.'})
 (args.output/'judge_prompt.txt').write_text(template)
 (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
 dump(args.output/'inputs.json',tasks)
 def evaluate(t):
  final=args.output/'results'/f"{t['qid']}.json"
  if args.resume and final.exists():
   prev=json.loads(final.read_text())
   if prev.get('status')=='completed' and prev.get('request_sha256')==t['request_sha256']:return prev
  from openai import OpenAI
  client=OpenAI(api_key=key,base_url=args.base_url,max_retries=0,timeout=args.timeout)
  previous=list((args.output/'request_trace').glob(t['qid']+'_*.json')) if (args.output/'request_trace').exists() else []
  offset=max([int(p.stem.rsplit('_',1)[1]) for p in previous],default=0)
  record=None
  try:
   for attempt in range(1,args.attempts+1):
    tp=args.output/'request_trace'/f"{t['qid']}_{offset+attempt:03d}.json"
    req={'model':args.model,'messages':t['messages'],'temperature':0,'max_tokens':512,'response_format':{'type':'json_object'}}
    record={'qid':t['qid'],'kind':t['kind'],'category':t['category'],'subset':t.get('subset'),'gold_source':t.get('gold_source'),'expected_label':t.get('expected_label'),'generated_answer':t.get('generated_answer'),'request_sha256':t['request_sha256'],'attempt':offset+attempt,'status':'pending','request':req,'started_at':datetime.now(timezone.utc).isoformat()}
    dump(tp,record);start=time.monotonic()
    try:
     response=client.chat.completions.create(**req)
     record['response']=response.model_dump(mode='json')
     if response.choices[0].finish_reason!='stop':raise ValueError('Incomplete completion')
     record['label']=label(response.choices[0].message.content or '')
     record['status']='completed'
    except Exception as exc:
     record['status']='error';record['error_type']=type(exc).__name__
     # Do not persist provider error strings, request headers, or credentials.
     record['http_status']=getattr(exc,'status_code',None)
    record['elapsed_seconds']=round(time.monotonic()-start,3);dump(tp,record)
    if record['status']=='completed':break
    code=record.get('http_status')
    if isinstance(code,int) and 400<=code<500 and code not in (408,409,429):break
    if attempt<args.attempts:time.sleep(min(2**attempt,8))
   dump(final,record);return record
  finally:client.close()
 completed=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  fs={pool.submit(evaluate,t):t for t in tasks}
  for f in concurrent.futures.as_completed(fs):
   r=f.result();completed.append(r)
   print(f"[{len(completed)}/{len(tasks)}] {r['qid']} {r['status']} {r.get('label','UNSCORED')}",flush=True)
 real=[r for r in completed if r['kind']=='real'];controls=[r for r in completed if r['kind']=='control']
 import csv
 with args.annotations.open(newline='',encoding='utf-8') as handle:
  subset_by_qid={x['qid']:('main' if x['main_subset']=='True' else 'adversarial') for x in csv.DictReader(handle) if x['dataset']=='locomo' and x['jev_label']=='explicit_temporal_filter'}
 def metrics(rs):
  ok=[r for r in rs if r['status']=='completed'];correct=sum(r.get('label')=='CORRECT' for r in ok)
  return {'expected':len(rs),'scored':len(ok),'errors':len(rs)-len(ok),'correct':correct,'wrong':len(ok)-correct,'accuracy_on_scored':correct/len(ok) if ok else None,'complete_accuracy':correct/len(rs) if rs and len(ok)==len(rs) else None}
 groups=collections.defaultdict(list)
 for r in completed:
  if r['status']=='completed':groups[r['request_sha256']].append(r)
 conflicts=[{'request_sha256':k,'qids':[r['qid'] for r in rs],'labels':[r['label'] for r in rs]} for k,rs in groups.items() if len({r['label'] for r in rs})>1]
 usage=collections.Counter();statuses=collections.Counter();returned=collections.Counter()
 for p in (args.output/'request_trace').glob('*.json'):
  r=json.loads(p.read_text());statuses[r['status']]+=1;resp=r.get('response') or {}
  if resp.get('model'):returned[resp['model']]+=1
  for k,v in (resp.get('usage') or {}).items():
   if k in ['prompt_tokens','completion_tokens','total_tokens'] and isinstance(v,int):usage[k]+=v
 main_real=[r for r in real if subset_by_qid.get(r['qid'])=='main'];adv_real=[r for r in real if subset_by_qid.get(r['qid'])=='adversarial']
 adv_refusal=sum(any(x in r.get('generated_answer','').lower() for x in ('no information available','not mentioned')) for r in adv_real)
 summary={'mode':args.mode,'judge_prompt':'Mem0 paper Appendix A','judge_model_requested':args.model,'judge_models_returned':dict(returned),'real_questions':metrics(real),'main_244':metrics(main_real),'adversarial_77_llm_j_synthetic_reference':metrics(adv_real),'adversarial_77_official_refusal_heuristic':{'expected':len(adv_real),'refusal_count':adv_refusal,'refusal_rate':adv_refusal/len(adv_real) if adv_real else None},'controls':{'expected':len(controls),'passed':sum(r['status']=='completed' and r.get('label')==r['expected_label'] for r in controls)},'by_category':{str(c):metrics([r for r in real if r['category']==c]) for c in sorted({r['category'] for r in real})},'usage_including_retries_and_controls':dict(usage),'request_statuses':dict(statuses),'excluded_qids':len(excluded),'note':'Main uses real LoCoMo answers. Category 5 has no ordinary gold; adversarial LLM-J uses a synthetic abstention reference and must be reported separately. The refusal heuristic follows LoCoMo official evaluation.py. Invalid labels/API errors are unscored, not WRONG.'}
 summary['identical_prompt_conflicts']=conflicts
 summary['smoke_acceptance_passed']=(not conflicts and not any(r['status']!='completed' for r in completed) and summary['controls']['passed']==len(controls)) if args.mode=='smoke' else None
 dump(args.output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
 if conflicts or any(r['status']!='completed' for r in completed) or summary['controls']['passed']!=len(controls):sys.exit(2)
if __name__=='__main__':main()
