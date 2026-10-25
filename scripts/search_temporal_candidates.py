"""Read-only whole-run graph/time search for outside-top20 source evidence."""
import os,json,re,sys
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase

from nltk.stem import PorterStemmer
ps=PorterStemmer();stop=set('what which when where who how did does do was were is are the a an in on at to of for and or have has had during before after as with their his her he she it they that this from by recently'.split())
def tokens(s):return {ps.stem(x) for x in re.findall('[a-z0-9]+',s.lower()) if x not in stop}
def norm(s):return ' '.join(s.split())
def dt(s):return datetime.fromisoformat(str(s).replace('Z','+00:00'))
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--group-prefix',default='locomo_tsm_siliconflow_qwen_full_tmux_20261021_')
args=parser.parse_args()
run=args.run.resolve();root=args.out.resolve()
if root.exists(): raise SystemExit('Output directory exists')
(root/'artifacts').mkdir(parents=True)
res=json.load(open(run/'answers/Locomo/locomo10_split_round_based_c6dd0749_results.json'));out=[];groups=[]
d=GraphDatabase.driver(os.environ.get('NEO4J_URI','bolt://localhost:7687'),auth=(os.environ.get('NEO4J_USER','neo4j'),os.environ['NEO4J_PASSWORD']))
with d.session(database='neo4j',default_access_mode='READ') as s:
 for item,qs in res.items():
  if not isinstance(qs,list):continue
  scope=qs[0]['question_id'].rsplit('_',1)[0];g=args.group_prefix+scope
  edges=[dict(x) for x in s.run('MATCH (a)-[e:RELATES_TO]->(b) WHERE e.group_id=$g UNWIND e.episodes AS id MATCH (ep:Episodic {uuid:id}) WHERE ep.group_id=$g RETURN e.uuid AS edge_uuid,e.fact AS fact,e.valid_at AS valid_at,e.invalid_at AS invalid_at,e.created_at AS created_at,ep.name AS episode_name,ep.content AS content',g=g)]
  groups.append({'scope':scope,'edge_episode_rows':len(edges)})
  tree=json.load(open(run/f'cache/locomo10/20261021/{item}/tree.json'))
  for q in qs:
   tr=q['retrieval_trace'];ranges=tr['parsed_time_ranges']
   if not ranges:continue
   order=list(dict.fromkeys(tr['dense_top100_ids']));top=set(tr['dense_top20_ids']);matched={}
   for e in edges:
    v=e['valid_at'] if e['valid_at'] is not None else e['created_at']
    if v is None:continue
    if not any(dt(z['start'])<=dt(v)<=dt(z['end']) for z in ranges):continue
    ep=e['episode_name'];lid=ep.replace('chunk_','leaf_',1)
    if not ep.startswith('chunk_') or lid not in tree or norm(e['content']) not in norm(tree[lid]['text']):continue
    matched.setdefault(lid,[]).append(e)
   filtered=[lid for lid in order if lid in matched]
   for lid in filtered:
    if lid in top:continue
    txt=tree[lid]['text'];qt=tokens(q['question']);at=tokens(str(q['answer']));lt=tokens(txt)
    out.append({'qid':q['question_id'],'question':q['question'],'gold_answer':q['answer'],'leaf_id':lid,'dense_rank':order.index(lid)+1,'temporal_filtered_rank':filtered.index(lid)+1,'within_temporal_top20':filtered.index(lid)<20,'text':txt,'time_matched_edges':matched[lid],'query_token_overlap':len(qt&lt)/max(1,len(qt)),'answer_token_overlap':len(at&lt)/max(1,len(at)),'query_ranges':ranges})
d.close()
(root/'artifacts/all_temporal_outside20_candidates.json').write_text(json.dumps({'groups':groups,'candidates':out},ensure_ascii=False,indent=2,default=str))
print('candidate_rows',len(out),'queries',len({x['qid'] for x in out}))
# Print highest answer-overlap item per question; only substantial answer matches.
best={}
for x in out:
 if not str(x['gold_answer']).strip():continue
 if x['answer_token_overlap']<0.5:continue
 key=x['qid']
 if key not in best or (x['answer_token_overlap'],x['query_token_overlap'])>(best[key]['answer_token_overlap'],best[key]['query_token_overlap']):best[key]=x
for x in sorted(best.values(),key=lambda x:(-x['answer_token_overlap'],-x['query_token_overlap'])):
 print('\n',x['qid'],x['question'],'GOLD',x['gold_answer'],'rank',x['dense_rank'],'->',x['temporal_filtered_rank'],'overlap',x['answer_token_overlap'])
 print(x['text'][:1000])
