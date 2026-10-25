#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
rows={}
for root in ['precomputed','conversations']:
    for p in (args.run/root).glob('*/answers/Locomo/*results.json'):
        try:obj=json.loads(p.read_text())
        except json.JSONDecodeError:continue
        for values in obj.values():
            if isinstance(values,list):
                for row in values:rows[row['question_id']]=row
counts={}
for qid in rows:
    conv=qid.rsplit('_',1)[0];counts[conv]=counts.get(conv,0)+1
data={'time':datetime.now(timezone.utc).isoformat(),'qa_unique_completed':len(rows),
      'empty_contexts':sum(not q.get('retrieved_evidences') for q in rows.values()),
      'api_error_answers':sum(q.get('hypothesis')=='API调用失败' for q in rows.values()),
      'by_conversation':dict(sorted(counts.items()))}
judges=list((args.run/'judge/mem0_gpt4omini_all/results').glob('*.json'))
data['judge_completed']=sum(json.loads(p.read_text()).get('status')=='completed' for p in judges)
(args.run/'live_progress.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,ensure_ascii=False))
