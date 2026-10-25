#!/usr/bin/env python3
"""Archive completed real QA and before/after judge evidence for handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

SCRIPTS=Path(__file__).resolve().parent
BASE=Path('/workspace/SA-mem/tsm_acceptance_20261023/judge/mem0_gpt4omini_all_20261025')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert json.loads((args.run/'status.json').read_text())['stage']=='completed'
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['REPORT.md','manifest.json','status.json','comparison.json','evidence_samples.json',
                 'evidence_summary.json','label_changes.json','failure_cases.json','usage_and_failures.json',
                 'baseline_judge_summary_reaudited.json','repaired_judge_summary_reaudited.json']:
        shutil.copy2(args.run/name,args.output/name)
    with tarfile.open(args.output/'qa_and_request_trace.tar.gz','w:gz') as tar:
        for name in ['answers','conversations','smoke','code_snapshot','lexical','judge_console.log']:
            tar.add(args.run/name,arcname=name,filter=lambda m: None if '__pycache__' in Path(m.name).parts else m)
    with tarfile.open(args.output/'judge_after.tar.gz','w:gz') as tar:
        tar.add(args.run/'judge/mem0_gpt4omini_all',arcname='judge_after')
    with tarfile.open(args.output/'judge_before.tar.gz','w:gz') as tar:
        tar.add(BASE,arcname='judge_before')
    dst=args.output/'scripts';dst.mkdir()
    for name in ['run_repaired_qa_comparison.py','precompute_repaired_qa.py','progress_repaired_qa.py',
                 'analyze_repaired_qa.py','compare_repaired_judges.py','write_repaired_qa_report.py',
                 'package_repaired_qa_results.py']:
        shutil.copy2(SCRIPTS/name,dst/name)
    shutil.copy2(Path('/workspace/SA-mem/tsm_acceptance_20261023/scripts/evaluate_llm_judge.py'),dst/'evaluate_llm_judge.py')
    prompt=Path('/workspace/SA-mem/tsm_acceptance_20261023/judge/prompts/mem0_locomo.txt')
    shutil.copy2(prompt,args.output/'judge_prompt.txt')
    hashes={str(p.relative_to(args.output)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
            for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'files_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
