#!/usr/bin/env python3
"""Execute source-extracted date/leaf logic offline, without importing TSM/Graphiti."""
import ast
import json
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ast.parse((ROOT/'TSM/graphiti_extractor.py').read_text())
    fn=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='_reference_time')
    fn.decorator_list=[]
    env={'re':re,'datetime':datetime,'timedelta':timedelta,'timezone':timezone}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'date_source','exec'),env)
    result=env['_reference_time']('Date: 7:55 pm on 9 June, 2023\nUser: example',22).isoformat()
    assert result=='2000-01-01T00:00:22+00:00'
    # Reproduce the archived pre-repair upstream issue, even after the active
    # TSM snapshot has received the source-identity repair.
    historical = ROOT/'patches/historical_tree_builder.py.txt'
    tree=ast.parse((historical if historical.exists() else ROOT/'TSM/build_tree_clustering_time.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(i, chunk_content)' and any(isinstance(x,ast.Name) and x.id=='leaf_text' for x in ast.walk(n)))
    class Log:
        def debug(self,*args):pass
    merged={};windows=[]
    for contents in [['WINDOW_A_EPISODE_3','WINDOW_A_EPISODE_4'],['WINDOW_B_EPISODE_2']]:
        context={'text_chunks':contents,'original_text':[{'full_text':f'GLOBAL_ROUND_{i}'} for i in range(5)],'cache':{},'logger':Log()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'leaf_source','exec'),context)
        merged.update(context['cache'])
        windows.append({'input':contents,'saved':context['cache']})
    assert len(merged)==2
    assert merged['leaf_0']['text']=='GLOBAL_ROUND_0'
    print(json.dumps({'reference_time':result,'window_results':windows,'merged_leaf_count':len(merged),'interpretation':'Known upstream defects reproduced, NOT an acceptance pass for memory quality.'},indent=2))


if __name__=='__main__':
    main()
