#!/usr/bin/env python3
"""Inspect, restore and verify the packaged TSM memory, without model calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import uuid

ROOT = Path(__file__).resolve().parent
TIME_FIELDS = {'created_at', 'valid_at', 'invalid_at', 'expired_at'}


def load_memory(conversation):
    path = ROOT / 'memory' / f'{conversation}.tar.gz'
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    expected = manifest['files'][str(path.relative_to(ROOT))]['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Archive checksum mismatch: {conversation}')
    with tarfile.open(path) as tar:
        def read(name):
            return json.load(tar.extractfile(f'{conversation}/{name}'))
        nodes = read('neo4j_export/nodes.json')
        rels = read('neo4j_export/relationships.json')
        source = read('cache/source_mapping.json')
        tree = read('cache/tree.json')
        mapping = read('cache/dense_index_mapping.json')
        vector_bytes = tar.extractfile(f'{conversation}/cache/dense_index.faiss').read()
    return nodes, rels, source, tree, mapping, vector_bytes


def transform(nodes, rels, namespace):
    if not namespace:
        return nodes, rels
    ids = {r['properties']['uuid'] for r in nodes + rels}
    groups = {r['properties']['group_id'] for r in nodes + rels if r['properties'].get('group_id')}
    replacements = {v: str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + v)) for v in ids}
    replacements.update({v: namespace + v for v in groups})
    def change(v):
        if isinstance(v, str):
            return replacements.get(v, v)
        if isinstance(v, list):
            return [change(x) for x in v]
        if isinstance(v, dict):
            return {k: change(x) for k, x in v.items()}
        return v
    return change(nodes), change(rels)


def properties(values):
    from neo4j.time import DateTime
    return {k: DateTime.from_iso_format(v) if k in TIME_FIELDS and isinstance(v, str) else v
            for k, v in values.items()}


def ident(value):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError('Invalid Neo4j label or relationship type')
    return '`' + value + '`'


def driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD']))


def verify_vectors(payload):
    import faiss
    import numpy as np
    nodes, rels, source, tree, mapping, raw = payload
    index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))
    assert len(source) == len(tree) == index.ntotal == len(mapping['doc_ids'])
    assert mapping['doc_ids'] == list(tree)
    assert mapping['tree_hash'] == hashlib.md5(json.dumps(tree, sort_keys=True).encode()).hexdigest()
    assert index.d == 1536
    vectors = index.reconstruct_n(0, index.ntotal)
    assert np.isfinite(vectors).all() and (np.linalg.norm(vectors, axis=1) > 0).all()
    scores, _ = index.search(vectors, 1)
    assert np.allclose(scores[:, 0], np.sum(vectors * vectors, axis=1), rtol=1e-4, atol=1e-4)
    episode = {n['properties']['uuid']: n['properties'] for n in nodes if 'Episodic' in n['labels']}
    assert len(episode) == len(source)
    for row in source:
        assert tree[row['leaf_id']]['text'] == row['full_text']
        assert episode[row['episode_uuid']]['name'] == row['episode_name']
        assert episode[row['episode_uuid']]['content'].strip() == row['user_only_text'].strip()
    return {'chunks': len(source), 'vectors': index.ntotal, 'nodes': len(nodes), 'relationships': len(rels)}


def restore(session, memories):
    # Check every selected group and UUID before starting any writes.
    for sid, nodes, rels in memories:
        groups = list({n['properties']['group_id'] for n in nodes})
        ids = [n['properties']['uuid'] for n in nodes]
        count = session.run('MATCH (n) WHERE n.group_id IN $groups OR n.uuid IN $ids RETURN count(n) AS n',
                            groups=groups, ids=ids).single()['n']
        if count:
            raise RuntimeError(f'{sid}: target group or UUID already exists; use a fresh database or namespace')
    for stmt in [
        'CREATE INDEX handoff_entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX handoff_episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX handoff_entity_group IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX handoff_episode_group IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX handoff_edge_valid IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.valid_at)',
        'CREATE INDEX handoff_edge_group IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.group_id)',
    ]:
        session.run(stmt).consume()
    for sid, nodes, rels in memories:
        by_labels = defaultdict(list)
        for n in nodes:
            by_labels[tuple(sorted(n['labels']))].append({'props': properties(n['properties'])})
        for labels, rows in by_labels.items():
            query = 'UNWIND $rows AS row CREATE (n:' + ':'.join(ident(x) for x in labels) + ') SET n = row.props'
            for offset in range(0, len(rows), 100):
                session.run(query, rows=rows[offset:offset+100]).consume()
        by_type = defaultdict(list)
        for r in rels:
            by_type[r['type']].append({'source': r['source'], 'target': r['target'], 'props': properties(r['properties'])})
        for kind, rows in by_type.items():
            src_label = ':Episodic' if kind == 'MENTIONS' else ':Entity'
            query = ('UNWIND $rows AS row MATCH (a' + src_label + ' {uuid: row.source}) '
                     'MATCH (b:Entity {uuid: row.target}) CREATE (a)-[r:' + ident(kind) + ']->(b) SET r = row.props')
            if kind not in {'MENTIONS', 'RELATES_TO'}:
                raise RuntimeError('This export importer expects MENTIONS and RELATES_TO only')
            for offset in range(0, len(rows), 100):
                session.run(query, rows=rows[offset:offset+100]).consume()
        print(f'{sid}: graph imported', flush=True)


def verify_graph(session, sid, nodes, rels):
    from neo4j.time import DateTime
    group = nodes[0]['properties']['group_id']
    actual_nodes = {r['p']['uuid']: r for r in session.run(
        'MATCH (n) WHERE n.group_id=$g RETURN labels(n) AS labels, properties(n) AS p', g=group)}
    actual_rels = {r['p']['uuid']: r for r in session.run(
        'MATCH (a)-[r]->(b) WHERE a.group_id=$g AND b.group_id=$g '
        'RETURN a.uuid AS source,b.uuid AS target,type(r) AS type,properties(r) AS p', g=group)}
    assert len(actual_nodes) == len(nodes), sid
    assert len(actual_rels) == len(rels), sid
    for n in nodes:
        row = actual_nodes[n['properties']['uuid']]
        assert set(row['labels']) == set(n['labels'])
        assert dict(row['p']) == properties(n['properties'])
    for r in rels:
        row = actual_rels[r['properties']['uuid']]
        assert (row['source'], row['target'], row['type']) == (r['source'], r['target'], r['type'])
        assert dict(row['p']) == properties(r['properties'])
    timed = [r for r in rels if r['type'] == 'RELATES_TO' and r['properties'].get('valid_at')]
    if timed:
        example = timed[0]
        start = properties(example['properties'])['valid_at']
        assert isinstance(actual_rels[example['properties']['uuid']]['p']['valid_at'], DateTime)
        end = start.to_native() + timedelta(seconds=1)
        rows = session.run('MATCH ()-[r:RELATES_TO]->() WHERE r.group_id=$g AND r.valid_at >= $start '
                           'AND r.valid_at < $end RETURN r.uuid AS uuid',g=group,start=start,end=end)
        assert example['properties']['uuid'] in {r['uuid'] for r in rows}
    print(f'{sid}: all graph properties, UUIDs, embeddings and temporal query PASS', flush=True)


def prepare(args, selected):
    import yaml
    import shutil
    if args.namespace:
        raise ValueError('Namespace is for isolated graph validation; QA preparation requires original exported UUIDs/groups')
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / 'query_config.yaml').read_text())
    for i, sid in enumerate(json.loads((ROOT / 'coverage.json').read_text())):
        conv = sid['conversation']
        target = args.work_dir / 'cache' / 'locomo10' / f'item_{i}_all_sessions'
        target.mkdir(parents=True)
        with tarfile.open(ROOT / 'memory' / f'{conv}.tar.gz') as tar:
            for member in tar:
                prefix = f'{conv}/cache/'
                if member.isfile() and member.name.startswith(prefix):
                    name = member.name[len(prefix):]
                    if Path(name).name != name:
                        raise RuntimeError('Unexpected archive path')
                    with tar.extractfile(member) as src, (target / name).open('wb') as dst:
                        shutil.copyfileobj(src, dst)
    config['dataset']['dataset_path'] = str(ROOT / 'locomo10.json')
    config['paths'] = {k: str(args.work_dir / v) for k,v in
                       [('cache_path','cache'),('answer_path','answers'),('log_path','logs')]}
    config['query_filter_path'] = str(ROOT / 'locomo_explicit_temporal_qids.json') if args.subset == '321' else None
    config['use_improved_paths'] = False
    config['promote_cache'] = False
    config['resume'] = {'start_index': 0, 'resumeIndex': 0}
    config['graphiti']['reset_group'] = False
    if args.llm_base_url:
        config['llm']['base_url'] = args.llm_base_url
    if args.llm_model:
        config['llm']['api_model'] = config['llm']['backend_model'] = args.llm_model
    if args.embedding_base_url:
        config['embedding']['base_url'] = args.embedding_base_url
    path = args.work_dir / 'query_config.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    print(f'Prepared {args.subset} QA config: {path}; no API calls made')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['inspect','restore','test','prepare'])
    ap.add_argument('--conversation', default='all')
    ap.add_argument('--namespace', default='')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--work-dir', type=Path)
    ap.add_argument('--subset', choices=['321','all'], default='all')
    ap.add_argument('--llm-base-url')
    ap.add_argument('--llm-model')
    ap.add_argument('--embedding-base-url')
    args = ap.parse_args()
    convs = [p.stem.removesuffix('.tar') for p in sorted((ROOT/'memory').glob('*.tar.gz'))]
    selected = convs if args.conversation == 'all' else [args.conversation]
    if not set(selected) <= set(convs):
        ap.error('Unknown conversation')
    if args.command == 'prepare':
        if not args.work_dir or args.conversation != 'all':
            ap.error('prepare requires --work-dir and all conversations')
        for sid in selected:
            verify_vectors(load_memory(sid))
        prepare(args, selected)
        return
    memories = []
    for sid in selected:
        payload = load_memory(sid)
        stats = verify_vectors(payload)
        nodes, rels = transform(payload[0], payload[1], args.namespace)
        memories.append((sid,nodes,rels))
        print(sid,stats,flush=True)
    if args.command == 'inspect':
        return
    if args.command == 'restore' and not args.execute:
        print('Dry run complete. Add --execute to import into Neo4j.')
        return
    with driver() as db, db.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as session:
        if args.command == 'restore':
            restore(session,memories)
        for sid,nodes,rels in memories:
            verify_graph(session,sid,nodes,rels)


if __name__ == '__main__':
    main()
