#!/usr/bin/env python3
"""Recover source leaves, export the existing TKG, and rebuild FAISS.

Requires the original config and completed cache, a NEW output directory, and
read access to Neo4j. No graph construction or answer LLM calls are made.
Embeddings use the configured model and the original unnormalised IndexFlatIP.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=serialise) + '\n')


def serialise(value):
    if hasattr(value, 'iso_format'):
        return value.iso_format()
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def source_chunks(piece):
    chunks = []
    for session in piece['rounds']:
        for r in session:
            user = f"Date: {r.get('date', '')}\nUser: {r['user']['content']}"
            chunks.append({
                'user_only_text': user,
                'full_text': user + '\nAssistant: ' + r['assistant']['content'],
                'session_id': r['session_id'],
                'user_dialogue_id': r['user']['dia_id'],
                'assistant_dialogue_id': r['assistant']['dia_id'],
            })
    return chunks


async def run(args):
    import faiss
    import numpy as np
    import yaml
    from neo4j import GraphDatabase
    from dataloader import locomoLoader
    from build_dense_index import get_tree_hash, save_dense_index_cache
    from provider.public_embedder import PublicEmbedder

    config = yaml.safe_load(args.config.read_text())
    dataset_path = Path(config['dataset']['dataset_path'])
    loader = locomoLoader(str(dataset_path))
    args.output.mkdir(parents=True, exist_ok=False)
    date_tag = datetime.now(timezone.utc).strftime('%Y%m%d')
    write(args.output / 'status.json', {'stage': 'validating', 'completed': 0})
    embed_config = config['embedding']
    if args.credentials_file:
        # Match the credential selector used by the previous experiment;
        # never persist credential values or print the file contents.
        lines = args.credentials_file.read_text().splitlines()
        if lines[0].strip().rstrip('/') != embed_config['base_url'].rstrip('/'):
            raise RuntimeError('Credential endpoint differs from configured embedding endpoint')
        os.environ[embed_config.get('api_key_env', 'OPENAI_API_KEY')] = lines[1].strip()
    os.environ['TSM_LLM_TRACE_DIR'] = str(args.output / 'request_trace')
    neo = config['graphiti']['neo4j']
    driver = GraphDatabase.driver(
        os.getenv(neo.get('uri_env', 'NEO4J_URI'), neo.get('uri', 'bolt://localhost:7687')),
        auth=(os.getenv(neo.get('user_env', 'NEO4J_USER'), 'neo4j'),
              os.environ[neo.get('password_env', 'NEO4J_PASSWORD')]),
    )
    database = os.getenv(neo.get('database_env', 'NEO4J_DATABASE'), 'neo4j')
    counts = []
    try:
        for i in range(len(loader)):
            piece = loader[i]
            sid = piece['group_id']
            gid = config['graphiti']['group_id_prefix'] + sid
            old = args.old_cache / loader.available_ids[i]
            target = args.output / 'cache' / dataset_path.stem / date_tag / loader.available_ids[i]
            target.mkdir(parents=True)
            chunks = source_chunks(piece)
            tree = {f'leaf_{j}': {'text': r['full_text'], 'children': None, 'parent': None}
                    for j, r in enumerate(chunks)}
            with driver.session(database=database) as session:
                nodes = [dict(r) for r in session.run(
                    'MATCH (n) WHERE n.group_id=$gid RETURN labels(n) AS labels, properties(n) AS properties', gid=gid)]
                relationships = [dict(r) for r in session.run(
                    'MATCH (a)-[r]->(b) WHERE a.group_id=$gid AND b.group_id=$gid '
                    'RETURN a.uuid AS source, b.uuid AS target, type(r) AS type, properties(r) AS properties', gid=gid)]
            episodes = [n['properties'] for n in nodes if 'Episodic' in n['labels']]
            if len(episodes) != len(chunks):
                raise RuntimeError(f'{sid}: episode/source count mismatch')
            mapping = json.loads((old / 'text_edge_mapping.json').read_text())
            provenance = []
            uuid_to_leaf = {}
            seen_indices = set()
            for ep in episodes:
                name = ep['name']
                if not name.startswith('chunk_'):
                    raise RuntimeError(f'{sid}: nonstandard episode name')
                j = int(name[6:])
                if not 0 <= j < len(chunks) or j in seen_indices:
                    raise RuntimeError(f'{sid}: invalid/duplicate source index')
                seen_indices.add(j)
                uuid = ep['uuid']
                source = chunks[j]
                if ep['content'].strip() != source['user_only_text'].strip():
                    raise RuntimeError(f'{sid}/{name}: source/Neo4j text mismatch')
                if uuid not in mapping or mapping[uuid]['text'].strip() != source['user_only_text'].strip():
                    raise RuntimeError(f'{sid}/{name}: source/cache text mismatch')
                uuid_to_leaf[uuid] = f'leaf_{j}'
                provenance.append({'leaf_id': f'leaf_{j}', 'episode_uuid': uuid,
                                   'episode_name': name, 'global_index': j, **source})
            write(target / 'tree.json', tree)
            write(target / 'source_mapping.json', sorted(provenance, key=lambda p: p['global_index']))
            for filename in ['graph.json', 'index.json', 'appearance_count.json',
                             'text_edge_mapping.json', 'cluster_index.json']:
                shutil.copy2(old / filename, target / filename)
            snapshots = json.loads((old / 'snapshots.json').read_text())
            for snapshot in snapshots:
                start = datetime.fromisoformat(snapshot['valid_from'])
                end = datetime.fromisoformat(snapshot['valid_to'])
                ids = set()
                for rel in relationships:
                    if rel['type'] != 'RELATES_TO':
                        continue
                    props = rel['properties']
                    valid = props.get('valid_at')
                    if valid is not None and start <= datetime.fromisoformat(serialise(valid)) < end:
                        ids.update(uuid_to_leaf[u] for u in props.get('episodes', []) if u in uuid_to_leaf)
                snapshot['cache'] = {k: tree[k] for k in sorted(ids, key=lambda x: int(x[5:]))}
            write(target / 'snapshots.json', snapshots)
            export = args.output / 'neo4j_export' / sid
            write(export / 'nodes.json', nodes)
            write(export / 'relationships.json', relationships)
            counts.append({'conversation': sid, 'group_id': gid, 'cache': str(target),
                           'old_leaf_count': len(json.loads((old / 'tree.json').read_text())),
                           'source_count': len(chunks), 'episode_count': len(episodes),
                           'mapping_verified': len(provenance), 'nodes': len(nodes),
                           'relationships': len(relationships)})
            print(f'[validated {i+1}/{len(loader)}] {sid}: {len(chunks)} sources, all UUIDs aligned', flush=True)
    finally:
        driver.close()

    # All ten provenance checks finish before the first paid embedding call.
    write(args.output / 'coverage.json', counts)
    shutil.copy2(dataset_path, args.output / dataset_path.name)
    query_filter = Path(config['query_filter_path'])
    shutil.copy2(query_filter, args.output / query_filter.name)
    embedder = PublicEmbedder.from_config(embed_config)
    try:
        for i, item in enumerate(counts):
            target = Path(item['cache'])
            tree = json.loads((target / 'tree.json').read_text())
            ids = list(tree)
            docs = [tree[k]['text'] for k in ids]
            vectors = []
            for offset in range(0, len(docs), embed_config.get('batch_size', 128)):
                batch = docs[offset:offset + embed_config.get('batch_size', 128)]
                result = await embedder.embed_texts(batch)
                if len(result) != len(batch):
                    raise RuntimeError('Embedding response count mismatch')
                vectors.extend(result)
            array = np.ascontiguousarray(vectors, dtype=np.float32)
            if array.shape != (len(ids), embed_config['dimension']):
                raise RuntimeError(f'Invalid embedding shape: {array.shape}')
            if not np.isfinite(array).all() or not (np.linalg.norm(array, axis=1) > 0).all():
                raise RuntimeError('Nonfinite or zero embedding: refusing index creation')
            index = faiss.IndexFlatIP(array.shape[1])
            index.add(array)
            save_dense_index_cache(str(target), index, ids, get_tree_hash(tree))
            check = faiss.read_index(str(target / 'dense_index.faiss'))
            if check.ntotal != len(docs) or not np.array_equal(check.reconstruct_n(0, check.ntotal), array):
                raise RuntimeError('Saved index readback failed')
            scores, ranks = check.search(array[:1], min(20, len(ids)))
            if ranks[0, 0] != 0:
                raise RuntimeError('Self retrieval smoke failed')
            item.update({'faiss_docs': check.ntotal, 'dimension': check.d, 'self_search_passed': True})
            write(args.output / 'status.json', {'stage': 'embedding', 'completed': i+1, 'total': len(counts)})
            write(args.output / 'coverage.json', counts)
            print(f'[indexed {i+1}/{len(counts)}] {item["conversation"]}: {check.ntotal} vectors, readback/self-search PASS', flush=True)
    finally:
        if embedder.client is not None:
            await embedder.client.close()

    # Query-only config targets this isolated cache; keep the existing graph groups.
    config['paths'] = {k: str(args.output / v) for k,v in
                       [('cache_path', 'cache'), ('answer_path', 'answers'), ('log_path', 'logs')]}
    config['graphiti']['reset_group'] = False
    config['promote_cache'] = False
    (args.output / 'query_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write(args.output / 'status.json', {'stage': 'completed', 'completed': len(counts), 'total':len(counts)})
    files = {str(p.relative_to(args.output)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(args.output.rglob('*')) if p.is_file()}
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'working_diff_sha256': hashlib.sha256(subprocess.check_output(['git','diff'],cwd=ROOT)).hexdigest(),
                'dataset_sha256': sha(dataset_path), 'embedding': embed_config,
                'source_config_sha256': sha(args.config), 'conversations': counts,
                'source_total': sum(x['source_count'] for x in counts),
                'faiss_total': sum(x['faiss_docs'] for x in counts),
                'graph_reextracted': False, 'qa_run': False, 'files': files}
    write(args.output / 'manifest.json', manifest)
    print(f'Completed: {manifest["faiss_total"]} verified vectors. {args.output}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--old-cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--credentials-file', type=Path)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
