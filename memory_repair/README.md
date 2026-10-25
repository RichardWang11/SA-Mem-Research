# TSM LoCoMo complete dense memory repair

This package contains verified construction artifacts for all 10 LoCoMo
conversations, together with the experiment code snapshot and source dataset.
It does not contain a new QA/LLM-J measurement.

See [HANDOFF_ZH.md](HANDOFF_ZH.md) for the Chinese handoff guide, executable
Neo4j import/verification commands, and both 321-question and full 1,986-question
QA entrypoints. `handoff_memory.py` restores ISO temporal strings as Neo4j
temporal properties; `run_queries.py` loads the recovered FAISS caches directly.

Code repair commit in the experiment repository:
`d7c124c1e1c53b9afab882e9d080d5eb9ba57a83`.
Official source baseline: `0da3633`, which includes official v0.1.1 (`7b45bf0`).
`code/` contains the actual local code including earlier provider/trace changes.
The previously untracked provider adapters are included explicitly.

## Construction results

- 2,871/2,871 source rounds aligned to Neo4j episode names and UUIDs.
- 317 old dense leaves replaced by 2,871 verified dense leaves/vectors.
- 100% coverage of the existing TSM loader rounds in every Conversation.
- All vector readback, nonzero/finite, self retrieval, source mapping and
  snapshot cache checks passed. Thirteen regression/release tests passed.
- Existing graph extraction was reused: 5,447 nodes, 8,490 relationships.
- Embedding: RelayRouter `text-embedding-3-small`, 1,536 dimensions, original
  unnormalised FAISS IndexFlatIP; 27 requests and 221,292 reported input tokens.

Read REPAIR_REPORT.md for the bug, completeness invariant, environment,
loader limitations, and distinction from unmodified TSM.

## Contents

Each `memory/conv-XX.tar.gz` contains:

```text
conv-XX/
  cache/
    tree.json
    dense_index.faiss
    dense_index_mapping.json
    source_mapping.json
    snapshots.json
    graph.json
    index.json
    appearance_count.json
    text_edge_mapping.json
    cluster_index.json
  neo4j_export/
    nodes.json
    relationships.json
```

The logical Neo4j export includes all properties (embeddings, episode links,
and ISO temporal fields), node labels, UUIDs, relationship types and endpoints.
It is not a physical Neo4j database dump. Restore timestamp properties as
Neo4j temporal values rather than plain strings before temporal Cypher queries.
The graph's existing extraction and temporal annotations are inherited.

`source_mapping.json` gives each leaf's global index, episode UUID/name,
session, user/assistant dialogue IDs, user-only text and full-text round.
This makes dialogue evidence -> leaf identity auditable without guessing.

Other files include `locomo10.json`, the 321-QID allowlist,
`coverage.json`, `manifest.json`, the request traces for rebuilding embeddings,
`environment.freeze.txt`, and an example `query_config.yaml` with the original
local paths. No credentials are included; paths/environment bindings must be
adapted on a different server.

Extract any Conversation using:

```bash
tar -xzf memory/conv-26.tar.gz
```

The 321 queries only restrict the answering stage. All 10 Conversation
histories were used for the existing graph and the repaired dense memories,
subject to the original loader's pairing rules.

## Attribution

TSM upstream: https://github.com/cosmicexotic/TSM (license in code/LICENSE).
LoCoMo upstream: https://github.com/snap-research/locomo.
The pinned dataset is identified by SHA-256 in manifest.json; this is a
redistributed upstream research dataset and derived memory, not newly authored
dialogue data. No new rights to the upstream data are asserted here.
