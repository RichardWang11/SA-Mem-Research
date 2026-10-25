# Stable source identity in temporal tree construction

The v0.1.1 temporal tree builder enumerated window-local episodes as `leaf_0`,
`leaf_1`, etc. and merged windows using `all_caches.update(cache)`. Leaf IDs
collided across windows. With `original_text` supplied, the local index also
selected `original_text[i]` from the global input, so a leaf could contain an
unrelated early source round.

The repair retains a complete source list, uses Graphiti's global `chunk_N`
episode name to resolve `leaf_N`, checks its content against the source, and
uses stable UUID identities for compatible unnamed episodes. Snapshot caches
contain window members with global IDs. The final tree includes every input
source, including chunks without a dated fact edge. That last completeness
invariant is explicit: it restores a full-history dense corpus, so repaired
results must be labelled as a repaired reproduction rather than unmodified TSM.

No retrieval, temporal parsing, reranking, answer prompt, embedding model,
vector normalisation, or Top-K rule is changed by this repair.

## Recover an existing run

Use `scripts/repair_dense_memory.py` with the original run config, its cache
directory containing the ten `item_*_all_sessions` folders, and a fresh output
directory. Use the previous experiment's `.venv312/bin/python`; it contains
Python 3.12.3 and the TSM dependencies. Neo4j credentials are selected via the
config's environment-variable names. The optional `--credentials-file` selects
the first RelayRouter URL/key pair used by the previous local experiment.

The script first validates ALL source/Neo4j/cache mappings, then rebuilds
FAISS through the configured embedding API. It rejects malformed, nonfinite,
or zero vectors rather than silently substituting zero embeddings. Saved
indexes are read back and compared to all vectors; self retrieval is checked.

It exports the existing Neo4j nodes and relationships including embeddings,
UUIDs, episode links, and ISO-formatted temporal fields. This is a logical JSON
export, not a Neo4j physical database dump. It corrects snapshot cache IDs
using the existing edges' `valid_at` and preserves the generated summaries.
The graph itself is not re-extracted or modified. The script does not run QA.

## Observed construction repair on 2026-10-25

The completed Qwen/SiliconFlow LoCoMo run has 10 conversations. All 2,871
source rounds matched the existing Neo4j `chunk_N` episode and cached UUID
content exactly after whitespace normalisation. The old tree contained 317
leaves in total. The recovered indexes contain 2,871 nonzero vectors.
The often quoted 11.3% was the unweighted per-conversation mean; the pooled
old leaf/source ratio is 317/2871 = 11.04%.

The logical graph export contains 5,447 nodes and 8,490 relationships.
Re-embedding made 27 requests using `text-embedding-3-small` at dimension 1536,
with 221,292 reported input tokens. No answer or extraction LLM call was made.
QA and LLM-J performance have not been remeasured by this construction run.

The source corpus follows the existing LoCoMo loader: adjacent different
speakers form a round; same-speaker positions are skipped, and a final unpaired
utterance is omitted. Completeness here means every TSM loader round, not every
raw dialogue utterance. Each source_mapping.json saves both dialogue IDs.

Regression tests exercise disjoint/overlapping monthly windows, source order
different from window order, full-text alignment, undated sources, and rejection
of inconsistent episode contents. Run:

```bash
.venv312/bin/python -m pytest -q tests/test_time_tree_source_identity.py tests/test_release_contracts.py
```

Result: 13 tests passed in the existing experiment environment.
