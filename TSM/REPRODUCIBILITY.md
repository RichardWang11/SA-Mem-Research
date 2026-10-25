# Reproducibility protocol

This protocol separates three evidence levels:

1. Offline engineering verification: no network, no benchmark content, no Neo4j requirement.
2. Public-backend benchmark run: sends dataset-derived text to the explicitly approved LLM and embedding endpoints.
3. Official score: produced by the dataset maintainers' evaluator, with its exact revision recorded.

Passing level 1 does not establish a benchmark result. A historical internal-backend score is not a public reproduction.

## Release acceptance record

The maintainer's Level 1 acceptance run on 2026-09-01 produced:

- Python compilation: passed.
- Configuration validation: 10/10 passed.
- Release leak scan: 0 findings.
- Contract tests: 10/10 passed.
- Offline smoke SHA256: `76f99f1c0a7c458773dd5c4692e193f32aff4b6deb323878218687e518908115`.
- Synthetic end-to-end questions: 1/1, result SHA256 `0281bce7004a92c0de20a75aee2cd80f8718320da3b3395857940d3bb5f0a6a6`.

No live paid API request was made for this release. Levels 2 and 3 remain pending and must not be inferred from the Level 1 record.

## Fixed release contract

- Python: 3.11.
- TSM version: `0.1.1`.
- Graphiti: `graphiti-core==0.19.0`.
- Neo4j image: `neo4j:5.26-community`.
- Random seeds in clustering code: `42`; legacy retriever seed: `1`.
- LLM temperature: `0` in all 10 configs.
- Retrieval cap: 20 chunks.
- FAISS: CPU in public configs.
- Dataset bytes must match `data/checksums.sha256`.
- Configs must pass `scripts/validate_configs.py`.
- A configured method must fail closed; it must not silently fall back to another method.

Temperature zero improves repeatability but does not guarantee identical responses from all hosted APIs. Archive raw responses, manifests, and provider revision/model identifiers when available.

## Level 1: clean CPU/no-network acceptance

From a clean checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m compileall -q .
python scripts/validate_configs.py
python scripts/scan_release.py
pytest -q
python scripts/verify_offline_repro.py
python scripts/synthetic_e2e.py
```

Acceptance criteria:

- 10/10 configs validate.
- All Python sources compile.
- Leak scan reports 0 findings.
- All tests pass.
- Two normalized offline smoke outputs are byte-identical and one SHA256 is printed.
- One fully synthetic pure-RAG item completes data loading, indexing, retrieval, generation, and result persistence through a local fake endpoint.

The smoke uses synthetic text and deterministic hash embeddings. Hash embeddings are never valid for benchmark reporting.

## Level 2: public-backend run

1. Download and verify data.
2. Start an empty Neo4j database.
3. Choose one immutable LLM model identifier and one immutable embedding model identifier.
4. Print and review endpoints/models with `--dry-run`.
5. Obtain explicit approval for sending LongMemEval/LoCoMo-derived content to both endpoints.
6. Run with `--acknowledge-data-transmission --force-rebuild`.
7. Repeat for all 10 configs without sharing caches across methods.

The required environment variables are exactly:

- `OPENAI_API_KEY`
- `TSM_LLM_BASE_URL`
- `TSM_EMBEDDING_BASE_URL`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

The runner records only whether credentials are present, never their values.

## Level 3: official scoring

For each run archive:

- `run_manifest.json`
- `config.resolved.yaml`
- raw result JSON
- exported prediction JSONL
- official evaluator repository URL and commit SHA
- official evaluator command
- evaluator stdout/stderr and final metric file

The included `evaluate_predictions.py` computes diagnostic exact match and token F1 only. Do not label those values as official benchmark accuracy.

## Reproduction report template

Record one row per dataset/method:

| Dataset | Method | Run ID | Dataset SHA256 | LLM endpoint/model | Embedding endpoint/model | Code SHA | Official evaluator SHA | Metric | Status |
|---|---|---|---|---|---|---|---|---:|---|

Statuses are `offline-verified`, `public-run-complete`, `officially-scored`, or `failed`. Never replace a failed method with a fallback result.
