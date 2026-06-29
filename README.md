# SA-Mem-Research

Code for VLDB Research Track. This README currently only documents the LoCoMo running scripts.

Run all commands from the project root:

```bash
cd /data/wjl/SA-Mem-Research
```

Set API configuration by environment variables or CLI flags:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"  # optional
```

If memory blocks have not been built yet, run:

```bash
python build_stage_locomo.py \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --limit-conversations -1
```

The following stages assume the build output is under `out/locomo/`.

## Retrieval

LoCoMo retrieval uses:

```bash
python retrieval/retrieve_stage_enhanced_locomo.py
```

### Baseline

Baseline is the plain vector retrieval setting.

```bash
python retrieval/retrieve_stage_enhanced_locomo.py \
  --mode baseline \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --overwrite
```

Output:

```text
out/locomo/retrieval_baseline.jsonl
out/locomo/retrieval_baseline.csv
```

### Time Filtering

Time filtering corresponds to the code's `enhanced` mode. It enables query parsing and temporal/metadata filtering.

```bash
python retrieval/retrieve_stage_enhanced_locomo.py \
  --mode enhanced \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --overwrite
```

Output:

```text
out/locomo/retrieval_enhanced.jsonl
out/locomo/retrieval_enhanced.csv
```

### Time Filtering + Graph Expansion

This uses `enhanced` mode and additionally enables graph expansion.

```bash
python retrieval/retrieve_stage_enhanced_locomo.py \
  --mode enhanced \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --graph-expand \
  --graph-min-score 0.7 \
  --graph-limit 200 \
  --graph-hops 1 \
  --overwrite \
  --output-suffix graph
```

Output:

```text
out/locomo/retrieval_enhanced_graph.jsonl
out/locomo/retrieval_enhanced_graph.csv
```

Useful retrieval options:

- `--raw-data-file`: LoCoMo data file.
- `--run-id`: output directory name under `out/`.
- `--final-content-file`: manually specify memory blocks if not using `out/<run-id>/final_boxes_content.jsonl`.
- `--output-dir`: manually specify output directory.
- `--output-suffix`: append a suffix to output filenames.
- `--limit-conversations`: run only part of the dataset.
- `--overwrite`: remove existing retrieval output before writing.

## Generation

Generate answers with the retrieval file you want to test.

### From Baseline Retrieval

```bash
python generate_stage_locomo.py \
  --run-id locomo \
  --raw-data-file /path/to/locomo.json \
  --retrieval-file out/locomo/retrieval_baseline.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --output-suffix baseline
```

### From Time Filtering Retrieval

```bash
python generate_stage_locomo.py \
  --run-id locomo \
  --raw-data-file /path/to/locomo.json \
  --retrieval-file out/locomo/retrieval_enhanced.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --output-suffix time_filtering
```

### From Time Filtering + Graph Expansion Retrieval

```bash
python generate_stage_locomo.py \
  --run-id locomo \
  --raw-data-file /path/to/locomo.json \
  --retrieval-file out/locomo/retrieval_enhanced_graph.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --use-graph-context \
  --output-suffix graph
```

Useful generation options:

- `--answer-topn`: number of retrieved memory blocks used for answering.
- `--text-modes`: usually `content`; also supports `event`, `content_trace_event`, and `trace_event`.
- `--provider openai|ollama`: choose generation provider.
- `--llm-model`, `--api-key`, `--base-url`: model/API overrides.
- `--use-graph-context`: inject graph context when using graph-expanded retrieval results.
- `--graph-context-categories`: only inject graph context for selected LoCoMo categories, e.g. `3,4`.

## Evaluation

Evaluate any generated LoCoMo result file:

```bash
python evaluate_locomo.py \
  --run-id locomo \
  --generation-file out/locomo/generation_results_locomo_time_filtering.jsonl
```

For LLM-as-judge evaluation:

```bash
python evaluate_locomo.py \
  --run-id locomo \
  --generation-file out/locomo/generation_results_locomo_time_filtering.jsonl \
  --use-llm-judge
```

Useful evaluation options:

- `--generation-file`: generation result JSONL to evaluate.
- `--sample-size`: evaluate only the first N samples.
- `--eval-output`: custom detailed evaluation output path.
- `--summary-output`: custom summary output path.

## LoCoMo B/B+TF Reproduction Notes

This branch adds a lightweight reproduction record for the non-graph LoCoMo B/B+TF path. It includes aggregate metrics, reusable scripts, and retrieval instrumentation, while leaving full raw `out/` artifacts outside the repository.

Detailed files:

- `docs/locomo_b_btf_reproduction.md`: full reproduction notes and paper comparison.
- `docs/locomo_b_btf_metrics_summary.md`: human-readable aggregate metrics.
- `docs/locomo_b_btf_metrics_summary.json`: machine-readable aggregate metrics.
- `scripts/run_locomo_b_btf.sh`: end-to-end runner.
- `scripts/retrieve_locomo_b_btf.py`: retrieval wrapper exposing top-k and method selection.
- `scripts/analyze_repro_metrics.py`: metric and latency summarizer.

### Scope

| Item | Setting |
|------|---------|
| Dataset | LoCoMo10, 10 conversations |
| QA samples | 1540 non-category-5 QA pairs |
| Methods | B and B+TF |
| Graph retrieval | Disabled |
| Retrieval top-k | 5 |
| Generation answer top-n | 5 |
| Generation context | `content` text mode |
| LLM | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |

### Key Results

| Method | F1 | BLEU | Hit@5 | Recall@5 | Complete-MRR | Core Search Mean |
|--------|----|------|-------|----------|--------------|------------------|
| B | 0.5133 | 0.3892 | 0.8201 | 0.7580 | 0.5608 | 0.1024s |
| B+TF | 0.4879 | 0.3692 | 0.7708 | 0.7065 | 0.5114 | 0.1037s |

The B baseline is close to the paper on LoCoMo QA, top-5 evidence metrics, and warm/core retrieval latency. The current public enhanced retrieval path does not reproduce a B+TF gain over B in this run.

For category-2 POINT/RANGE-triggered questions (n=40), temporal filtering does reduce the paper-like core-search boundary:

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean |
|--------|-----------|----------|----------|-------------------|--------------------|
| B | 0.1060s | 0.1053s | 0.1449s | 96.00 | 96.00 |
| B+TF | 0.0429s | 0.0085s | 0.1362s | 96.00 | 33.65 |

On all queries, this local latency gain is diluted because most category-2 questions are parsed as `NONE`, and online B+TF latency includes rewritten-query embedding plus cache flush. The report separates online wall time from the paper-like core-search timing boundary.

### Notes

The public code uses `QueryParser.rewritten_query` for enhanced retrieval. The paper mentions rewriting but does not provide the exact prompt or an isolated ablation that separates rewriting from temporal filtering. The reported B+TF numbers should therefore be read as the behavior of the public enhanced path with rewritten-query embeddings enabled. A useful follow-up is a `B+TF-original-query` ablation that keeps temporal candidate pruning but ranks with the original question embedding.
