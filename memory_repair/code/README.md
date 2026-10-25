# TSM: Temporal Semantic Memory

[![CI](https://github.com/cosmicexotic/TSM/actions/workflows/ci.yml/badge.svg)](https://github.com/cosmicexotic/TSM/actions/workflows/ci.yml)

TSM is a research implementation of temporal semantic memory for long-horizon conversational question answering. This public release contains the complete temporal/snapshot path, four declared ablations, public OpenAI-compatible providers, and reproducibility tooling for LongMemEval and LoCoMo.

Status: `0.1.1`. Level 1 offline reproducibility is verified. No live API-backed benchmark run has been performed for this public release, and benchmark scores are intentionally not claimed here until all 10 public-backend configurations have been rerun.

## Provenance

TSM is derived from [YiboZhao624/E-2GraphRAG](https://github.com/YiboZhao624/E-2GraphRAG), released under the MIT License. The complete TSM method uses `main_time.py`, `query_refactored.py`, and `build_tree_clustering_time.py`. The non-temporal and ablation paths use `main.py`.

The release was built as a new repository without the development repository's Git history. See [NOTICE](NOTICE) and [LICENSE](LICENSE).

## Experiment matrix

| Method | Entrypoint | Snapshot/persona | Graph | Temporal filter | Retrieval corpus |
|---|---|---:|---:|---:|---|
| `tsm` | `main_time.py` | yes | yes | yes | chunks + selected summaries |
| `no_temporal` | `main.py` | persona only | yes | no | chunks + summaries |
| `no_persona` | `main.py` | built but filtered at retrieval | yes | yes | chunks only |
| `traditional` | `main.py` | no persona | no | no | chunk clusters + summaries |
| `pure_rag` | `main.py` | no | no | no | raw chunks only |

Each method has one configuration under `configs/longmemeval/` and one under `configs/locomo/`. Run `python scripts/validate_configs.py` to verify all 10 contracts.

## Quick start

Python 3.11 is the supported environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python scripts/validate_configs.py
python scripts/verify_offline_repro.py
pytest -q
```

Obtain the datasets without committing them:

```bash
python scripts/download_data.py all
shasum -a 256 -c data/checksums.sha256
```

Start Neo4j:

```bash
cp .env.example .env
# Edit .env, then:
docker compose up -d neo4j
```

Set the public endpoints and credentials in your shell. Configuration files contain no credentials.

```bash
# Enter the key without writing it into shell history, then press Return.
read -s OPENAI_API_KEY
export OPENAI_API_KEY
export TSM_LLM_BASE_URL='https://api.openai.com/v1'
export TSM_EMBEDDING_BASE_URL='https://api.openai.com/v1'
export NEO4J_URI='bolt://localhost:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='...'
export NEO4J_DATABASE='neo4j'
```

Inspect a run before transmitting benchmark content:

```bash
python scripts/run_experiment.py \
  --config configs/longmemeval/tsm.yaml \
  --dry-run
```

The command prints the exact LLM endpoint/model and embedding endpoint/model. Only after those are explicitly approved, run:

```bash
python scripts/run_experiment.py \
  --config configs/longmemeval/tsm.yaml \
  --acknowledge-data-transmission \
  --force-rebuild
```

Every run writes `runs/<run-id>/run_manifest.json` with code revision, dependency versions, redacted configuration, dataset checksum, endpoint/model identities, command, timestamps, and exit status.

After approval, run the complete matrix with:

```bash
python scripts/run_matrix.py \
  --dataset all \
  --acknowledge-data-transmission \
  --force-rebuild
```

## Evaluation

The local evaluator provides deterministic diagnostics and an export for upstream evaluators:

```bash
python scripts/evaluate_predictions.py path/to/results.json \
  --output path/to/diagnostic_metrics.json \
  --export-jsonl path/to/predictions.jsonl
```

Its exact match and token F1 are explicitly marked diagnostic. Report official LongMemEval/LoCoMo scores only with the corresponding official evaluator and archive that evaluator's commit plus output in the run directory.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full acceptance protocol.

## Repository layout

```text
configs/                 2 x 5 public experiment matrix
data/                    download instructions, checksums, synthetic smoke data
provider/                public LLM and embedding adapters
scripts/                 download, validation, run, manifest, evaluation, leak scan
tests/                   offline and mechanism-contract tests
main_time.py             full TSM temporal/snapshot entrypoint
main.py                  non-temporal and ablation entrypoint
```

## Citation

Copy the official ACL Anthology entry below into your `.bib` file. The same entry is available as [CITATION.bib](CITATION.bib); GitHub citation metadata is provided in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{su-etal-2026-beyond,
    title = "Beyond Dialogue Time: Temporal Semantic Memory for Personalized {LLM} Agents",
    author = "Su, Miao  and
      Guo, Yucan  and
      Hou, Zhongni  and
      Bai, Long  and
      Li, Zixuan  and
      Zhang, Yufei  and
      Yin, Guojun  and
      Lin, Wei  and
      Jin, Xiaolong  and
      Guo, Jiafeng  and
      Cheng, Xueqi",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1496/",
    doi = "10.18653/v1/2026.findings-acl.1496",
    pages = "29935--29951",
    ISBN = "979-8-89176-395-1"
}
```
