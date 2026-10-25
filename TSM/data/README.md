# Data

Benchmark data is not redistributed in this repository.

- LongMemEval: obtain the official cleaned `longmemeval_s_cleaned.json` from the [LongMemEval dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) at revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`. The downloader stores it as `data/longmemeval/longmemeval_s.json` for compatibility with the original TSM configuration.
- LoCoMo: obtain `locomo10.json` from the [official LoCoMo repository](https://github.com/snap-research/locomo) and place it at `data/locomo/locomo10.json`.

The reproducible download path is:

```bash
python scripts/download_data.py all
shasum -a 256 -c data/checksums.sha256
```

The two files in `data/smoke/` are fully synthetic and are used only for loader and offline tests. They are not benchmark samples and must not be used for reporting accuracy.

Dataset access and use remain subject to the terms published by the respective dataset maintainers.
