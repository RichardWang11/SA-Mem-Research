from pathlib import Path

import yaml

from scripts.offline_smoke import normalized_result
from scripts.download_data import DATASETS
from scripts.validate_configs import ROOT, validate_matrix


def test_public_matrix_has_two_datasets_and_five_methods_each():
    paths, errors = validate_matrix()
    assert errors == []
    assert len(paths) == 10
    assert {path.parent.name for path in paths} == {"longmemeval", "locomo"}
    assert {path.stem for path in paths} == {
        "tsm",
        "no_temporal",
        "no_persona",
        "traditional",
        "pure_rag",
    }


def test_full_tsm_uses_time_entrypoint_and_snapshot_contract():
    for dataset in ("longmemeval", "locomo"):
        config = yaml.safe_load((ROOT / "configs" / dataset / "tsm.yaml").read_text())
        assert config["experiment"]["entrypoint"] == "main_time.py"
        assert config["retriever"]["kwargs"]["enable_temporal_filter"] is True
        assert config["retriever"]["kwargs"]["chunks_only"] is False
        assert config["entity_clustering"]["enable"] is True


def test_pure_rag_contract_has_no_graph_or_summary_retrieval():
    for dataset in ("longmemeval", "locomo"):
        config = yaml.safe_load((ROOT / "configs" / dataset / "pure_rag.yaml").read_text())
        assert config["extraction"]["method"] == "none"
        assert config["entity_clustering"]["enable"] is False
        assert config["retriever"]["kwargs"]["chunks_only"] is True
        assert config["retriever"]["kwargs"]["enable_temporal_filter"] is False


def test_offline_smoke_is_deterministic_in_process():
    first = normalized_result()
    second = normalized_result()
    assert first == second
    assert first["embedding_shape"] == [2, 32]
    assert first["selected_snapshots"] == ["feb"]
    assert first["temporally_filtered_chunks"] == {"dense": ["leaf_1"]}


def test_release_root_is_resolved():
    assert Path(ROOT, "main.py").is_file()


def test_every_config_instantiates_the_declared_entrypoint_bundle():
    from main import extract_config_bundle as extract_main_bundle
    from main_time import extract_config_bundle as extract_time_bundle

    paths, _ = validate_matrix()
    for path in paths:
        config = yaml.safe_load(path.read_text())
        extractor = (
            extract_time_bundle
            if config["experiment"]["entrypoint"] == "main_time.py"
            else extract_main_bundle
        )
        bundle = extractor(config)
        assert bundle.dataset_name in {"LongMemEval", "Locomo"}
        assert bundle.llm_api["type"] == "openai_compatible"
        assert bundle.embedding_config["backend"] == "openai_compatible"


def test_pure_rag_leaf_builder_has_no_summary_nodes(tmp_path):
    from main import build_leaf_only_tree

    tree, _ = build_leaf_only_tree(
        [
            {"full_text": "Date: 2025-01-01\nUser: synthetic\nAssistant: synthetic"},
            {"full_text": "Date: 2025-01-02\nUser: fictional\nAssistant: fictional"},
        ],
        str(tmp_path),
    )
    assert list(tree) == ["leaf_0", "leaf_1"]
    assert all(node["children"] == [] and node["parent"] is None for node in tree.values())


def test_synthetic_files_match_both_dataset_loaders():
    from dataloader import locomoLoader, longmemevalLoader

    longmem = longmemevalLoader(str(ROOT / "data/smoke/longmemeval_synthetic.json"))
    locomo = locomoLoader(str(ROOT / "data/smoke/locomo_synthetic.json"))
    assert (len(longmem), len(longmem[0]["rounds"]), len(longmem[0]["qa"])) == (1, 1, 1)
    assert (len(locomo), len(locomo[0]["rounds"]), len(locomo[0]["qa"])) == (1, 1, 1)


def test_benchmark_downloads_are_revision_pinned():
    longmem = DATASETS["longmemeval"]
    locomo = DATASETS["locomo"]
    assert longmem["sha256"] == "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    assert all("98d7416c24c778c2fee6e6f3006e7a073259d48f" in url for url in longmem["urls"])
    assert locomo["sha256"] == "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
    assert all("3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376" in url for url in locomo["urls"])


def test_citation_metadata_has_paper_and_no_placeholder():
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation = yaml.safe_load(citation_text)
    assert "TSM Contributors" not in citation_text
    assert len(citation["authors"]) == 11
    assert citation["preferred-citation"]["doi"] == "10.18653/v1/2026.findings-acl.1496"
