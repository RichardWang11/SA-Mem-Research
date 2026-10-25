"""Regression: window-local indices must never replace global source identity."""
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import build_tree_clustering_time as builder


@pytest.mark.parametrize('original_text', [None, [{'full_text': f'full-{i}'} for i in range(5)]])
def test_month_windows_keep_global_source_identity_and_untimed_source(tmp_path, original_text):
    source = [f'user-{i}' for i in range(5)]
    windows = [(datetime(2023, m, 1, tzinfo=timezone.utc),
                datetime(2023, m+1, 1, tzinfo=timezone.utc)) for m in [1, 2]]
    contents = [
        {'u3': {'name': 'chunk_3', 'content': source[3]}},
        {'u1': {'name': 'chunk_1', 'content': source[1]},
         'u3': {'name': 'chunk_3', 'content': source[3]}},
    ]
    with patch.multiple(builder,
                        get_time_range_from_graph=lambda *a: (windows[0][0], windows[-1][1]),
                        split_time_range_by_month=lambda *a: windows,
                        get_entities_with_embeddings=lambda *a: ([{'uuid': 'entity', 'name': 'E', 'summary': 's'}], [[1., 0.]]),
                        get_entity_chunk_mapping=lambda *a: {},
                        get_edges_with_episodes=lambda *a: [],
                        perform_clustering=lambda *a, **k: [[0]],
                        _call_llm_unified=lambda *a, **k: 'offline summary'), \
         patch.object(builder, 'get_episode_contents', side_effect=contents):
        tree, _, _ = builder.build_tree_with_clustering(
            source, {}, str(tmp_path), object(), 'test', original_text=original_text)
    assert set(tree) == {f'leaf_{i}' for i in range(5)}
    expected = [f'full-{i}' if original_text else source[i] for i in range(5)]
    assert [tree[f'leaf_{i}']['text'] for i in range(5)] == expected
    snapshots = json.loads((tmp_path / 'snapshots.json').read_text())
    assert set(snapshots[0]['cache']) == {'leaf_3'}
    assert set(snapshots[1]['cache']) == {'leaf_1', 'leaf_3'}


def test_named_episode_with_wrong_source_is_rejected(tmp_path):
    windows = [(datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 2, 1, tzinfo=timezone.utc))]
    with patch.multiple(builder,
                        get_time_range_from_graph=lambda *a: windows[0],
                        split_time_range_by_month=lambda *a: windows,
                        get_entities_with_embeddings=lambda *a: ([{'uuid': 'e', 'name': 'E'}], [[1., 0.]]),
                        get_entity_chunk_mapping=lambda *a: {},
                        get_edges_with_episodes=lambda *a: [],
                        get_episode_contents=lambda *a: {'u': {'name': 'chunk_0', 'content': 'wrong'}}):
        with pytest.raises(RuntimeError, match='Episode/source mismatch'):
            builder.build_tree_with_clustering(['correct'], {}, str(tmp_path), object(), 'test')
