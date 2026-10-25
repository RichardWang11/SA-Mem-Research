from extract_graph import build_graph


def test_legacy_weights_still_merge():
    graph = build_graph([['a','b',2],['b','a',3]])
    assert graph['a']['b'] == {'weight':5}


def test_graphiti_attributes_restore_without_mutating_cache():
    attrs = {'weight':1.0,'fact':'a fact','relation':'KNOWS'}
    graph = build_graph([['a','b',attrs]])
    assert graph['a']['b'] == {'weight':1,'fact':'a fact','relation':'KNOWS'}
    assert attrs['weight'] == 1.0


def test_graphiti_default_weight_and_duplicate_merge():
    graph = build_graph([['a','b',{'fact':'first'}],['b','a',{'weight':2,'fact':'last'}]])
    assert graph['a']['b'] == {'weight':3,'fact':'last'}
