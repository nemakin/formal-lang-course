from project.task1 import GraphInfo, build_labeled_two_cycles_graph
import filecmp

from project.task1 import get_graph_info


def test_from_dataset_graph_info():
    info_expected = GraphInfo(
        node_count=144,
        edge_count=252,
        labels={
            "label",
            "type",
            "range",
            "description",
            "creator",
            "example",
            "unionOf",
            "first",
            "rest",
            "contributor",
            "comment",
            "subClassOf",
            "scopeNote",
            "definition",
            "seeAlso",
            "title",
            "subPropertyOf",
            "inverseOf",
            "isDefinedBy",
            "disjointWith",
            "domain",
        },
    )

    info_actual = get_graph_info("skos")
    assert info_expected == info_actual


def test_graph_building(tmp_path):
    actual_path = tmp_path / "actual.dot"
    build_labeled_two_cycles_graph(3, 4, ("a", "b"), actual_path)
    assert filecmp.cmp(actual_path, "tests/tmp/expected.dot")
