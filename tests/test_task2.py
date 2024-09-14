from project.task2 import regex_to_dfa, graph_to_nfa
from project.task1 import load_graph


def test_regex_to_dfa():
    dfa = regex_to_dfa("abc|d")
    assert dfa.accepts(["abc"])
    assert dfa.accepts(["abc"])
    assert not dfa.accepts("abc")


def test_graph_to_nfa_empty_():
    skos = load_graph("skos")
    nfa = graph_to_nfa(skos)
    assert len(nfa.start_states) == skos.number_of_nodes()
    assert len(nfa.final_states) == skos.number_of_nodes()
