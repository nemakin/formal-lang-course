from networkx import MultiDiGraph
from typing import Set
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    State,
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int] = None, final_states: Set[int] = None
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)
    if not start_states:
        start_states = {int(x) for x in graph.nodes}
    if not final_states:
        final_states = {int(x) for x in graph.nodes}
    for state in start_states:
        nfa.add_start_state(State(state))
    for state in final_states:
        nfa.add_final_state(State(state))
    return nfa
