from project.task2 import (
    NondeterministicFiniteAutomaton,
    regex_to_dfa,
    graph_to_nfa,
    State,
)

from collections import defaultdict
from typing import Dict, Set, Iterable
import numpy as np
import scipy.sparse as sp
from networkx import MultiDiGraph


class AdjacencyMatrixFA:
    def __init__(
        self,
        automaton: NondeterministicFiniteAutomaton = None,
    ):
        self.state_index: Dict[State, int] = {}
        self.start_state_index: Set[int] = set()
        self.final_state_index: Set[int] = set()
        self.states_count = 0
        self.index_state = {}
        self.boolean_decomposition: Dict[str, sp.csr_matrix] = {}

        if automaton is None:
            return

        graph = automaton.to_networkx()
        self.states_count = graph.number_of_nodes()
        self.state_index = {state: idx for idx, state in enumerate(graph.nodes)}
        self.index_state = {idx: state for state, idx in self.state_index.items()}

        for node, attributes in graph.nodes(data=True):
            if attributes.get("is_start", False):
                self.start_state_index.add(self.state_index[node])
            if attributes.get("is_final", False):
                self.final_state_index.add(self.state_index[node])

        transitions = defaultdict(
            lambda: np.zeros((self.states_count, self.states_count), dtype=bool)
        )

        for source, target, symbol in graph.edges(data="label"):
            if symbol:
                transitions[symbol][
                    self.state_index[source], self.state_index[target]
                ] = True

        self.boolean_decomposition = {
            sym: sp.csr_matrix(matrix) for sym, matrix in transitions.items()
        }

    def accepts(self, word: Iterable[str]) -> bool:
        current_states = set(self.start_state_index)

        for symbol in word:
            next_states = set()
            for state in current_states:
                for dst in self.boolean_decomposition[symbol].nonzero()[1]:
                    next_states.add(dst)
            current_states = next_states
            if not current_states:
                return False

        return bool(set(self.final_state_index).intersection(current_states))

    def is_empty(self) -> bool:
        reachability_matrix = self.transitive_closure()

        for start_state in self.start_state_index:
            for final_state in self.final_state_index:
                if reachability_matrix[start_state, final_state]:
                    return False

        return True

    def transitive_closure(self):
        closure = sp.csr_matrix((self.states_count, self.states_count), dtype=bool)
        closure.setdiag(True)

        if not self.boolean_decomposition:
            return closure

        closure = closure + sum(self.boolean_decomposition.values())
        res = np.linalg.matrix_power(closure.toarray(), self.states_count)
        return sp.csr_matrix(res)


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> AdjacencyMatrixFA:
    intersected_automaton = AdjacencyMatrixFA()

    intersected_automaton.states_count = (
        automaton1.states_count * automaton2.states_count
    )

    intersected_automaton.state_index = {
        (s1, s2): (
            automaton1.state_index[s1] * automaton2.states_count
            + automaton2.state_index[s2]
        )
        for s1 in automaton1.state_index
        for s2 in automaton2.state_index
    }

    intersected_automaton.index_state = {
        idx: state for state, idx in intersected_automaton.state_index.items()
    }

    intersected_automaton.start_state_index = {
        s1 * automaton2.states_count + s2
        for s1 in automaton1.start_state_index
        for s2 in automaton2.start_state_index
    }

    intersected_automaton.final_state_index = {
        f1 * automaton2.states_count + f2
        for f1 in automaton1.final_state_index
        for f2 in automaton2.final_state_index
    }

    intersected_automaton.boolean_decomposition = {}

    common_symbols = set(automaton1.boolean_decomposition.keys()).intersection(
        automaton2.boolean_decomposition.keys()
    )

    for symbol in common_symbols:
        matrix1 = automaton1.boolean_decomposition[symbol]
        matrix2 = automaton2.boolean_decomposition[symbol]

        intersected_automaton.boolean_decomposition[symbol] = sp.kron(
            matrix1, matrix2, format="csr"
        )

    return intersected_automaton


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    regex_adj = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_adj = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersect = intersect_automata(regex_adj, graph_adj)
    result_set = set()
    transitive_closure = intersect.transitive_closure()

    regex_start_states = [
        key
        for key in regex_adj.state_index
        if regex_adj.state_index[key] in regex_adj.start_state_index
    ]
    regex_final_states = [
        key
        for key in regex_adj.state_index
        if regex_adj.state_index[key] in regex_adj.final_state_index
    ]

    for st in start_nodes:
        for fn in final_nodes:
            for st_reg in regex_start_states:
                for fn_reg in regex_final_states:
                    if transitive_closure[
                        intersect.state_index[(st_reg, st)],
                        intersect.state_index[(fn_reg, fn)],
                    ]:
                        result_set.add((st, fn))

    return result_set
