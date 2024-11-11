import numpy as np
from scipy import sparse
from project.task3 import AdjacencyMatrixFA
from project.task2 import graph_to_nfa, regex_to_dfa
from networkx import MultiDiGraph


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    dfa_m = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa_m = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    nfa_start_states_count = len(nfa_m.start_state_index)

    def init_front():
        dfa_start = list(dfa_m.start_state_index)[0]
        rows = [
            dfa_start + dfa_m.states_count * i for i in range(nfa_start_states_count)
        ]
        cols = [start_state_ind for start_state_ind in nfa_m.start_state_index]
        data = np.ones(nfa_start_states_count, dtype=bool)
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(dfa_m.states_count * nfa_start_states_count, nfa_m.states_count),
            dtype=bool,
        )

    def update_front(front):
        fronts_decomposed = {}
        dfa_matricies_tr = {
            key: m.transpose() for key, m in dfa_m.boolean_decomposition.items()
        }
        labels = dfa_m.boolean_decomposition.keys() & nfa_m.boolean_decomposition.keys()
        for label in labels:
            fronts_decomposed[label] = front @ nfa_m.boolean_decomposition[label]
            for ind in range(nfa_start_states_count):
                fronts_decomposed[label][
                    ind * dfa_m.states_count : (ind + 1) * dfa_m.states_count
                ] = (
                    dfa_matricies_tr[label]
                    @ fronts_decomposed[label][
                        ind * dfa_m.states_count : (ind + 1) * dfa_m.states_count
                    ]
                )

        front_new = sparse.csr_matrix(
            (dfa_m.states_count * nfa_start_states_count, nfa_m.states_count),
            dtype=bool,
        )
        for front in fronts_decomposed.values():
            front_new += front

        return front_new

    visited = sparse.csr_matrix(
        (dfa_m.states_count * nfa_start_states_count, nfa_m.states_count), dtype=bool
    )
    front = init_front()
    while front.count_nonzero() > 0:
        visited += front
        front = update_front(front)
        front = front > visited

    dfa_final_states_index = dfa_m.final_state_index
    nfa_idx_to_st = {index: state for state, index in nfa_m.state_index.items()}
    nfa_final_states = np.array(
        [i in nfa_m.final_state_index for i in range(nfa_m.states_count)], dtype=bool
    )
    pairs = set()

    for i, nfa_start_state_id in enumerate(nfa_m.start_state_index, 0):
        for dfa_final_state_id in dfa_final_states_index:
            row = visited[i * dfa_m.states_count + dfa_final_state_id]
            row_vector = np.array(
                [i in row.indices for i in range(nfa_m.states_count)],
                dtype=bool,
            )
            vector = row_vector & nfa_final_states
            nfa_final_states_reached = np.nonzero(vector)[0]
            for reached_nfa_final_state_ind in nfa_final_states_reached:
                pairs.add(
                    (
                        nfa_idx_to_st[nfa_start_state_id],
                        nfa_idx_to_st[reached_nfa_final_state_ind],
                    )
                )

    return pairs
