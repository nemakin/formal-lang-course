import networkx as nx

from pyformlang.cfg import CFG
from scipy.sparse import csr_matrix

from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_nodes = set(graph.nodes)
    start_nodes = start_nodes if start_nodes else graph_nodes
    final_nodes = final_nodes if final_nodes else graph_nodes

    node_to_idx_mapping = {node: i for i, node in enumerate(graph_nodes)}
    idx_to_node_mapping = {i: node for i, node in enumerate(graph_nodes)}

    n = len(graph_nodes)
    cfg = cfg_to_weak_normal_form(cfg)
    decomposition = {var: csr_matrix((n, n), dtype=bool) for var in cfg.variables}

    for u, v, label in graph.edges.data("label"):
        for prod in cfg.productions:
            if (
                len(prod.body) == 1
                and prod.body[0] in cfg.terminals
                and prod.body[0].value == label
            ):
                decomposition[prod.head][
                    node_to_idx_mapping[u], node_to_idx_mapping[v]
                ] = True

    for N in cfg.get_nullable_symbols():
        decomposition[N].setdiag(True)

    changed = True
    while changed:
        changed = False
        for prod in cfg.productions:
            if len(prod.body) != 2:
                continue
            A_i = prod.head
            A_j, A_k = prod.body
            head_matrix = decomposition[A_i] + (decomposition[A_j] @ decomposition[A_k])
            if (decomposition[A_i] != head_matrix).nnz != 0:
                changed = True
                decomposition[A_i] = head_matrix

    return {
        (idx_to_node_mapping[u], idx_to_node_mapping[v])
        for u, v in zip(*decomposition[cfg.start_symbol].nonzero())
        if idx_to_node_mapping[u] in start_nodes
        and idx_to_node_mapping[v] in final_nodes
    }
