from pyformlang.cfg import CFG, Production, Variable, Epsilon
import networkx as nx


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    nf_cfg = cfg.to_normal_form()
    productions = set(nf_cfg.productions)
    for var in cfg.get_nullable_symbols():
        productions.add(Production(Variable(var.value), [Epsilon()]))

    wnf_cfg = CFG(
        start_symbol=cfg.start_symbol, productions=productions
    ).remove_useless_symbols()

    return wnf_cfg


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    weak_cnf_cfg = cfg_to_weak_normal_form(cfg)

    r = []
    for v1, v2, symbol in graph.edges(data="label"):
        for production in weak_cnf_cfg.productions:
            if len(production.body) == 1 and production.body[0].value == symbol:
                r.append((production.head, v1, v2))

    for variable in weak_cnf_cfg.variables:
        if Production(variable, []) in weak_cnf_cfg.productions:
            for vertex in graph.nodes:
                r.append((variable, vertex, vertex))
    new = r.copy()

    while new:
        (N, n, m) = new.pop()

        for M, n_prime, m_prime in r:
            if m_prime == n:
                for production in weak_cnf_cfg.productions:
                    if (
                        len(production.body) == 2
                        and production.body[0] == M
                        and production.body[1] == N
                    ):
                        N_prime = production.head
                        new_relation = (N_prime, n_prime, m)
                        if new_relation not in r:
                            r.append(new_relation)
                            new.append(new_relation)

        for M, n_prime, m_prime in r:
            if m == n_prime:
                for production in weak_cnf_cfg.productions:
                    if (
                        len(production.body) == 2
                        and production.body[0] == N
                        and production.body[1] == M
                    ):
                        N_prime = production.head
                        new_relation = (N_prime, n, m_prime)
                        if new_relation not in r:
                            r.append(new_relation)
                            new.append(new_relation)

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    return {
        (start, final)
        for variable, start, final in r
        if start in start_nodes
        and final in final_nodes
        and variable == cfg.start_symbol
    }
