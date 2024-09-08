import cfpq_data as cfpq
from typing import Tuple, Set, Any
import networkx as nx
from dataclasses import dataclass


@dataclass
class GraphInfo:
    node_count: int
    edge_count: int
    labels: Set[Any]


def load_graph(name: str) -> nx.MultiDiGraph:
    path = cfpq.download(name)
    graph = cfpq.graph_from_csv(path)
    return graph


def get_graph_info(name: str) -> GraphInfo:
    G = load_graph(name)
    return GraphInfo(
        G.number_of_nodes(), G.number_of_edges(), set(cfpq.get_sorted_labels(G))
    )


def save_to_dot(graph: nx.MultiDiGraph, filename: str) -> None:
    dot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    dot_graph.write_raw(filename)


def build_labeled_two_cycles_graph(
    n: int, m: int, labels: Tuple[str, str], output_file: str
) -> None:
    G = cfpq.labeled_two_cycles_graph(n, m, labels=labels)
    save_to_dot(G, output_file)
