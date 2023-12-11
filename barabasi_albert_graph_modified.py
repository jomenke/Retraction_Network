"""
A custom network adapted from NetworkX's implementation of the Barabási–Albert preferential attachment graph.
"""

import numpy as np
from scipy.stats import rv_continuous
import networkx as nx
from networkx import Graph
from collections.abc import Iterable
from typing import Generator
from numpy.random import default_rng
from numba import jit


class InvalidNodeEdgeCounts(Exception):
    """Raised when the amount of unique edge counts < number of random values to generate"""
    pass


def _random_subset(seq: Iterable, m: int, rng: Generator):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    if rng is None or isinstance(rng, int):
        rng = default_rng(rng)
    if len(set(seq)) < m:
        raise InvalidNodeEdgeCounts()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


class skewnorm_gen_modified(rv_continuous):
    def __init__(self):
        super().__init__()

    def rvs(self, mean, std, skew, size=1, random_state=default_rng(42)):
        u0 = random_state.normal(loc=mean, scale=std, size=size)
        v = random_state.normal(loc=mean, scale=std, size=size)
        d = skew/np.sqrt(1 + skew**2)
        u1 = d*u0 + v*np.sqrt(1 - d**2)
        output = np.where(u0 >= 0, u1, -u1).astype(int)
        output[output <= 0] = 1  # minimum number of references to be included in citation network is 1
        return output


def barabasi_albert_graph_modified(
        n: int,
        seed: Generator | int | None = None,
):
    """Returns a small world network graph (10% of nodes), expanded using Barabási–Albert preferential attachment

    A directed graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """
    initial_nodes = round(n*0.1)
    initial_graph = nx.watts_strogatz_graph(n=initial_nodes, k=26, p=0.1)

    G = initial_graph.copy()

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]

    # Start adding the other n - m0 nodes.
    source = len(G)

    #  References in articles based on skew probability distribution
    edge_count_idx = 0
    refs = skewnorm_gen_modified().rvs(mean=25.7, std=18.5, skew=10, size=n-initial_nodes)
    while source < n:
        # Now choose # unique nodes from the existing nodes
        edge_count = refs[edge_count_idx]
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, edge_count, seed)
        # Add edges to # nodes from the source.
        G.add_edges_from(zip([source] * edge_count, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has # edges to add to the list.
        repeated_nodes.extend([source] * edge_count)

        edge_count_idx += 1
        source += 1

    DiG = G.to_directed().copy()
    to_remove = [(citing, cited) for citing, cited in DiG.edges() if citing < cited]
    DiG.remove_edges_from(to_remove)

    return DiG


if __name__ == "__main__":
    import timeit
    print(timeit.timeit('barabasi_albert_graph_modified(1000, seed=42)', globals=globals(), number=1))
    # G = barabasi_albert_graph_modified(1000, seed=42)
    # print(G)
    # node_tuples = []
    # references = []
    # citations = []
    # for node in list(G):
    #     node_tuples.append((node, G.in_degree(node), G.out_degree(node)))
    #     citations.append(G.in_degree(node))
    #     references.append(G.out_degree(node))
    #     # sample = [(node, in, out), (node, citations, references)]
    # print(f"References: {np.mean(references)} +/- {np.std(references)} -> Median: {np.median(references)}")
    # print(f"Citations: {np.mean(citations)} +/- {np.std(citations)}  -> Median: {np.median(citations)}")
    # node_tuples.sort(reverse=True, key=lambda a: a[1])
    # print(f"Most references (top 3): {node_tuples[:3]}")
    # node_tuples.sort(reverse=True, key=lambda a: a[2])
    # print(f"Most citations (top 3): {node_tuples[:3]}")
