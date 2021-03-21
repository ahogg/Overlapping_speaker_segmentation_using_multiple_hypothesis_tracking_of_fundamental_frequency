"""
This file is based on code from:  https://github.com/jonperdomo/openmht
"""
import operator
import random
from graph import Graph


class WeightedGraph(Graph):
    """
    A graph with weighted vertices.
    """
    def __init__(self, graph_dict=None):
        Graph.__init__(self, graph_dict)
        self.__weights = {}

    def mwis(self):
        """Determine the maximum weighted independent set."""

        # Find all maximal independent sets
        complement = self.complement()
        ind_sets = []

        self.bron_kerbosch3(complement, ind_sets)

        # Find the maximum weighted set
        min_weight = 100000000000000000 # min(self.__weights.values())
        mwis = []
        for ind_set in ind_sets:
            set_weight = sum([self.__weights[str(i)] for i in ind_set])
            if set_weight < min_weight:
                min_weight = set_weight
                mwis = ind_set

        return mwis

    def indsets(self):
        """Determine the maximum weighted independent set."""

        # Find all maximal independent sets
        complement = self.complement()
        ind_sets = []
        self.bron_kerbosch3(complement, ind_sets)

        return ind_sets, self.__weights

    def bron_kerbosch3(self, g, results):
        """With vertex ordering."""
        P = set(range(len(self.vertices())))
        R, X = set(), set()
        deg_ord = self.degeneracy_ordering(g)

        for v in deg_ord:
            N_v = self.N(v, g)
            self.bron_kerbosch2(R | {v}, P & N_v, X & N_v, g, results)

            P = P - {v}
            X = X | {v}

    def bron_kerbosch2(self, R, P, X, g, results):
        """With pivoting."""
        if not any((P, X)):
            results.append(R)
            return

        u = random.choice(tuple(P | X))
        for v in P - self.N(u, g):
            N_v = self.N(v, g)
            self.bron_kerbosch(R | {v}, P & N_v, X & N_v, g, results)

            P = P - {v}
            X = X | {v}

    def bron_kerbosch(self, R, P, X, g, results):
        """Without pivoting."""
        if not any((P, X)):
            results.append(R)

        for v in set(P):
            N_v = self.N(v, g)
            self.bron_kerbosch(R | {v}, P & N_v, X & N_v, g, results)

            P = P - {v}
            X = X | {v}

    def degeneracy_ordering(self, g):
        """Order such that each vertex has d or fewer neighbors that come later in the ordering."""
        v_ordered = set()
        degrees = list(enumerate(self.vertex_degrees(g)))
        while degrees:
            min_index, min_value = min(degrees, key=operator.itemgetter(1))
            v_ordered.add(min_index)
            degrees.remove((min_index, min_value))

        return v_ordered

    def N(self, v, g):
        return set([i for i, n_v in enumerate(g[v]) if n_v])

    def add_weighted_vertex(self, vertex, weight):
        """
        Add a weighted vertex to the graph.
        """
        self.add_vertex(vertex)
        self.__weights[vertex] = weight

    def __str__(self):
        res = super(WeightedGraph, self).__str__()
        res += "\nWeights: "
        for w in self.__weights.values():
            res += str(w) + " "

        return res
