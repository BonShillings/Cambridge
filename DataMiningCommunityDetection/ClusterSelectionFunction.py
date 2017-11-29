from abc import ABCMeta, abstractmethod
import numpy as np

class VariationClusterSelectionFunction:
    __metaclass__ = ABCMeta

    # constructor parameters
    # structure is the nodes
    def __init__(self, node_count):
        self.node_count = node_count

    #There is a more efficient tree-based selection implementation
    # There is a problem here
    def cluster_selection(self,adjacent_clusters,distances):
        min = 10000000000
        chosen_pair = adjacent_clusters[0]

        for pair in adjacent_clusters:
            cluster_a = pair[0]
            cluster_b = pair[1]
            if cluster_a != cluster_b:
                distance = distances[cluster_a][cluster_b]
                variation = self.compute_variation(cluster_a.nodes,cluster_b.nodes,distance)
                if variation < min:
                    min = variation
                    chosen_pair = pair

        return chosen_pair


    def compute_variation(self,cluster_a,cluster_b, distance):

        len_a = len(cluster_a)
        len_b = len(cluster_b)
        variation = len_a*len_b/float((len_a + len_b))
        variation = variation*(distance**2)
        # could divide by node_count, but it is constant across values so doesnt change the minima

        # debug
        if variation == 0:
            print(distance)
            print(variation)
        return variation

