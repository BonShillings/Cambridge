#Sean Billings 2017

from abc import ABCMeta, abstractmethod
import numpy as np

class DistanceFunction:
    __metaclass__ = ABCMeta

    # constructor parameters
    def __init_(self, structure):
        self.structure = structure

    #Computes distances for all clusters in @partitions
    @abstractmethod
    def initialize_distances(self, partitions): pass

    @abstractmethod
    def compute_distances(self, cluster, partitions): pass


    # compute distance between 2 nodes/clusters
    @abstractmethod
    def compute_distance(self, cluster_a, cluster_b): pass


class RandomWalkDistanceFunction(DistanceFunction):
    __metaclass__ = ABCMeta

    # constructor parameters
    # t is the number of times to hypothetically iterate markov chain process
    # transfer_probabilities is the base of the markov chain
    # degree_matrix is a diagonal matrix of the degrees of the nodes in graph under consideration
    def __init__(self, t, transfer_probabilities, node_degrees):
        self.t = t
        self.transfer_probabilities = transfer_probabilities
        self.degree_matrix = node_degrees
        #self.t_transfer_probabilities = np.linalg.matrix_power(transfer_probabilities,t)
        self.t_transfer_probabilities = transfer_probabilities[:]

    #Computes distances for all clusters in @partitions
    # should maybe use a specific distance function
    def initialize_distances(self, partitions):
        all_distances = {}
        count = 0.0
        total = len(partitions)
        for cluster in partitions:
            count += 1.0
            print(str(100*count/total) + "% complete")
            all_distances[cluster] = self.compute_distances(cluster,partitions)

        return all_distances


    def compute_distances(self,cluster, partitions):
        cluster_distances = {}
        for cluster_b in partitions:
            if cluster != cluster_b:
                cluster_distances[cluster_b] = self.compute_distance(cluster.nodes,cluster_b.nodes)
        return cluster_distances



    # compute distance between 2 nodes/clusters
    # cluster_a, cluster_b should be the nodes from respective communities
    def compute_distance(self, cluster_a, cluster_b):
        if len(cluster_a) > 1:

            #compute cluster_a transition probabilities

            cluster_a_transfer_probabilities = np.average([self.t_transfer_probabilities[i] for i in cluster_a], axis =0)

            #compute cluster -> cluster distance
            if len(cluster_b) > 1:
                cluster_b_transfer_probabilities = np.average([self.t_transfer_probabilities[i] for i in cluster_b], axis = 0)

                r_CaCb = 0
                for k in range(len(self.degree_matrix)):
                    r_CaCb += (cluster_a_transfer_probabilities[k] - cluster_b_transfer_probabilities[k])**2/self.degree_matrix[k]
                return np.sqrt(r_CaCb)

            # compute cluster -> node distance
            else:
                node_j = cluster_b[0]
                r_Caj = 0
                for k in range(len(self.degree_matrix)):
                    # this isnt right
                    r_Caj += (cluster_a_transfer_probabilities[k] - self.t_transfer_probabilities[node_j,k])** 2 /self.degree_matrix[k]
                return np.sqrt(r_Caj)
        else:

            # compute node -> cluster distance
            if len(cluster_b) > 1:
                return self.compute_distance(cluster_b,cluster_a)
            else:
                #compute node-> node distance r_ij
                node_i = cluster_a[0]
                node_j = cluster_b[0]
                r_ij = 0
                for k in range(len(self.degree_matrix)):
                    r_ij += (self.t_transfer_probabilities[node_i,k] - self.t_transfer_probabilities[node_j,k])**2/self.degree_matrix[k]
                return np.sqrt(r_ij)





