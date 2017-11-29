# Sean Billings 2017

from abc import ABCMeta, abstractmethod
import numpy as np
import random

class RandomWalkMachine:
    __metaclass__ = ABCMeta

    # Construct a RandomWalk Machine on a graph structure
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges # assume edge structure is a map of index to array
        self.rand = random.Random(13186578)

    def compute_transfer_probabilities(self, numWalks,t):
        transfer_probabilities = np.zeros((len(self.nodes),len(self.nodes)))
        for node in self.nodes:
            for _ in range(numWalks):
                transfer_probabilities[node,self.do_random_walk(node,t)] += 1

        return transfer_probabilities / numWalks

    # tested
    def do_random_walk(self, starting_node,t):

        ending_node = starting_node
        for _ in range(t):
            ending_node = self.rand.choice(self.edges[ending_node])

        return ending_node
