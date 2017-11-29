from Walktrap import community_walktrap
import RandomWalkMachines
import DistanceFunctions
import ClusterSelectionFunction


# initialize graph
# read input nodes, edges
edges = {} # symmetric hash map representation
nodes = []
t = 4
walks = 60

with open("walktrap_test_input.txt", 'r') as graph_file:
    for line in graph_file:
        if not line.contains('#'):
            a_b = line.split(' ')
            if len(line) == 2:
                a = line[0]
                b = line[1]
                if edges.contains(a):
                    edges[a].append(b)
                else:
                    edges[a] = [b]

                if edges.contains(b):
                    edges[b].append(a)
                else:
                    edges[b] = [a]

                if not nodes.contains(a):
                    nodes.append(a)

                if not nodes.contains(b):
                    nodes.append(b)


rwm = RandomWalkMachines(nodes, edges)
initialized_transition_probabilities = rwm.compute_transfer_probabilities(t,walks)

node_degrees = {}
for node in edges:
    node_degrees[node] = len(edges[node])

distance_function = DistanceFunctions.RandomWalkDistanceFunction(t,initialized_transition_probabilities,node_degrees)
cluster_selection_function = ClusterSelectionFunction.VariationClusterSelectionFunction(len(nodes))

best_modularity, best_partition = community_walktrap(nodes,edges,cluster_selection_function, distance_function)