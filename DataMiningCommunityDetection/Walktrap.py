#Sean Billings 2017
# Does walktrap I guess lol
import argparse
import RandomWalkMachines
import DistanceFunctions
import ClusterSelectionFunction
from Community import Community

def community_walktrap(nodes, edges, cluster_selection_function, distance_function):

    # Initialize clusters
    partitions = [Community([n]) for n in nodes]

    best_modularity = -1
    best_partition = partitions

    # collection of all partitions
    all_partitions = [partitions]

    print("Initializing distances")
    #distances should be a map

    distances = {}
    count = 0.0
    total = len(partitions)
    for cluster in partitions:
        count += 1.0
        print(str(100*count/total) + "% completed\n" )
        adjacent_to_cluster = compute_adjacent(cluster, partitions, edges)
        distances[cluster] = distance_function.compute_distances(cluster, adjacent_to_cluster)

    for k in range(1,len(nodes)):

        adjacent_clusters = compute_all_adjacent(partitions, edges)

        # clusters are communities
        cluster_a,cluster_b = cluster_selection_function.cluster_selection(adjacent_clusters,distances)

        # construct new cluster from old cluster
        cluster_c = cluster_a.merge_community(cluster_b)

        # remove distances for c1,c2
        distances.pop(cluster_a, None)
        distances.pop(cluster_b, None)

        # prune distances to cluster_a, cluster_b

        # remove merged clusters
        partitions.remove(cluster_a)
        partitions.remove(cluster_b)

        # recompute distances for new cluster
        #should only recompute for adjacent clusters

        # next line may be unnecessary
        #distances[cluster_c] = distance_function.compute_distances(cluster_c,partitions)

        distances[cluster_c] = {}

        adjacent_to_cluster_c = compute_adjacent(cluster_c,partitions,edges)
        for ci in adjacent_to_cluster_c:
            # distances is map to map {a : {b:1}}
            distance = distance_function.compute_distance(cluster_c.nodes,ci.nodes)
            # maybe don't need to compute distances
            distances[cluster_c][ci] = distance
            distances[ci][cluster_c] = distance

        partitions += [cluster_c]

        partition = partitions[:]
        all_partitions = all_partitions + [partition]

        q = compute_modularity(partition, edges)
        if q > best_modularity:
            best_modularity = q
            best_partition = partition


        if k % 10 == 0:
            print("Iteration: " + str(k))
            print('Best Modularity: ' + str(best_modularity))
            print('Best Partition size: ' + str(len(best_partition)))

    return best_modularity, best_partition




###### AUXILLARY METHODS ######



#Generates a list of adjacent clusters
def compute_all_adjacent(partitions,edges):
    adjacent_clusters = []

    for i in range(len(partitions)):
        for j in range(i,len(partitions)):
            cluster_a = partitions[i]
            cluster_b = partitions[j]
            cluster_b_set = set(cluster_b.nodes)
            for node in cluster_a.nodes:
                for receiving_node in edges[node]:
                    if receiving_node in cluster_b_set:
                        adjacent_clusters += [(cluster_a,cluster_b)]
    return adjacent_clusters

#Generates a list of adjacent clusters
def compute_adjacent(cluster_a,partitions,edges):
    adjacent_clusters = []
    for j in range(0,len(partitions)):
        cluster_b = partitions[j]
        cluster_b_set = set(cluster_b.nodes)
        for node in cluster_a.nodes:
            for receiving_node in edges[node]:
                if receiving_node in cluster_b_set:
                    adjacent_clusters += [cluster_b]
                    break;
    return adjacent_clusters

# Compute Modularity for the given partition
def compute_modularity(partition, edges):
    modularity = 0
    total_edges = 0
    for node in edges:
        total_edges += len(edges[node])
    total_edges = total_edges /2
    for cluster in partition:
        internal_edges = 0
        external_edges = 0
        #build identification map of nodes in cluster
        node_set = set(cluster.nodes)
        for node in cluster.nodes:
            for receiving_node in edges[node]:
                if receiving_node in node_set:
                    internal_edges += 0.5

                external_edges += 0.5

        modularity += internal_edges/total_edges - (external_edges/total_edges)**2
    return modularity


def do_walktrap(nodes, edges, t, walks):
    node_degrees = {}
    for node in edges:
        node_degrees[node] = len(edges[node])

    rwm = RandomWalkMachines.RandomWalkMachine(nodes, edges)
    initialized_transition_probabilities = rwm.compute_transfer_probabilities(t, walks)

    distance_function = DistanceFunctions.RandomWalkDistanceFunction(t, initialized_transition_probabilities,
                                                                     node_degrees)
    cluster_selection_function = ClusterSelectionFunction.VariationClusterSelectionFunction(len(nodes))

    return community_walktrap(nodes, edges, cluster_selection_function, distance_function)

parser = argparse.ArgumentParser(description='Compute communities with walktrap.')
parser.add_argument('edges', help='A file that contains the edges in the graph to compute')
parser.add_argument('t', type=int, help='length of markov chain process')
parser.add_argument('walks', type=int, help='number of random walks per node to compute')
args = parser.parse_args()

# read input nodes, edges
edges = {} # symmetric hash map representation
nodes = set([])
t = args.t
walks = args.walks

with open(args.edges, 'r') as graph_file:
    for line in graph_file:
        if not '#' in line:
            a_b = line.split(' ')
            if len(a_b) == 2:
                a = int(a_b[0]) -1
                b = int(a_b[1].replace('\n','')) -1
                if a in edges:
                    edges[a].append(b)
                else:
                    edges[a] = [b]

                if b in edges:
                    edges[b].append(a)
                else:
                    edges[b] = [a]

                if not a in nodes:
                    nodes.add(a)

                if not b in nodes:
                    nodes.add(b)

print("Edges: " + str(edges))
print("Nodes: " + str(nodes))


best_modularity, best_partition = do_walktrap(nodes,edges,t,walks)



print("Best Results")
print best_modularity
print len(best_partition)
for cluster in best_partition:
    print cluster.nodes


# Compute modularity of walktrap test result
#karate kid
community_a= Community([16, 5, 6, 4, 10])
community_b=Community([8, 30, 9, 2, 13, 0, 11, 19, 12, 21, 17, 1, 3, 7])
community_c=Community([26, 29, 22, 18, 15, 14, 20, 33, 32, 28, 31, 24, 25, 23, 27])

#standard test
#community_a = Community([10,11,13])
#community_b = Community([3, 6, 4, 5, 7, 0, 1, 2])
#community_c = Community([15, 8, 12, 9, 14])

walktrap_actual_partition = [community_a,community_b,community_c]

actual_q = compute_modularity(walktrap_actual_partition, edges)

print("Actual Q")
print(actual_q)

