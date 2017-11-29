import networkx as nx
import random

def generate_community_graph(number_of_communities,mean_size,variance,p_in,p_out):
    sizes = []
    for i in range(number_of_communities):
        sizes += [random.randint(mean_size-2*variance,mean_size+2*variance)]
    g = nx.random_partition_graph(sizes,p_in,p_out)
    edges = g.edges()
    nodes = g.nodes()
    return nodes,edges

nodes,edges = generate_community_graph(20,30,4,0.25,0.05)

with open("samples/Test_Graph_p_25_05.txt", 'w') as graph_file:
    for edge in edges:
        graph_file.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')