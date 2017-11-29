import networkx as nx
import argparse
import matplotlib.pyplot as plt

g = nx.Graph()

parser = argparse.ArgumentParser(description='Construct a graph with networkx')
parser.add_argument('edges', help='A file that contains the edges in the graph to compute')
args = parser.parse_args()

# read input nodes, edges
nodes = set([])

#pos = {}
#for node in g.nodes_iter():
#    pos[node] = (xcoord, ycoord)


with open(args.edges, 'r') as graph_file:
    for line in graph_file:
        if not '#' in line:
            a_b = line.split(' ')
            if len(a_b) == 2:
                a = int(a_b[0])
                b = int(a_b[1].replace('\n',''))

                if not a in nodes:
                    g.add_node(a)
                if not b in nodes:
                    g.add_node(b)

                g.add_edge(a, b)

#use distances to annotate graph


#nx.draw(g, with_labels=True, font_weight='bold')

#plt.draw()
#plt.show()

pos=nx.spring_layout(g)

nx.draw_networkx_nodes(g, pos,
                        nodelist=[17, 6, 7, 5, 11],
                        node_color='r',
                        node_size=500,
                        with_labels=True,
                       font_weight='bold',
                       alpha=1)

nx.draw_networkx_nodes(g, pos,
                        nodelist=[27, 30, 16, 15, 23, 21, 19, 0, 33, 29, 32, 25, 26, 24, 28],
                        node_color='b',
                        node_size=500,
                        with_labels=True,
                       font_weight='bold',
                       alpha=1)

nx.draw_networkx_nodes(g, pos,
                        nodelist=[9, 31, 10, 3, 14, 1, 12, 20, 13, 18, 22, 2, 4, 8],
                        node_color='y',
                        node_size=500,
                        with_labels=True,
                       font_weight='bold',
                       alpha=1)

nx.draw_networkx_edges(g,pos,width=1.0,alpha=0.5)
nx.draw_networkx_edges(g,pos,
                       edgelist=g.edges,
                       width=8,alpha=0.5,edge_color='w')

labels = {}
for i in g.nodes:
    labels[i] = str(i)

nx.draw_networkx_labels(g,pos,labels,font_size=16)

plt.draw()
plt.show()

#[17, 21, 4, 10, 6, 16, 28, 5, 7, 1, 3, 2, 13, 25, 24, 31, 11, 0, 12]
#[30, 19, 15, 20, 9, 27, 18, 14, 8, 22, 23, 33]
#[26, 29, 32]

#[28, 25, 31]
#[12, 8, 18, 32, 17, 30, 7, 2, 9, 1, 19]
#[22, 26, 24, 27, 20, 14, 23, 29, 15, 33]
#[4, 6, 16, 5, 10, 11, 0, 21, 3, 13]

#karate kid
#community_a= Community([16, 5, 6, 4, 10])
#community_b=Community([8, 30, 9, 2, 13, 0, 11, 19, 12, 21, 17, 1, 3, 7])
#community_c=Community([26, 29, 22, 18, 15, 14, 20, 33, 32, 28, 31, 24, 25, 23, 27])