from Walktrap import do_walktrap
import argparse
import matplotlib
import matplotlib.pyplot as plt

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

t_set = xrange(2,24)
walk_set = [150,250,500,1000,1500,2000,2500]

best_modularity, best_partition = do_walktrap(nodes,edges,t,walks)
t_modularities = []
t_partitions = []
for ti in t_set:
    modularity,partition = do_walktrap(nodes,edges,ti,walks)
    t_modularities.append(modularity)
    t_partitions.append(partition)

with matplotlib.rc_context({'figure.figsize': [20, 5], 'xtick.labelsize': 8}):
    plt.bar(t_set, t_modularities)
    plt.ylabel('Modularity')
    plt.xlabel('t')
plt.show()

print(t_set)
print(t_modularities)
for t_partition in t_partitions:
    print(t_partition)

w_modularities = []
w_partitions = []

for wi in walk_set:
    modularity,partition = do_walktrap(nodes,edges,t,wi)
    w_modularities.append(modularity)
    w_partitions.append(partition)

with matplotlib.rc_context({'figure.figsize': [20, 5], 'xtick.labelsize': 8}):
    plt.plot(walk_set, w_modularities)
    plt.ylabel('Modularity')
    plt.xlabel('Walk_size')
plt.show()

print(walk_set)
print(w_modularities)
for w_partition in w_partitions:
    print(w_partition)