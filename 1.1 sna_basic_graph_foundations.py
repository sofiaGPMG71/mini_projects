#%% Import libraries

import os
import numpy as np
import pandas as pd # pandas.__version__
import matplotlib.pyplot as plt
#import time

import networkx as nx

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# Working directory
os.chdir("D://trainings//sna")
exec(open(os.path.abspath('sna_CommonUtils.py')).read())

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#First clean old graph images
remove_nonempty_folder('./images/layouts/', b_remove_base_folder = False)
remove_nonempty_folder('./images/descriptive/', b_remove_base_folder = False)

os.makedirs('./images/layouts/', exist_ok=True)
os.makedirs('./images/descriptive/', exist_ok=True)

#%%  Read data in Panda data frame for explorations

#First read node/entity
train_nodes = pd.read_csv("./data/sna_nodes_data.csv")
train_nodes.columns = map(str.upper, train_nodes.columns)

# First view
train_nodes.shape
train_nodes.dtypes
train_nodes.head()
train_nodes.info()

#Get different types of columns
col_numerics = np.array(GetNumericColumns(train_nodes))
col_category = np.array(GetFactorLogicalColumns(train_nodes))
col_chars = np.setdiff1d(train_nodes.columns, col_numerics)
col_chars = np.setdiff1d(col_chars, col_category)

#Convert all content to lower case
train_nodes[col_chars] = train_nodes[col_chars].apply(lambda x: x.astype(str).str.lower())

#Now read relationship/transactions/edges amoung node/entity
train_edges = pd.read_csv("./data/sna_edges_data.csv")
train_edges.columns = map(str.upper, train_edges.columns)

# First view
train_edges.shape
train_edges.dtypes
train_edges.head()
train_edges.info()

#Get different types of columns
col_numerics = np.array(GetNumericColumns(train_edges))
col_category = np.array(GetFactorLogicalColumns(train_edges))
col_chars = np.setdiff1d(train_edges.columns, col_numerics)
col_chars = np.setdiff1d(col_chars, col_category)

#Convert all content to lower case
train_edges[col_chars] = train_edges[col_chars].apply(lambda x: x.astype(str).str.lower())

#%% Explore undirectional graph
#What are we going to learn?

#get graph object from data frame
G = get_graph(train_edges) # also possible to use "from_pandas_edgelist"o

#Let us see the basic view
nx.draw(G, with_labels=True, font_weight='bold', node_color = 'lightblue')
show_graph(G)

#First see what all we have
print(nx.info(G))

G.nodes()
G.edges()

G['shiv'] # same as G.adj['shiv']
G.edges['shiv','shayam']

#Weight: How important/strong friendhip is. Use weight to show thickness of relationship
add_weight(G, train_edges)

#density: ratio of actual edges in the network to all possible edges in the network. It gives sense of how closely knit your network is.
nx.density(G)

#Degree:  who is the most important nodes. A node’s degree is the sum of its edges.
dict_degree = dict(G.degree(G.nodes()))
dict_degree = {k: v for k, v in sorted(dict_degree.items(), key=lambda item: item[1], reverse=True)} # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
dict_degree # These large nodes (top most) are known as hubs

# Add degree to show the size of node
add_node_attributes(G, dict_degree, 'node_size')

#When someone is not good or need to be highligted for some reason. Let us use 'FRAUD' in node data to highlight
add_node_attributes_from_df(G, train_nodes, 'FRAUD_MANUAL')

#The data is an abstract representation of connections between entities; the network is the specific instantiation of those assumptions.
#A network has a topology, or a connective shape, that could be centralized or decentralized; dense or sparse; cyclical or linear.
layouts = ['bipartite_layout', 'circular_layout', 'kamada_kawai_layout', 'planar_layout', 'random_layout', 'rescale_layout', 'shell_layout', 'spring_layout', 'spectral_layout', 'spiral_layout']
for layout in layouts:
    file_name_to_save = './images/layouts/' + layout + '.png'
    show_graph(G, layout, file_name_to_save)

layout = 'circular_layout'
show_graph(G, layout)

#shortest paths: find friends-of-friends
friends_path = nx.shortest_path(G, source="shiv", target="harry")
len(friends_path)-1 # # Need to travel steps
#
#Small world: Concepts on ppt
#It tests whether the graph is small world and then calulates the average shortest path length
small_world(G, seed_value)
#Note: It takes time and hence use once till there is change in edges

#diameter: the longest of all shortest paths. The measure is designed to give you a sense of the network’s overall size, the
#distance from one end of the network to another.
if nx.is_connected(G):
    print('Graph is connected')
    diameter = nx.diameter(G)
else:
    print('Graph is NOT connected hence using subgraph method')
    sub_graphs = nx.connected_components(G)
    max_component = max(sub_graphs, key=len)

    # Create a "subgraph" of just the largest component.  Then calculate the diameter of the subgraph
    diameter = nx.diameter(G.subgraph(max_component))

print(diameter)

#Transitivity: the ratio of all triangles over all possible triangles. It expresses how interconnected a graph is in terms of
# a ratio of actual over possible connections. Transitivity allows you a way of thinking about all the relationships in your
#graph that may exist but currently do not.
nx.transitivity(G)

#How to search big tree. Search or starting point
SG = nx.bfs_tree(G, 'shiv')
show_graph(SG, layout)

#Eccentricity: the largest distance between given node and all other nodes.
if nx.is_connected(G):
    print('Graph is connected')
    dict_eccentricity = nx.diameter(G)
else:
    print('Graph is NOT connected hence using subgraph method')
    sub_graphs = nx.connected_components(G)
    max_component = max(sub_graphs, key=len)

    # Create a "subgraph" of just the largest component.  Then calculate the diameter of the subgraph
    dict_eccentricity = nx.eccentricity(G.subgraph(max_component))

    dict_eccentricity = {k: round(v,2) for k, v in sorted(dict_eccentricity.items(), key=lambda item: item[1], reverse=False)}

print(dict_eccentricity)

#Closeness Centrality: Determin closeness to all other nodes. Used for importance
dict_closeness_centrality = nx.closeness_centrality(G)
dict_closeness_centrality = {k: round(v,2) for k, v in sorted(dict_closeness_centrality.items(), key=lambda item: item[1], reverse=True)}
dict_closeness_centrality

#Eigenvector centrality: It looks at a combination of a node’s edges and the edges of that node’s neighbors. It cares if you
#are a hub, but it also cares how many hubs you are connected to. It’s calculated as a value from 0 to 1: the closer to one,
#the greater the centrality. Eigenvector centrality is useful for understanding which nodes can get information to many other
#nodes quickly.  If you’ve used Google, then you’re already somewhat familiar with Eigenvector centrality. Their PageRank
#algorithm uses an extension of this formula to decide which webpages get to the top of its search results.
dict_eigenvector = nx.eigenvector_centrality(G)
dict_eigenvector = {k: round(v,2) for k, v in sorted(dict_eigenvector.items(), key=lambda item: item[1], reverse=True)}
dict_eigenvector # a well-connected people, who can spread a message very efficiently.

#Betweenness centrality: It finds 'broker' who connects two clusters. It looks at all the shortest paths that pass through a
#particular node . It is fairly good at finding nodes that connect two otherwise disparate parts of a network. If you’re
#the only thing connecting two clusters, every communication between those clusters has to pass through you.
#it’s a quick way of giving you a sense of which nodes are important not because they have lots of connections themselves
#but because they stand between groups, giving the network connectivity and cohesion.
dict_betweenness_centrality = nx.betweenness_centrality(G)
dict_betweenness_centrality = {k: round(v,2) for k, v in sorted(dict_betweenness_centrality.items(), key=lambda item: item[1], reverse=True)}
dict_betweenness_centrality #Few nodes have high degree as well as high betweenness centrality

#Is network one big, happy family where everyone knows everyone else? OR there are few communities
communities = nx.algorithms.community.greedy_modularity_communities(G)

#Now let us add these community number as attributes
dict_modularity = {} # Create a blank dictionary
for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
    for name in c: # Loop through each person in a community
        dict_modularity[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.

# Now you can add community(also known as modularity) information like we did the other metrics
add_node_attributes(G, dict_modularity, 'community')

#CW: How to find important node for each comunity

#G.nodes(data = True)
#G.edges(data = True)
#exec(open(os.path.abspath('sna_CommonUtils.py')).read())

#Export
#nx.writ*

#triangles : It is very basic of cluster with 3 nodes. It compute the number of triangles
dict_triangles = nx.triangles(G)

#Sort and keep highest on top
dict_triangles = {k: round(v,2) for k, v in sorted(dict_triangles.items(), key=lambda item: item[1], reverse=True)}
dict_triangles

#squares : It is very basic of cluster with 4 nodes. It is fraction of possible squares that exist at the node
dict_square = nx.square_clustering(G)

#Sort and keep highest on top
dict_square = {k: round(v,2) for k, v in sorted(dict_square.items(), key=lambda item: item[1], reverse=True)}
dict_square

#To get nodes for above triangles and squares
#clique: a small close-knit group of people who do not readily allow others to join them.
cliques = nx.enumerate_all_cliques(G)

for clique in cliques:
    if len(clique) > 2:
        print(clique)

#Clusters: It is the fraction of pairs of the node’s friends (that is connections) that are connected with each other.
dict_clustering = nx.clustering(G)

#Sort and keep highest on top
dict_clustering = {k: round(v,2) for k, v in sorted(dict_clustering.items(), key=lambda item: item[1], reverse=True)}
dict_clustering

#sum of all the local clustering coefficients divided by the number of nodes
nx.average_clustering(G)

#Adjacency matrix
A = nx.adjacency_matrix(G,nodelist=G.nodes())
print(A.todense())
print(nx.to_numpy_matrix(G))

#Convert to DF for easy exploations
df = pd.DataFrame(A.todense())
df.columns = list(G.nodes())
df.index = list(G.nodes())
df[df > 0] = 1 # Make readable - 1 means connected
df
#%%Directional Graph (Also known as Asymmetric Networks): Where the relationship is asymmetric (A is related to B, does not
#necessarily means that B is associated with A) is called an Asymmetric network.

#get graph object from data frame
GD = get_graph(train_edges, directional = True)

#Let us see the basic view
nx.draw(GD, with_labels=True, font_weight='bold', node_color = 'lightblue')
show_graph(GD)
