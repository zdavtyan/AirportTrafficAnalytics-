import networkx as nx
import numpy as np
import math

#degree distribution
def degree_distribution(S):
    #networkX function to obtain degree per node (stored as list of tuples)
    Degs = nx.degree(S)
    #array where ith element represents number of nodes with degree i
    DD = [0]*len(S.nodes())

    #iterate through each node's degree, incrementing respective counter
    for i in Degs:
        x = i[1]
        DD[x] += 1

    #remove trailing 0s on resulting array
    while DD[-1] == 0:
        DD.pop(-1)

    #shrink numbers to be % of total nodes
    amt = len(S.nodes)
    for i in range(len(DD)):
        DD[i] = DD[i] / amt
    return DD

def weighted_degree_distribution(S):
    #networkX function to obtain degree per node (stored as list of tuples)
    Degs = nx.degree(S, weight="weight")
    #array where ith element represents number of nodes with degree i
    DD = [0]*len(S.nodes())

    #iterate through each node's degree, incrementing respective counter
    for i in Degs:
        x = i[1]
        if x>=len(DD):
            tmp=[0]*(x-len(DD)+1)
            DD.extend(tmp)
        DD[x] += 1

    #shrink numbers to be % of total nodes
    amt = len(S.nodes)
    for i in range(len(DD)):
        DD[i] = DD[i] / amt
    return DD

#degree distribution
def degree_sequence(S):
    print("ds")
    #networkX function to obtain degree per node (stored as list of tuples)
    Degs = nx.degree(S)
    #array where ith element represents number of nodes with degree i
    DS = []
    #iterate through each node's degree, incrementing respective counter
    for i in Degs:
        DS.append(i[1])
    print("done")
    return DS

def node_strength(S):
    #networkX function to obtain degree per node (stored as list of tuples)
    Degs = nx.degree(S, weight="weight")
    NS = [x[1] for x in Degs]
    return Degs

def degree_distribution_CCDF(S):
    #networkX function to obtain degree per node (stored as list of tuples)
    Degs = nx.degree(S, weight="weight")
    #array where ith element represents number of nodes with degree i
    DD = [0]*len(S.nodes())

    #iterate through each node's degree, incrementing respective counter
    for i in Degs:
        x = i[1]
        if x>=len(DD):
            tmp=[0]*(x-len(DD)+1)
            DD.extend(tmp)
        DD[x] += 1

    #remove trailing 0s on resulting array
    while DD[-1] == 0:
        DD.pop(-1)

    y_data = np.array(DD)
    cdf = y_data.cumsum() / y_data.sum()
    ccdf = 1 - cdf
    ccdf = list(ccdf)
    for i in range(len(ccdf)):
        ccdf[i]=float(ccdf[i])
    return ccdf

#distribution of local clustering coefficient
def cluster_distribution(S):
    #networkX function to obtain C per node (stored as dictionary)
    clst = nx.clustering(S)
    #array where ith element contains count of nodes with C of i*0.001 to (i+1)*0.001
    DC = [0]*1001
    #iterate through cluster coeff dictionary of each node
    for i in clst.keys():
        #increment counter for respective C value
        DC[math.floor(clst[i]*1000)] += 1

    #remove trailing 0s on resulting array
    while DC[-1] == 0:
        DC.pop(-1)

    #shrink values to be % of nodes
    amt = len(S.nodes)
    for i in range(len(DC)):
        DC[i] = DC[i] / amt
    return DC

#global clustering coefficient (number)
def cluster_coefficient(S):
    return nx.average_clustering(S)

#distribution of shortest path lengths (plot)
def shortest_path_distribution(S):
    #networkX function all pairs shortest paths, convert to dictionary of dictionary
    SP = dict(nx.all_pairs_shortest_path_length(S))
    #array where ith element contains number of shortest paths with length i
    PL = [0]*len(SP.keys())
    #iterate through starting nodes
    for start in SP.keys():
        #iterate through ending nodes, increment corresponding path length element
        for end in SP.keys():
            PL[SP[start][end]] += 1

    #remove trailing 0s
    while PL[-1] == 0:
        PL.pop(-1)

    # shrink values to be % of nodes
    amt = len(S.nodes)*(len(S.nodes)-1)
    for i in range(len(PL)):
        PL[i] = PL[i] / amt

    return PL

#average shortest path
def average_shortest_path(S):
    #average shortest path length (number)
    SP = dict(nx.all_pairs_shortest_path_length(S))
    sum = 0

    #iterate through start/end combinations, adding path length to sum
    for start in SP.keys():
        for end in SP.keys():
            sum += SP[start][end]

    #divide sum by total number of start/end combinations in S
    sum /= (len(S.nodes))*(len(S.nodes)-1)
    return sum

#graph diameter
def diameter(S):
    return nx.diameter(S)

#creates a subgraph out of the nodes found in LCC
def LCC_subgraph(G):
    LCC = max(nx.weakly_connected_components(G), key=len)
    S = G.subgraph(LCC)
    return S