import networkx as nx
import pandas as pd
import measurements as ms
import numpy as np  # for correlations

start_year = 2015
end_year = 2024
filename = "condensed.csv"

# ---------------------------------------------------------------------
# Load data and build yearly directed graphs
# ---------------------------------------------------------------------

# open and read condensed file
my_file = pd.read_csv(filename, sep=",", on_bad_lines="skip", header=None)

num_years = end_year - start_year + 1

# initialize unique graphs (one per year)
graphs = [nx.DiGraph() for _ in range(num_years)]

# fill the graphs
for _, row in my_file.iterrows():
    year = int(row.iloc[0])
    graph_index = year - start_year
    graphs[graph_index].add_edge(row.iloc[1], row.iloc[2], weight=row.iloc[3])

# find LCC and undirected LCC for each graph
LCC = [nx.DiGraph() for _ in range(num_years)]
LCCu = [nx.Graph() for _ in range(num_years)]
for i in range(num_years):
    LCC[i] = ms.LCC_subgraph(graphs[i])
    LCCu[i] = LCC[i].to_undirected()

# ---------------------------------------------------------------------
# Basic counts: nodes, routes, density
# ---------------------------------------------------------------------
print("nodes_routes")
node_count = [0] * len(LCC)
edge_count = [0] * len(LCC)
route_count = [0] * len(LCC)
with open("generated_data/nodes_routes.csv", "wt") as out_file:
    out_file.write("year,nodes,routes\n")
    for i in range(len(LCC)):
        print(i)
        node_count[i] = len(LCC[i].nodes)
        edge_count[i] = len(LCC[i].edges)
        route_count[i] = LCC[i].size(weight="weight")
        out_file.write(f"{i + start_year},{node_count[i]},{int(route_count[i])}\n")

print("density")
with open("generated_data/density.csv", "wt") as out_file:
    for i in range(len(LCC)):
        print(i)
        if node_count[i] > 1:
            dens = edge_count[i] / (node_count[i] * (node_count[i] - 1))
        else:
            dens = 0.0
        out_file.write(f"{dens}\n")

# ---------------------------------------------------------------------
# Degree and strength distributions
# ---------------------------------------------------------------------
print("unweighted degree")
with open("generated_data/unweighted_degree.csv", "wt") as out_file:
    DDu = [[0] for _ in range(len(LCCu))]
    for i in range(len(LCCu)):
        print(i)
        DDu[i] = ms.degree_distribution(LCCu[i])
    eql = len(max(DDu, key=len))
    for i in range(len(DDu)):
        if eql > len(DDu[i]):
            tmp = [0] * (eql - len(DDu[i]))
            DDu[i].extend(tmp)
    for i in range(len(LCCu)):
        line = f"{DDu[i]}"[1:-1]
        out_file.write(f"{line}\n")

print("weighted degree distribution")
with open("generated_data/weighted_degree.csv", "wt") as out_file:
    DDw = [[0] for _ in range(len(LCCu))]
    # calculate node distributions for every LCC (undirected)
    for i in range(len(LCCu)):
        print(i)
        DDw[i] = ms.weighted_degree_distribution(LCCu[i])
    # take longest distribution length
    eql = len(max(DDw, key=len))
    # artificially extend all distributions to be that length
    for i in range(len(DDw)):
        if eql > len(DDw[i]):
            tmp = [0] * (eql - len(DDw[i]))
            DDw[i].extend(tmp)
    # write line to CSV, trimming brackets
    for i in range(len(LCCu)):
        line = f"{DDw[i]}"[1:-1]
        out_file.write(f"{line}\n")

print("DD CCDF")
with open("generated_data/CCDF.csv", "wt") as out_file:
    CCDF = [[0] for _ in range(len(LCCu))]
    # calculate node distributions for every LCC (undirected)
    for i in range(len(LCCu)):
        print(i)
        CCDF[i] = ms.degree_distribution_CCDF(LCCu[i])
    # take longest distribution length
    eql = len(max(CCDF, key=len))
    # artificially extend all distributions to be that length
    for i in range(len(CCDF)):
        if eql > len(CCDF[i]):
            tmp = [0] * (eql - len(CCDF[i]))
            CCDF[i].extend(tmp)
    # write line to CSV, trimming brackets
    for i in range(len(CCDF)):
        line = f"{CCDF[i]}"[1:-1]
        out_file.write(f"{line}\n")

# ---------------------------------------------------------------------
# Distances, clustering
# ---------------------------------------------------------------------
print("diameter")
with open("generated_data/diameter.csv", "wt") as out_file:
    for i in range(len(LCCu)):
        print(i)
        out_file.write(f"{i + start_year},{ms.diameter(LCCu[i])}\n")

print("shortest path length")
avg_spl = [0] * len(LCCu)
with open("generated_data/avg_spl.csv", "wt") as out_file:
    for i in range(len(LCCu)):
        print(i)
        avg_spl[i] = ms.average_shortest_path(LCCu[i])
        out_file.write(f"{avg_spl[i]}\n")

print("cluster coeff")
with open("generated_data/cluster.csv", "wt") as out_file:
    for i in range(len(LCCu)):
        print(i)
        out_file.write(f"{ms.cluster_coefficient(LCCu[i])}\n")

# ---------------------------------------------------------------------
# Node strength (weighted degree, by year)
# ---------------------------------------------------------------------
print("node strength")
with open("generated_data/node_strength.csv", "wt") as out_file:
    strongest = [[] for _ in range(len(LCCu))]
    for i in range(len(LCCu)):
        print(i)
        strongest[i] = list(
            sorted(
                nx.degree(LCCu[i], weight="weight"),
                key=lambda x: x[1],
                reverse=True,
            )
        )
    maxlen = len(max(strongest, key=len))
    for i in range(len(LCCu)):
        for _ in range(maxlen - len(strongest[i])):
            strongest[i].append((0, 0))
    for i in range(len(strongest)):
        line = ""
        for j in range(len(strongest[i])):
            line += f"{strongest[i][j][1]},"
        out_file.write(f"{line[:-1]}\n")

# ---------------------------------------------------------------------
# PageRank: per-year top-10, correlation with strength,
# and averages over all years
# ---------------------------------------------------------------------

print("pagerank top10")

# accumulators for averages over years
strength_sum = {}
pagerank_sum = {}

with open("generated_data/pagerank_top10.csv", "wt") as out_file:
    out_file.write("year,rank,airport,strength,pagerank\n")
    for i, G in enumerate(LCC):
        print(i)
        year = start_year + i

        # weighted PageRank on directed LCC
        pr = nx.pagerank(G, weight="weight")

        # strength (weighted degree) on same directed graph
        strength = dict(G.degree(weight="weight"))

        # accumulate sums for averages
        for n in G.nodes():
            strength_sum[n] = strength_sum.get(n, 0.0) + strength[n]
            pagerank_sum[n] = pagerank_sum.get(n, 0.0) + pr[n]

        # top 10 airports by PageRank for this year
        top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (airport, pr_val) in enumerate(top, start=1):
            out_file.write(
                f"{year},{rank},{airport},{strength.get(airport, 0)},{pr_val}\n"
            )

# Correlation between strength and PageRank (for report justification)
print("pagerank-strength correlation")
with open("generated_data/pagerank_strength_corr.csv", "wt") as out_file:
    out_file.write("year,pearson_corr\n")
    for i, G in enumerate(LCC):
        print(i)
        year = start_year + i

        pr = nx.pagerank(G, weight="weight")
        strength = dict(G.degree(weight="weight"))

        s_vals = []
        pr_vals = []
        for n in G.nodes():
            s_vals.append(strength[n])
            pr_vals.append(pr[n])

        if len(s_vals) > 1:
            corr = float(np.corrcoef(s_vals, pr_vals)[0, 1])
        else:
            corr = float("nan")

        out_file.write(f"{year},{corr}\n")

# Average strength and PageRank over all years
print("average strength and pagerank over years")

avg_strength = {n: strength_sum[n] / num_years for n in strength_sum}
avg_pagerank = {n: pagerank_sum[n] / num_years for n in strength_sum}

# full list for all airports
with open("generated_data/avg_strength_pagerank_all.csv", "wt") as out_file:
    out_file.write("airport,avg_strength,avg_pagerank\n")
    for airport in sorted(avg_strength, key=avg_strength.get, reverse=True):
        out_file.write(
            f"{airport},{avg_strength[airport]},{avg_pagerank.get(airport, 0.0)}\n"
        )

# top 10 airports by average strength with their average PageRank
top10_avg_strength = sorted(
    avg_strength.items(), key=lambda x: x[1], reverse=True
)[:10]

with open("generated_data/avg_strength_pagerank_top10.csv", "wt") as out_file:
    out_file.write("rank,airport,avg_strength,avg_pagerank\n")
    for rank, (airport, s) in enumerate(top10_avg_strength, start=1):
        out_file.write(
            f"{rank},{airport},{s},{avg_pagerank.get(airport, 0.0)}\n"
        )
