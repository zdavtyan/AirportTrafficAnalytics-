import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load 2025 T-100-like data
# ---------------------------------------
PATH = r"/Users/zaruhidavtyan/Downloads/T_T100D_SEGMENT_ALL_CARRIER (2).csv"  # <- update if needed

df = pd.read_csv(PATH)

# Keep only 2025 (this file is already 2025, but this makes it explicit)
df_2025 = df[df["YEAR"] == 2025].copy()

print("Rows 2025:", len(df_2025))
print(df_2025.head())

# ---------------------------------------
# 2. Build directed weighted graph for 2025
#    weight = number of records for (ORIGIN, DEST)
# ---------------------------------------
edges = (
    df_2025.groupby(["ORIGIN", "DEST"])
           .size()
           .reset_index(name="weight")
)

print("Distinct routes:", len(edges))

G_2025 = nx.DiGraph()
for _, row in edges.iterrows():
    origin = row["ORIGIN"]
    dest   = row["DEST"]
    w      = row["weight"]
    G_2025.add_edge(origin, dest, weight=w)

print("Graph nodes:", G_2025.number_of_nodes())
print("Graph edges:", G_2025.number_of_edges())

# For robustness, typically use undirected structure
G_year = G_2025.to_undirected()
G_year = nx.Graph(G_year)  # simple graph
N = G_year.number_of_nodes()
print("Undirected graph nodes:", N, "edges:", G_year.number_of_edges())

# ---------------------------------------
# 3. Helper: largest connected component fraction
# ---------------------------------------
def lcc_fraction(G, N0):
    """Relative size of largest connected component vs original N0."""
    if G.number_of_nodes() == 0:
        return 0.0
    comps = nx.connected_components(G)
    largest = max(comps, key=len)
    return len(largest) / N0

# ---------------------------------------
# 4. Robustness: random vs targeted removal
# ---------------------------------------
fractions = np.linspace(0, 0.5, 11)  # remove up to 50% of nodes
n_reps_random = 20

lcc_random   = []
lcc_targeted = []

# Targeted by "strength" = weighted degree using 'weight' from edges
strength = dict(G_year.degree(weight="weight"))
targeted_order = sorted(strength, key=strength.get, reverse=True)
targeted_order = np.array(targeted_order)

for f in fractions:
    k = int(f * N)

    # ----- targeted removal -----
    G_t = G_year.copy()
    if k > 0:
        nodes_to_remove = list(targeted_order[:k])
        G_t.remove_nodes_from(nodes_to_remove)
    lcc_targeted.append(lcc_fraction(G_t, N))

    # ----- random removal (average over repetitions) -----
    vals = []
    for _ in range(n_reps_random):
        G_r = G_year.copy()
        if k > 0:
            nodes_to_remove = np.random.choice(list(G_r.nodes()), size=k, replace=False)
            G_r.remove_nodes_from(nodes_to_remove)
        vals.append(lcc_fraction(G_r, N))
    lcc_random.append(np.mean(vals))

# ---------------------------------------
# 5. Plot robustness curve for LaTeX
# ---------------------------------------
plt.figure(figsize=(6,4))
plt.plot(fractions, lcc_targeted, marker="o", label="Targeted (by strength)")
plt.plot(fractions, lcc_random,   marker="s", label="Random removal")
plt.xlabel("Fraction of airports removed")
plt.ylabel("Relative size of largest component")
plt.title("Robustness of airport network to node removal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("robustness_lcc_2025.pdf", bbox_inches="tight")
plt.close()
