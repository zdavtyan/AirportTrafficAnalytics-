import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

#configurations
DELAY_PATH = r"/Users/zaruhidavtyan/Downloads/ot_delaycause1_DL (1)/Airline_Delay_Cause.csv"

# Path to avg strength 
STRENGTH_PATH = r"/Users/zaruhidavtyan/Downloads/flightGraphCode/generated_data/avg_strength_pagerank_all.csv"

# output directory for charts 
OUTPUT_DIR = r"/Users/zaruhidavtyan/Downloads/flightGraphCode/generated_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time window: Dec 2015 – Dec 2024
START_YEAR = 2015
END_YEAR   = 2024

# Thresholds
MIN_FLIGHTS_PER_AIRPORT_MONTH = 20   # airport-months with fewer flights are too noisy
MIN_MONTHS_PER_PAIR           = 6    # min months for computing a correlation
RHO_THRESHOLD_FOR_GRAPH       = 0.3  # only edges with rho_pos >= this go into the graph

TOP_N_STRENGTH = 20          # how many top strength airports to keep
T_STEPS        = 10          # number of simulation steps
ALPHA          = 0.8         

# ============================================================
# 1. LOAD DATA & AGGREGATE BY AIRPORT-MONTH
# ============================================================
df = pd.read_csv(DELAY_PATH)

print("Columns:", df.columns.tolist())

# Filter time: Dec 2015 – Dec 2024
mask_time = (
    ((df["year"] > START_YEAR) & (df["year"] < END_YEAR)) |
    ((df["year"] == START_YEAR) & (df["month"] >= 12)) |
    ((df["year"] == END_YEAR) & (df["month"] <= 12))
)
df = df[mask_time].copy()

# Aggregate across carriers: one row per (year, month, airport)
airport_month = (
    df.groupby(["year", "month", "airport"], as_index=False)
      .agg(
          arr_flights_sum = ("arr_flights", "sum"),
          arr_delay_sum   = ("arr_delay", "sum"),
          arr_del15_sum   = ("arr_del15", "sum"),
      )
)

print("Airport-month rows before filtering:", len(airport_month))

# Filter out airport-months with very few flights
airport_month = airport_month[
    airport_month["arr_flights_sum"] >= MIN_FLIGHTS_PER_AIRPORT_MONTH
].copy()

print("Airport-month rows after flight threshold:",
      len(airport_month))

# Mean delay per arriving flight
airport_month["avg_delay_per_flight"] = (
    airport_month["arr_delay_sum"] / airport_month["arr_flights_sum"]
)

# Monthly baseline delay over all airports (keep this BEFORE filtering to top-20)
baseline = (
    airport_month.groupby(["year", "month"], as_index=False)
                 .agg(baseline_delay=("avg_delay_per_flight", "mean"))
)

airport_delay = airport_month.merge(
    baseline,
    on=["year", "month"],
    how="left"
)

# Excess delay = airport avg delay - monthly baseline
airport_delay["excess_delay"] = (
    airport_delay["avg_delay_per_flight"]
    - airport_delay["baseline_delay"]
)

# Period index for time
airport_delay["ym"] = pd.PeriodIndex(
    year=airport_delay["year"],
    month=airport_delay["month"],
    freq="M"
)

print("Distinct airports (before top-20 filter):", airport_delay["airport"].nunique())
print("Distinct months:", airport_delay["ym"].nunique())

# Load top strength airports
strength_df = pd.read_csv(STRENGTH_PATH)  # columns: airport, avg_strength, avg_pagerank
strength_df = strength_df.sort_values("avg_strength", ascending=False)
top20_airports = strength_df["airport"].head(TOP_N_STRENGTH).tolist()

print(f"Top {TOP_N_STRENGTH} airports by avg strength:", top20_airports)

# Filter the delay data to only those top-20 airports
airport_delay = airport_delay[airport_delay["airport"].isin(top20_airports)].copy()
print("Distinct airports after top-20 filter:", airport_delay["airport"].nunique())

# Pivot to time × airport matrix
pivot = airport_delay.pivot_table(
    index="ym",
    columns="airport",
    values="excess_delay"
).sort_index()

print("Pivot shape (months × airports):", pivot.shape)

airports = list(pivot.columns)
months   = list(pivot.index)

# Paiewise lagged correlations  E_i(t) vs E_j(t+1)
rows = []

for ai in airports:
    Ei = pivot[ai].to_numpy() 
    Ei_t = Ei[:-1]             

    for aj in airports:
        if ai == aj:
            continue
        Ej = pivot[aj].to_numpy()
        Ej_t1 = Ej[1:]         #

        mask = ~np.isnan(Ei_t) & ~np.isnan(Ej_t1)
        n = mask.sum()
        if n < MIN_MONTHS_PER_PAIR:
            continue

        x = Ei_t[mask]
        y = Ej_t1[mask]

        rho = np.corrcoef(x, y)[0, 1]
        rows.append((ai, aj, rho, n))

edge_corr_df = pd.DataFrame(
    rows, columns=["origin", "dest", "rho", "n_obs"]
)

edge_corr_df = edge_corr_df.dropna(subset=["rho"]).copy()
edge_corr_df["rho_pos"] = edge_corr_df["rho"].clip(lower=0)

print("Number of airport pairs with valid correlation:", len(edge_corr_df))
edge_corr_df.head()

# Histongrams for rho 
plt.figure(figsize=(6,4))
plt.hist(edge_corr_df["rho"], bins=40, edgecolor="black", alpha=0.7)
plt.xlabel(r"Lagged correlation $\rho_{ij}$")
plt.ylabel("Count of airport pairs")
plt.title("Distribution of delay propagation coefficients")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rho_histogram.pdf"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,4))
plt.hist(edge_corr_df["rho_pos"], bins=40, edgecolor="black", alpha=0.7)
plt.xlabel(r"Positive component $\rho_{ij}^+$")
plt.ylabel("Count of airport pairs")
plt.title("Distribution of positive delay propagation strengths")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rho_pos_histogram.pdf"), bbox_inches="tight")
plt.close()

print("\n=== Edge-level rho summary ===")
print(edge_corr_df["rho"].describe())

print("\n=== Edge-level rho_pos summary ===")
print(edge_corr_df["rho_pos"].describe())

# Airport-llevel propagtion indicies(onlt top-20 airports)
airport_out = (
    edge_corr_df.groupby("origin", as_index=False)
                .agg(
                    P_out=("rho_pos", "mean"),
                    n_out_pairs=("rho_pos", "size")
                )
                .rename(columns={"origin": "airport"})
)

airport_in = (
    edge_corr_df.groupby("dest", as_index=False)
                .agg(
                    P_in=("rho_pos", "mean"),
                    n_in_pairs=("rho_pos", "size")
                )
                .rename(columns={"dest": "airport"})
)

propagation_scores = (
    airport_out.merge(airport_in, on="airport", how="outer")
)

for col in ["P_out", "P_in", "n_out_pairs", "n_in_pairs"]:
    propagation_scores[col] = propagation_scores[col].fillna(0)

print("\nTop 10 airports by P_out (delay spreaders), restricted to top-20 strength:")
print(
    propagation_scores.sort_values("P_out", ascending=False).head(10)
)

# Build a table of propagation scores for all top-20 airports
strong_edges = edge_corr_df[edge_corr_df["rho_pos"] >= RHO_THRESHOLD_FOR_GRAPH].copy()
print("Edges in propagation graph (rho_pos >= threshold):", len(strong_edges))

G_prop = nx.DiGraph()
for _, row in strong_edges.iterrows():
    if (row["origin"] in top20_airports) and (row["dest"] in top20_airports):
        G_prop.add_edge(row["origin"], row["dest"], weight=row["rho_pos"])

# Induced subgraph on the top-20 hubs (some may become isolated)
G_sub = G_prop.subgraph(top20_airports).copy()
print("Nodes in propagation subgraph:", G_sub.number_of_nodes())
print("Edges in propagation subgraph:", G_sub.number_of_edges())

# Node colors = P_out
node_Pout = {row["airport"]: row["P_out"] for _, row in propagation_scores.iterrows()}
node_color = [node_Pout.get(n, 0.0) for n in G_sub.nodes()]

edges = G_sub.edges()
weights = [G_sub[u][v]["weight"] for u, v in edges]

plt.figure(figsize=(7,7))
pos = nx.spring_layout(G_sub, seed=0)

nodes = nx.draw_networkx_nodes(G_sub, pos,
                               node_size=300,
                               node_color=node_color,
                               cmap="viridis")
nx.draw_networkx_edges(G_sub, pos,
                       arrows=True,
                       width=[2*w for w in weights],
                       alpha=0.7)
nx.draw_networkx_labels(G_sub, pos, font_size=8)

plt.colorbar(nodes, label=r"$P_{\text{out}}$ (delay spread index)")
plt.title(f"Propagation graph on top {TOP_N_STRENGTH} strength hubs\n(edges: $\\rho^+_{{ij}} \\geq {RHO_THRESHOLD_FOR_GRAPH}$)")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "propagation_graph_top20.pdf"), bbox_inches="tight")
plt.close()

# Delay propagation simulation starting from a top hub

nodes_list = list(G_sub.nodes())
if not nodes_list:
    raise RuntimeError("Propagation subgraph has no nodes - check thresholds/time window.")

idx = {a: i for i, a in enumerate(nodes_list)}
n = len(nodes_list)

# Build row-normalized weight matrix W for G_sub
W = np.zeros((n, n))
for u, v, data in G_sub.edges(data=True):
    i = idx[u]
    j = idx[v]
    W[i, j] = data["weight"]

row_sums = W.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
W = W / row_sums

# Find the top-strength hub present in the subgraph
strength_in_sub = strength_df[strength_df["airport"].isin(nodes_list)].copy()
if not strength_in_sub.empty:
    shock_airport = strength_in_sub.sort_values("avg_strength", ascending=False)["airport"].iloc[0]
else:
    shock_airport = nodes_list[0]  # fallback

print("Shock starts at airport:", shock_airport)

shock_index = idx[shock_airport]

# Simulate x_{t+1} = alpha * x_t * W
history = np.zeros((T_STEPS + 1, n))
history[0, shock_index] = 1.0

for t in range(T_STEPS):
    history[t + 1] = ALPHA * history[t] @ W

total_intensity = history.sum(axis=1)

plt.figure(figsize=(6,4))
plt.plot(range(T_STEPS + 1), total_intensity, marker="o")
plt.xlabel("Simulation step")
plt.ylabel("Total delay intensity (arbitrary units)")
plt.title(f"Delay propagation simulation starting from {shock_airport}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "delay_sim_total_intensity.pdf"), bbox_inches="tight")
plt.close()

# Pick 5 airports with highest P_out in this subgraph to show
sub_scores = propagation_scores.set_index("airport").loc[nodes_list]
top5_airports = sub_scores.sort_values("P_out", ascending=False).head(5).index.tolist()
top5_indices = [idx[a] for a in top5_airports]

plt.figure(figsize=(7,4))
for a, i in zip(top5_airports, top5_indices):
    plt.plot(range(T_STEPS + 1), history[:, i], marker="o", label=a)

plt.xlabel("Simulation step")
plt.ylabel("Delay intensity (arbitrary units)")
plt.title("Delay propagation over time for top 5 spreading airports\n(within top-20 strength hubs)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "delay_sim_top5_airports.pdf"), bbox_inches="tight")
plt.close()
