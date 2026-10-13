"""
AI-based Public Transport Optimization (prototype) - Delhi Metro
Part 1: Predict how crowded each station is (Random Forest)
Part 2: Recommend a route that balances travel time AND crowding (Dijkstra)
Part 3: Suggest train frequency (rule-based) from the predicted crowd

Run this file directly to train the model and save charts:  python transport_ai.py
"""
import math
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MIN_PER_HOP = 2.5      # assumed minutes between two adjacent stations
CROWD_PENALTY = 15.0   # extra "minutes" of discomfort for passing a 100% crowded station
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURES = ["Num_Lines", "Is_Interchange", "degree", "betweenness",
            "hour", "day", "is_weekend"]


# ---------------------------------------------------------------- 1. Network
def load_network(nodes_path="nodes.csv", edges_path="edges.csv"):
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["Id"], lines=r["Lines"], num_lines=int(r["Num_Lines"]),
                   interchange=bool(r["Is_Interchange"]))
    for _, r in edges.iterrows():
        G.add_edge(r["Source"], r["Target"], line=r["Line"],
                   time=float(r["Weight"]) * MIN_PER_HOP)
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G)
    nodes["degree"] = nodes["Id"].map(deg)
    nodes["betweenness"] = nodes["Id"].map(btw)
    nodes["Is_Interchange"] = nodes["Is_Interchange"].astype(int)
    return G, nodes


# ------------------------------------------------- 2. Synthetic demand data
def generate_demand(nodes, seed=42):
    """Simulated hourly passengers per station (real ridership is not public)."""
    rng = np.random.default_rng(seed)
    station_factor = rng.uniform(0.85, 1.15, len(nodes))   # hidden station quirks
    base = (200 + 150 * nodes["Num_Lines"] + 60 * nodes["degree"]
            + 2500 * nodes["betweenness"]).values * station_factor
    rows = []
    for i, st in enumerate(nodes["Id"]):
        for day in range(7):
            weekend = day >= 5
            for hour in range(5, 24):
                if weekend:
                    prof = 0.25 + 0.55 * math.exp(-((hour - 13) ** 2) / (2 * 3.0 ** 2))
                else:
                    prof = (0.25 + 1.00 * math.exp(-((hour - 9) ** 2) / (2 * 1.2 ** 2))
                                 + 1.10 * math.exp(-((hour - 18) ** 2) / (2 * 1.5 ** 2)))
                noise = rng.uniform(0.9, 1.1)
                rows.append([st, day, hour, int(base[i] * prof * noise)])
    df = pd.DataFrame(rows, columns=["station", "day", "hour", "passengers"])
    df["is_weekend"] = (df["day"] >= 5).astype(int)
    df = df.merge(nodes[["Id", "Num_Lines", "Is_Interchange", "degree", "betweenness"]],
                  left_on="station", right_on="Id").drop(columns="Id")
    return df


# ------------------------------------------------ 3. AI part 1: demand model
def train_demand_model(df, seed=42):
    X, y = df[FEATURES], df["passengers"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(n_estimators=150, max_depth=14, n_jobs=-1,
                                  random_state=seed)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"MAE": mean_absolute_error(y_te, pred), "R2": r2_score(y_te, pred),
               "mean_passengers": float(y_te.mean())}
    return model, metrics, (y_te.values, pred)


def predict_crowd(model, nodes, hour, day, capacity):
    """Predicted passengers and crowd ratio (0-1) for every station at a given time."""
    X = nodes[["Num_Lines", "Is_Interchange", "degree", "betweenness"]].copy()
    X["hour"], X["day"], X["is_weekend"] = hour, day, int(day >= 5)
    X = X[FEATURES]
    pred = model.predict(X)
    ratio = np.clip(pred / capacity, 0, 1)
    return (dict(zip(nodes["Id"], pred)), dict(zip(nodes["Id"], ratio)))


# ------------------------------------------- 4. AI part 2: route optimization
def route_stats(G, path, ratio):
    time = sum(G[u][v]["time"] for u, v in zip(path[:-1], path[1:]))
    crowds = [ratio[s] for s in path]
    return {"stations": len(path), "time_min": round(time, 1),
            "avg_crowd": round(float(np.mean(crowds)), 2),
            "max_crowd": round(float(np.max(crowds)), 2)}


def find_routes(G, source, target, ratio, penalty=CROWD_PENALTY):
    """Baseline = fastest route. Optimized = fastest route that also avoids crowds."""
    base_path = nx.shortest_path(G, source, target, weight="time")

    def crowd_aware(u, v, d):
        return d["time"] + penalty * ratio[v]

    opt_path = nx.shortest_path(G, source, target, weight=crowd_aware)
    return {"baseline": (base_path, route_stats(G, base_path, ratio)),
            "optimized": (opt_path, route_stats(G, opt_path, ratio))}


# ------------------------------------------------- 5. Frequency suggestion
def suggest_frequency(max_crowd):
    """Simple, explainable rule: more crowd -> shorter wait between trains."""
    if max_crowd >= 0.8:
        return "Very crowded - run a train every 3 min (add extra trains)"
    if max_crowd >= 0.5:
        return "Moderately crowded - run a train every 5 min"
    return "Light crowd - a train every 8 min is enough"


# --------------------------------------------------------------- build all
def build_system(nodes_path="nodes.csv", edges_path="edges.csv"):
    G, nodes = load_network(nodes_path, edges_path)
    df = generate_demand(nodes)
    model, metrics, (y_true, y_pred) = train_demand_model(df)
    capacity = float(df["passengers"].quantile(0.99))
    return {"G": G, "nodes": nodes, "df": df, "model": model, "metrics": metrics,
            "test": (y_true, y_pred), "capacity": capacity}


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = build_system()
    G, nodes, df, model = S["G"], S["nodes"], S["df"], S["model"]
    os.makedirs("figures", exist_ok=True)

    print(f"Stations: {G.number_of_nodes()}, connections: {G.number_of_edges()}")
    print(f"Demand rows generated: {len(df)}")
    print(f"MAE: {S['metrics']['MAE']:.1f} passengers/hour "
          f"(average station-hour = {S['metrics']['mean_passengers']:.0f})")
    print(f"R2 : {S['metrics']['R2']:.3f}")

    # Chart 1: predicted vs actual
    y_true, y_pred = S["test"]
    plt.figure(figsize=(5, 5))
    idx = np.random.default_rng(0).choice(len(y_true), 1500, replace=False)
    plt.scatter(y_true[idx], y_pred[idx], s=6, alpha=0.5)
    m = max(y_true.max(), y_pred.max())
    plt.plot([0, m], [0, m], "r--", label="perfect prediction")
    plt.xlabel("Actual passengers/hour"); plt.ylabel("Predicted passengers/hour")
    plt.title("Demand model: predicted vs actual"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/predicted_vs_actual.png", dpi=150); plt.close()

    # Chart 2: hourly demand profile weekday vs weekend
    prof = df.groupby(["is_weekend", "hour"])["passengers"].mean().unstack(0)
    prof.columns = ["Weekday", "Weekend"]
    prof.plot(figsize=(7, 4), marker="o")
    plt.ylabel("Avg passengers/hour per station"); plt.xlabel("Hour of day")
    plt.title("Demand pattern (simulated)"); plt.tight_layout()
    plt.savefig("figures/hourly_demand.png", dpi=150); plt.close()

    # Chart 3: feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(figsize=(6, 4)); plt.title("What the model uses to predict crowding")
    plt.tight_layout(); plt.savefig("figures/feature_importance.png", dpi=150); plt.close()

    # Route comparison over many random pairs at 6 PM on a weekday
    _, ratio = predict_crowd(model, nodes, 18, 1, S["capacity"])
    rng = np.random.default_rng(1)
    names = list(G.nodes)
    diff, rows = 0, []
    for _ in range(300):
        a, b = rng.choice(names, 2, replace=False)
        r = find_routes(G, a, b, ratio)
        bs, os_ = r["baseline"][1], r["optimized"][1]
        if r["baseline"][0] != r["optimized"][0]:
            diff += 1
            rows.append([a, b, bs["time_min"], os_["time_min"], bs["avg_crowd"], os_["avg_crowd"]])
    print(f"\nRoute changed for {diff} of 300 random trips (6 PM weekday)")
    if rows:
        out = pd.DataFrame(rows, columns=["from", "to", "base_time", "opt_time", "base_crowd", "opt_crowd"])
        print("Avg time change (min):", round((out.opt_time - out.base_time).mean(), 2))
        print("Avg crowd change    :", round((out.opt_crowd - out.base_crowd).mean(), 3))
        print(out.sort_values("base_crowd", ascending=False).head(5).to_string(index=False))
