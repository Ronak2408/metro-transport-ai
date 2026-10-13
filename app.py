"""Streamlit app: AI-based public transport optimization (Delhi Metro prototype).
Run:  streamlit run app.py
Needs nodes.csv, edges.csv and transport_ai.py in the same folder."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from transport_ai import (build_system, predict_crowd, find_routes,
                          suggest_frequency, DAY_NAMES)

st.set_page_config(page_title="Metro Crowd-Aware Planner", layout="wide")


@st.cache_resource
def get_system():
    return build_system()


S = get_system()
G, nodes, model, capacity = S["G"], S["nodes"], S["model"], S["capacity"]

st.title("AI-Based Public Transport Optimization")
st.caption("Delhi Metro network (real structure) + simulated passenger demand. "
           "A Random Forest predicts station crowding; the route planner then "
           "balances travel time against crowding.")

stations = sorted(G.nodes)
c1, c2, c3, c4 = st.columns(4)
src = c1.selectbox("From", stations, index=stations.index("Chhattarpur") if "Chhattarpur" in stations else 0)
dst = c2.selectbox("To", stations, index=stations.index("Ashok Park Main") if "Ashok Park Main" in stations else 1)
day = c3.selectbox("Day", DAY_NAMES, index=1)
hour = c4.slider("Hour of travel", 5, 23, 18)

if st.button("Find best route", type="primary"):
    if src == dst:
        st.warning("Pick two different stations.")
        st.stop()
    day_i = DAY_NAMES.index(day)
    pred, ratio = predict_crowd(model, nodes, hour, day_i, capacity)
    res = find_routes(G, src, dst, ratio)
    bpath, bst = res["baseline"]
    opath, ost = res["optimized"]

    def level(x):
        return "High" if x >= 0.8 else "Medium" if x >= 0.5 else "Low"

    a, b = st.columns(2)
    for col, title, path, stt in [(a, "Fastest route (baseline)", bpath, bst),
                                  (b, "Crowd-aware route (AI optimized)", opath, ost)]:
        with col:
            st.subheader(title)
            m1, m2, m3 = st.columns(3)
            m1.metric("Travel time", f"{stt['time_min']} min")
            m2.metric("Avg crowd", f"{int(stt['avg_crowd'] * 100)}%")
            m3.metric("Busiest stop", level(stt["max_crowd"]))
            st.write(f"**{stt['stations']} stations:** " + " → ".join(path))
            st.info(suggest_frequency(stt["max_crowd"]))

    if bpath == opath:
        st.success("Both plans use the same route: no less-crowded alternative exists "
                   "for this trip at this time.")
    else:
        dt = ost["time_min"] - bst["time_min"]
        dc = (bst["avg_crowd"] - ost["avg_crowd"]) * 100
        st.success(f"The AI route takes {dt:+.1f} min but cuts average crowding by {dc:.0f} points.")

    # Route diagram (uses a spring layout, not GPS, because station coordinates are unreliable)
    sub = G.subgraph(set(bpath) | set(opath))
    pos = nx.spring_layout(sub, seed=3)
    fig, ax = plt.subplots(figsize=(9, 4))
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#cccccc")
    nx.draw_networkx_edges(sub, pos, ax=ax, edgelist=list(zip(bpath[:-1], bpath[1:])),
                           edge_color="tab:red", width=2, style="dashed")
    nx.draw_networkx_edges(sub, pos, ax=ax, edgelist=list(zip(opath[:-1], opath[1:])),
                           edge_color="tab:green", width=2)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=60,
                           node_color=[ratio[n] for n in sub.nodes], cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title("Green = AI route, red dashed = fastest route, node colour = predicted crowd")
    ax.axis("off")
    st.pyplot(fig)

with st.expander("Top 10 most crowded stations at the selected time"):
    pred, ratio = predict_crowd(model, nodes, hour, DAY_NAMES.index(day), capacity)
    top = pd.Series(pred).sort_values(ascending=False).head(10).round(0).astype(int)
    st.dataframe(top.rename("Predicted passengers/hour"))

with st.expander("Model quality"):
    st.write(f"MAE: {S['metrics']['MAE']:.1f} passengers/hour, R²: {S['metrics']['R2']:.3f}. "
             "Demand data is simulated, so this shows the pipeline works; it is not a claim about real ridership.")
