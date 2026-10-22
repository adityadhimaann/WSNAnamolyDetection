import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from src.simulation import generate_dataset, simulate_events, FEATURES, LABELS
from src.models import train_models, evaluate_models, predict_events

st.set_page_config(page_title="WSN Sentinel | Security Lab", page_icon="🛡️", layout="wide")

st.title("🛡️ WSN Sentinel")
st.caption("Software-only Wireless Sensor Network anomaly detection and security simulation")
st.info("Research prototype: simulated telemetry only. Do not interpret results as validation on real sensor hardware or a live network.")

@st.cache_resource
def get_models():
    train = generate_dataset(n_samples=12000, seed=42)
    return train_models(train)

models, model_info = get_models()

with st.sidebar:
    st.header("Simulation controls")
    n_nodes = st.slider("Virtual sensor nodes", 10, 200, 50, 10)
    n_events = st.slider("Events per simulation", 20, 500, 120, 20)
    attack_rate = st.slider("Injected attack rate (%)", 0, 40, 15, 5)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=7, step=1)
    run = st.button("▶ Run simulation", type="primary", use_container_width=True)
    st.divider()
    st.caption("Models train on generated labelled telemetry at startup. The evaluation tab uses a held-out synthetic test set.")

if "events" not in st.session_state:
    st.session_state.events = simulate_events(n_nodes=50, n_events=120, attack_rate=0.15, seed=7)
    st.session_state.has_run = False
if run:
    st.session_state.events = simulate_events(n_nodes=n_nodes, n_events=n_events, attack_rate=attack_rate/100, seed=int(seed))
    st.session_state.has_run = True

events = st.session_state.events
scored = predict_events(events, models)

alerts = scored[scored["predicted_attack"] != "Normal"]
total = len(scored)
attack_truth = int((scored["true_label"] != "Normal").sum())
detected = int((scored["predicted_attack"] != "Normal").sum())
critical = int((scored["risk_level"] == "High").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Virtual nodes", f"{scored['node_id'].nunique()}")
c2.metric("Events simulated", f"{total:,}")
c3.metric("Predicted suspicious events", f"{detected:,}")
c4.metric("High-risk alerts", f"{critical:,}")

tab1, tab2, tab3, tab4 = st.tabs(["Live monitoring", "Node explorer", "Model evaluation", "Methodology"])

with tab1:
    st.subheader("Network event stream")
    if st.session_state.has_run:
        st.success("New simulation generated. Predictions below are produced by the trained models.")
    left, right = st.columns([1.5, 1])
    with left:
        fig = px.scatter(scored.sort_values("timestamp"), x="timestamp", y="temperature",
                         color="predicted_attack", symbol="risk_level",
                         hover_data=["node_id", "battery", "packet_rate", "packet_delivery", "true_label"],
                         title="Sensor readings and predicted class")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = scored["predicted_attack"].value_counts().rename_axis("class").reset_index(name="events")
        fig2 = px.pie(counts, names="class", values="events", hole=0.45, title="Predicted event distribution")
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Latest alerts")
    if alerts.empty:
        st.success("No events were classified as attacks in this run.")
    else:
        st.dataframe(alerts.sort_values("risk_score", ascending=False)[
            ["timestamp", "node_id", "predicted_attack", "risk_level", "risk_score",
             "temperature", "battery", "packet_rate", "packet_delivery", "anomaly_flag"]
        ].head(50), use_container_width=True, hide_index=True)
    st.download_button("Download all scored events (CSV)", scored.to_csv(index=False).encode("utf-8"),
                       file_name="wsn_simulation_results.csv", mime="text/csv")

with tab2:
    st.subheader("Inspect a virtual sensor node")
    selected = st.selectbox("Select node", sorted(scored["node_id"].unique()))
    node = scored[scored["node_id"] == selected].sort_values("timestamp")
    last = node.iloc[-1]
    a, b, c = st.columns(3)
    a.metric("Last predicted class", str(last["predicted_attack"]))
    b.metric("Risk score", f"{last['risk_score']:.1f}/100")
    c.metric("Anomaly flag", "Yes" if bool(last["anomaly_flag"]) else "No")
    st.line_chart(node.set_index("timestamp")[["temperature", "humidity", "battery", "packet_delivery"]])
    st.write("**Latest event details**")
    st.dataframe(node.tail(10), use_container_width=True, hide_index=True)
    st.caption("Risk score is a prototype ranking derived from model outputs, not a calibrated probability of compromise.")

with tab3:
    st.subheader("Held-out synthetic benchmark")
    st.warning("These metrics evaluate models on a separate synthetic test split. They are not results on the real WSN-DS dataset.")
    metrics = evaluate_models(models, model_info["test_data"])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    fig = px.bar(metrics, x="model", y=["accuracy", "macro_f1", "macro_recall"], barmode="group",
                 title="Model comparison (synthetic test set)")
    st.plotly_chart(fig, use_container_width=True)
    st.write("The detailed classification report and confusion matrices can be generated in the experiment script described in README.md.")

with tab4:
    st.subheader("System workflow")
    st.markdown("""
1. **Simulation:** create virtual sensor-node telemetry.
2. **Attack injection:** inject controlled patterns such as flooding, blackhole-like packet drops, grayhole-like selective drops, TDMA/scheduling disruption, and sensor-value manipulation.
3. **Unsupervised detector:** Isolation Forest is fitted on normal telemetry to flag unusual feature combinations.
4. **Supervised classifier:** Random Forest predicts a class from labelled synthetic training examples.
5. **Risk triage:** combine predicted class and anomaly flag into a simple high/medium/low alert ranking.
6. **Evaluation:** compare baseline classifiers on held-out synthetic data, then repeat the protocol on WSN-DS for the research study.
""")
    st.subheader("Feature definitions")
    st.dataframe(pd.DataFrame([
        ["temperature", "Simulated temperature reading (°C)"],
        ["humidity", "Simulated relative humidity (%)"],
        ["battery", "Remaining node battery (%)"],
        ["packet_rate", "Packets attempted per time interval"],
        ["packet_delivery", "Fraction of packets successfully delivered (0–1)"],
        ["latency_ms", "Simulated network delay in milliseconds"],
        ["retransmissions", "Number of packet retransmissions"],
        ["neighbor_count", "Number of neighboring nodes observed"],
        ["signal_strength", "Simulated signal strength (dBm)"],
    ], columns=["Feature", "Meaning"]), use_container_width=True, hide_index=True)
    st.subheader("Important research limitations")
    st.markdown("""
- Synthetic data is useful for software demonstration but cannot establish real-world effectiveness.
- The simulated attack signatures are simplified and may be easier to classify than real attacks.
- A high risk score is not a calibrated probability.
- Before a capstone claim, evaluate on the public WSN-DS dataset with leakage-safe splits and compare against prior work.
""")
