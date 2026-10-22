import numpy as np
import pandas as pd

FEATURES = [
    "temperature", "humidity", "battery", "packet_rate", "packet_delivery",
    "latency_ms", "retransmissions", "neighbor_count", "signal_strength"
]
LABELS = ["Normal", "Blackhole", "Grayhole", "Flooding", "TDMA", "SensorManipulation"]

def _sample_class(rng, label, n):
    # Plausible synthetic ranges for demonstration only, not physical calibration.
    temperature = rng.normal(24, 3, n)
    humidity = np.clip(rng.normal(55, 12, n), 5, 100)
    battery = np.clip(rng.normal(75, 15, n), 1, 100)
    packet_rate = np.clip(rng.normal(35, 8, n), 1, 100)
    packet_delivery = np.clip(rng.normal(0.96, 0.025, n), 0.70, 1.0)
    latency_ms = np.clip(rng.normal(45, 12, n), 5, 200)
    retransmissions = np.clip(rng.poisson(2, n), 0, 20)
    neighbor_count = rng.integers(3, 12, n)
    signal_strength = np.clip(rng.normal(-60, 8, n), -100, -25)

    if label == "Blackhole":
        packet_delivery = np.clip(rng.normal(0.20, 0.12, n), 0.0, 0.55)
        latency_ms = np.clip(rng.normal(90, 25, n), 15, 250)
        retransmissions = np.clip(rng.poisson(7, n), 0, 25)
    elif label == "Grayhole":
        packet_delivery = np.clip(rng.normal(0.62, 0.13, n), 0.20, 0.90)
        retransmissions = np.clip(rng.poisson(5, n), 0, 25)
    elif label == "Flooding":
        packet_rate = np.clip(rng.normal(90, 10, n), 55, 150)
        latency_ms = np.clip(rng.normal(120, 30, n), 30, 300)
        battery = np.clip(battery - rng.uniform(5, 20, n), 1, 100)
    elif label == "TDMA":
        latency_ms = np.clip(rng.normal(160, 35, n), 60, 350)
        packet_delivery = np.clip(rng.normal(0.72, 0.12, n), 0.25, 0.95)
        retransmissions = np.clip(rng.poisson(6, n), 0, 25)
    elif label == "SensorManipulation":
        high_temp = rng.random(n) < 0.5
        temperature = np.where(high_temp, rng.normal(48, 5, n), rng.normal(5, 2, n))
        high_humidity = rng.random(n) < 0.5
        humidity = np.clip(np.where(high_humidity, rng.normal(95, 3, n), rng.normal(10, 5, n)), 0, 100)
    return pd.DataFrame({
        "temperature": temperature, "humidity": humidity, "battery": battery,
        "packet_rate": packet_rate, "packet_delivery": packet_delivery,
        "latency_ms": latency_ms, "retransmissions": retransmissions,
        "neighbor_count": neighbor_count, "signal_strength": signal_strength,
        "true_label": label
    })

def generate_dataset(n_samples=12000, seed=42):
    rng = np.random.default_rng(seed)
    labels = ["Normal", "Blackhole", "Grayhole", "Flooding", "TDMA", "SensorManipulation"]
    weights = [0.72, 0.06, 0.06, 0.06, 0.05, 0.05]
    counts = rng.multinomial(n_samples, weights)
    frames = [_sample_class(rng, label, int(n)) for label, n in zip(labels, counts)]
    df = pd.concat(frames, ignore_index=True)
    df["node_id"] = [f"SN-{i:03d}" for i in rng.integers(1, 101, len(df))]
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="s")
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def simulate_events(n_nodes=50, n_events=120, attack_rate=0.15, seed=7):
    rng = np.random.default_rng(seed)
    labels = np.array(LABELS)
    probs = np.array([1-attack_rate, attack_rate/5, attack_rate/5, attack_rate/5, attack_rate/5, attack_rate/5])
    probs = probs / probs.sum()
    chosen = rng.choice(labels, size=n_events, p=probs)
    frames = []
    for label in chosen:
        frames.append(_sample_class(rng, str(label), 1))
    df = pd.concat(frames, ignore_index=True)
    df["node_id"] = [f"SN-{i:03d}" for i in rng.integers(1, n_nodes+1, len(df))]
    df["timestamp"] = pd.date_range(pd.Timestamp.now().floor("s"), periods=len(df), freq="s")
    df["true_label"] = chosen
    return df.sort_values("timestamp").reset_index(drop=True)
