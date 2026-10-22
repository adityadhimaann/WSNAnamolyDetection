from src.simulation import generate_dataset, simulate_events, FEATURES

def test_generated_dataset_schema():
    df = generate_dataset(n_samples=300, seed=123)
    assert set(FEATURES).issubset(df.columns)
    assert "true_label" in df.columns
    assert len(df) == 300

def test_simulation_event_count():
    df = simulate_events(n_nodes=10, n_events=40, attack_rate=0.2, seed=9)
    assert len(df) == 40
    assert df["node_id"].nunique() <= 10
