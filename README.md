# WSN Sentinel — Software-only Sensor Network Security Simulation

**Capstone title:** An Intelligent Approach for Anomaly Detection and Security Enhancement in Sensor Networks

This is an original educational prototype for simulating virtual sensor nodes, injecting simplified attack-like patterns, training ML models, and inspecting predictions in a Streamlit dashboard. It requires no physical hardware.

> **Research integrity:** The current dashboard models use generated synthetic data. Synthetic metrics are not evidence of real-world security effectiveness. For the capstone paper, add an independent experiment using the public WSN-DS dataset and report only results you actually reproduce.

## Features
- Virtual sensor nodes and time-stamped telemetry
- Normal, Blackhole-like, Grayhole-like, Flooding, TDMA disruption, and sensor-manipulation patterns
- Random Forest multiclass prediction
- Isolation Forest anomaly flag trained on synthetic normal data
- Combined risk-ranking heuristic (not a calibrated compromise probability)
- Interactive Streamlit charts, node explorer, alert table, CSV export
- Held-out synthetic benchmark comparison: Dummy, Decision Tree, Random Forest, Logistic Regression
- Reproducible seed and configurable event count / attack rate

## Run locally (macOS, Linux, or Windows)

```bash
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Files
```
app.py                  Streamlit dashboard
src/simulation.py       Synthetic telemetry and attack-pattern generator
src/models.py           ML training, scoring, and evaluation
tests/                  Starter location for tests
requirements.txt        Python dependencies
```

## Research workflow for the capstone
1. Use this prototype to demonstrate the software-only architecture.
2. Inspect the baseline repository: https://github.com/Kaleemullah-Younas/Intrusion-Detection-System
3. Check source-code license before copying or redistributing any code. The repository metadata inspected during preparation did not expose a license; if no license file exists, do not assume permission to reuse its code.
4. Download WSN-DS from a documented source and record the exact URL, date, checksum, schema, and label distribution.
5. Implement a separate dataset adapter. Do not assume WSN-DS columns match the synthetic features in `src/simulation.py`.
6. Split data before fitting preprocessing, feature selection, or resampling. Apply any resampling only to training folds.
7. Compare the baseline with your proposed approach using macro-F1, per-class precision/recall, confusion matrix, false-positive rate, inference latency, and model size.
8. Use the same held-out test split for all models. Do not tune on the test set.
9. Replace planned-result placeholders in the paper with your measured outputs.
10. Discuss the limitations of simulated traffic and dataset shift.

## Proposed research question
Does a combined anomaly flag and multiclass intrusion classifier, evaluated with class-aware metrics and inference-cost measurements, offer a useful trade-off compared with simpler baseline classifiers?

This is a research question to test, not a claim of novelty or improvement.

## Attack simulation semantics
- **Blackhole-like:** extremely low packet delivery.
- **Grayhole-like:** selective/intermediate packet loss.
- **Flooding:** elevated packet rate and latency.
- **TDMA disruption:** elevated latency, retransmissions, and packet loss.
- **Sensor manipulation:** extreme synthetic temperature/humidity readings.

These are simplified signatures created for an educational simulation, not protocol-accurate attacks or a validated digital twin.

## Reproducibility
The model generator uses fixed seeds for its training data and simulation controls. Record your Python version, dependency versions, hardware, data source, and exact command for each experiment.

## References to start with
- Almomani, I., Al-Kasasbeh, B., & Al-Akhras, M. (2016). WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks. *Journal of Sensors*, 2016, Article 4731953. https://doi.org/10.1155/2016/4731953
- Talukder et al. (2025). A hybrid machine learning model for intrusion detection in wireless sensor networks leveraging data balancing and dimensionality reduction. *Scientific Reports*. https://doi.org/10.1038/s41598-025-87028-1
- Baseline research/code reference: https://github.com/Kaleemullah-Younas/Intrusion-Detection-System
