import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from .simulation import FEATURES, generate_dataset

def train_models(data=None):
    data = data if data is not None else generate_dataset()
    X = data[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(data[FEATURES].median())
    y = data["true_label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=160, class_weight="balanced_subsample", random_state=42, n_jobs=-1, min_samples_leaf=2),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced")),
        "Dummy baseline": DummyClassifier(strategy="most_frequent")
    }
    for model in models.values():
        model.fit(X_train, y_train)

    iso = IsolationForest(n_estimators=150, contamination=0.08, random_state=42)
    normal_X = X_train[y_train == "Normal"]
    iso.fit(normal_X)
    bundle = {"classifiers": models, "isolation_forest": iso, "feature_columns": FEATURES}
    test_data = X_test.copy()
    test_data["true_label"] = y_test.values
    return bundle, {"test_data": test_data}

def predict_events(events, bundle):
    X = events[bundle["feature_columns"]].copy()
    rf = bundle["classifiers"]["Random Forest"]
    preds = rf.predict(X)
    probabilities = rf.predict_proba(X)
    anomaly = bundle["isolation_forest"].predict(X) == -1
    # Risk is an operational ranking heuristic, not a calibrated probability.
    risk = []
    levels = []
    for pred, anom, prob_row in zip(preds, anomaly, probabilities):
        class_conf = float(np.max(prob_row))
        score = min(100.0, (0 if pred == "Normal" else 55) + (25 if anom else 0) + 20 * (1-class_conf))
        risk.append(round(score, 1))
        levels.append("High" if score >= 75 else ("Medium" if score >= 45 else "Low"))
    out = events.copy()
    out["predicted_attack"] = preds
    out["anomaly_flag"] = anomaly
    out["risk_score"] = risk
    out["risk_level"] = levels
    return out

def evaluate_models(bundle, test_data):
    X = test_data[bundle["feature_columns"]]
    y = test_data["true_label"]
    rows = []
    for name, model in bundle["classifiers"].items():
        start = time.perf_counter()
        pred = model.predict(X)
        elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(X), 1)
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(y, pred, average="macro", zero_division=0), 4),
            "macro_precision": round(precision_score(y, pred, average="macro", zero_division=0), 4),
            "latency_ms_per_event": round(elapsed_ms, 4)
        })
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
