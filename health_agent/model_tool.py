"""The trained classifiers, wrapped as plain Python functions (the agent's tools).

Models come from the active version in models/registry.json (see training.py).
Each version has feature tiers; a prediction uses the tier with the best
measured test F1 among the tiers whose features the user gave. Minimum inputs:
age, glucose, BMI. Preprocessing parameters come from that version's manifest,
so inference always matches training.
"""

from __future__ import annotations

import threading
import warnings

import pandas as pd

from . import training
from .training import COLUMNS, REQUIRED

MODEL_LABELS = {"mlp": "MLP neural network", "naive_bayes": "Gaussian Naive Bayes"}
FIELD_LABELS = {
    "age": "age",
    "glucose": "glucose",
    "bmi": "BMI",
    "insulin": "insulin",
    "blood_pressure": "blood pressure",
    "pregnancies": "number of pregnancies",
    "skin_thickness": "skin-fold thickness",
    "pedigree": "diabetes pedigree (family history score)",
}

_cache: dict[str, dict] = {}  # dataset -> {"version", "manifest", "pipes"}
_lock = threading.Lock()


def default_dataset() -> str:
    available = training.available_datasets()
    return available[0] if available else "pima"


def load_models(dataset: str | None = None, reload: bool = False) -> dict:
    """Load the active version of a dataset once; thread-safe. Only load pickles you trust."""
    with _lock:
        training.register_legacy_v1()
        dataset = dataset or default_dataset()
        target = training.active_version(dataset)
        if target is None:
            raise ValueError(f"no trained models for dataset {dataset!r}")
        state = _cache.get(dataset)
        if reload or state is None or state["version"] != target:
            manifest, pipes = training.load_version(target)
            state = _cache[dataset] = {"dataset": dataset, "version": target, "manifest": manifest, "pipes": pipes}
        return state


def load_all() -> list[str]:
    return [load_models(d)["dataset"] for d in training.available_datasets()]


def models_loaded() -> list[str]:
    return sorted({m for st in _cache.values() for (_, m) in st["pipes"]})


def loaded_versions() -> dict:
    return {d: st["version"] for d, st in _cache.items()}


def tier_summary(dataset: str | None = None) -> list[dict]:
    """Tiers of a dataset's active version with their measured test metrics."""
    manifest = load_models(dataset)["manifest"]
    out = []
    for name, info in manifest["tiers"].items():
        out.append({"tier": name, "features": info["features"], "test_metrics": info.get("test_metrics", {})})
    return out


def _f1(info: dict, model: str) -> float:
    """Cross-validated F1 (steadier than one small test split); falls back to test F1."""
    m = info.get("test_metrics", {}).get(model) or {}
    return m.get("cv_f1_mean") or m.get("f1") or 0.0


def choose_tier(known: dict, model: str = "mlp", dataset: str | None = None) -> tuple[str, list[str]]:
    """Best-F1 tier whose features are all known. Returns (tier, fields_imputed)."""
    manifest = load_models(dataset)["manifest"]
    impute = set(manifest.get("impute_missing", []))
    candidates = []
    for name, info in manifest["tiers"].items():
        lacking = [f for f in info["features"] if known.get(f) is None]
        if not lacking or set(lacking) <= impute:
            candidates.append((len(lacking) == 0, _f1(info, model), len(info["features"]), name, lacking))
    if not candidates:
        raise ValueError("need at least: " + ", ".join(REQUIRED))
    # Prefer tiers needing no imputation, then best F1, then fewer features.
    candidates.sort(key=lambda c: (c[0], c[1], -c[2]), reverse=True)
    _, _, _, name, lacking = candidates[0]
    return name, lacking


def upgrade_hint(known: dict, tier: str, model: str = "mlp", dataset: str | None = None) -> dict | None:
    """The best tier that needs more values than the user gave, if it scored higher."""
    manifest = load_models(dataset)["manifest"]
    current = _f1(manifest["tiers"][tier], model)
    best = None
    for name, info in manifest["tiers"].items():
        extra = [f for f in info["features"] if known.get(f) is None]
        m = info.get("test_metrics", {}).get(model) or {}
        noise = max(0.01, m.get("cv_f1_std") or 0.0)  # only suggest gains bigger than CV noise
        if extra and _f1(info, model) > current + noise:
            if best is None or _f1(info, model) > best["f1"]:
                best = {"tier": name, "ask_for": extra, "f1": _f1(info, model), "current_f1": current}
    return best


def _frame(values: dict) -> pd.DataFrame:
    row = {col: float(values.get(f) or 0.0) for f, col in COLUMNS.items()}  # 0 = not measured
    return pd.DataFrame([row])


def _run(model: str, values: dict, dataset: str | None = None) -> dict:
    state = load_models(dataset)
    dataset = state["dataset"]
    tier, imputed = choose_tier(values, model, dataset)
    info = state["manifest"]["tiers"][tier]
    X = training.transform(_frame(values), info["preprocessing"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probability = float(state["pipes"][(tier, model)].predict_proba(X)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "risk_class": "high" if prediction else "low",
        "prediction": prediction,
        "probability": round(probability, 4),
        "model_used": model,
        "model_label": MODEL_LABELS[model],
        "model_version": state["version"],
        "dataset": dataset,
        "dataset_label": training.DATASETS[dataset]["label"],
        "tier": tier,
        "features_used": info["features"],
        "fields_imputed": imputed,
        "tier_test_metrics": info.get("test_metrics", {}).get(model),
        "more_values_could_help": upgrade_hint(values, tier, model, dataset),
    }


def _values(age, glucose, bmi, **optional) -> dict:
    return {"age": age, "glucose": glucose, "bmi": bmi, **{k: v for k, v in optional.items() if v is not None}}


def predict_diabetes_risk(age: float, glucose: float, bmi: float, model: str = "mlp", dataset: str | None = None, **optional) -> dict:
    """Run one trained model. Optional: insulin, blood_pressure, pregnancies, skin_thickness, pedigree."""
    if model not in MODEL_LABELS:
        raise ValueError(f"model must be one of {sorted(MODEL_LABELS)}")
    result = _run(model, _values(age, glucose, bmi, **optional), dataset)
    result["insulin_imputed"] = "insulin" in result["fields_imputed"]
    return result


def compare_models(age: float, glucose: float, bmi: float, dataset: str | None = None, **optional) -> dict:
    """Run both models on the same inputs and report whether they agree."""
    values = _values(age, glucose, bmi, **optional)
    results = {name: _run(name, values, dataset) for name in MODEL_LABELS}
    agree = len({r["prediction"] for r in results.values()}) == 1
    return {
        "results": results,
        "agree": agree,
        "probability_gap": round(abs(results["mlp"]["probability"] - results["naive_bayes"]["probability"]), 4),
        "insulin_imputed": any("insulin" in r["fields_imputed"] for r in results.values()),
    }


# Backwards-compatible helper used by the original tests: legacy v1 features.
def build_features(age, glucose, bmi, insulin):
    df = _frame({"age": age, "glucose": glucose, "bmi": bmi, "insulin": insulin})
    return training.transform(df, training.LEGACY_PREP)
