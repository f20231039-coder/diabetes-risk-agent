"""Data-driven training with feature tiers and versioned models.

- Minimum inputs to predict: age, glucose, BMI ("minimum" tier).
- Each extra value the user knows can unlock a tier trained on more features.
  The tier used for a prediction is the one with the best *measured* test F1
  among the tiers whose features are all known, so extra values are only used
  when they actually helped on held-out data.
- Every training run is a new version in models/versions/vN with a manifest
  (preprocessing parameters + metrics). A new version becomes active only if
  its MLP F1 on the standard tier does not drop versus the active version,
  measured on the SAME test rows.

CLI:  python -m health_agent.training            # train on data/, promote if not worse
      python -m health_agent.training --force    # always activate
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET = DATA_DIR / "diabetes_full.csv"
# Each dataset gets its own model versions; they are never merged because the
# measurements differ (Pima glucose = 2-hour OGTT, NHANES glucose = fasting).
DATASETS = {
    "nhanes": {"path": DATA_DIR / "nhanes.csv", "label": "NHANES 2009-2018 (US adults, fasting glucose)"},
    "pima": {"path": DATASET, "label": "Pima Indians (women 21+, 2-hour glucose test)"},
    # Undocumented origin, looks synthetic, label tied to a glucose cut-off: comparison only.
    "kaggle": {"path": DATA_DIR / "kaggle.csv", "label": "Kaggle diabetes prediction (likely synthetic; glucose test type unknown)"},
}
DEFAULT_DATASET = "nhanes"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
LEGACY_MODELS_DIR = ROOT / "legacy" / "models"  # the original assignment's pickles
VERSIONS_DIR = MODELS_DIR / "versions"
REGISTRY = MODELS_DIR / "registry.json"

# API field name -> dataset column
COLUMNS = {
    "age": "Age",
    "glucose": "Glucose",
    "bmi": "BMI",
    "insulin": "Insulin",
    "blood_pressure": "BloodPressure",
    "pregnancies": "Pregnancies",
    "skin_thickness": "SkinThickness",
    "pedigree": "DiabetesPedigreeFunction",
}
TARGET = "Outcome"
# In Pima, 0 means "not measured" for these columns.
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

REQUIRED = ["age", "glucose", "bmi"]
TIERS: dict[str, list[str]] = {
    "minimum": ["age", "glucose", "bmi"],
    "standard": ["age", "glucose", "bmi", "insulin"],
    "extended": ["age", "glucose", "bmi", "blood_pressure", "pregnancies"],
    "extended_insulin": ["age", "glucose", "bmi", "insulin", "blood_pressure", "pregnancies"],
    "full": list(COLUMNS),
}
MODEL_NAMES = ("mlp", "naive_bayes")
SPLIT_SEED = 42
TEST_SIZE = 0.2
MAX_ROWS = 200_000

_train_lock = threading.Lock()


# ---------------------------------------------------------------- data
class DatasetError(ValueError):
    pass


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check an uploaded/loaded dataset. Needs Outcome + the 3 required columns;
    the other feature columns are optional (missing -> treated as not measured)."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    lower = {c.lower(): c for c in df.columns}
    for api, col in {**COLUMNS, "outcome": TARGET}.items():
        for alias in (col.lower(), api):
            if alias in lower and col not in df.columns:
                df = df.rename(columns={lower[alias]: col})
    missing = [COLUMNS[f] for f in REQUIRED if COLUMNS[f] not in df.columns] + ([TARGET] if TARGET not in df.columns else [])
    if missing:
        raise DatasetError(f"missing required columns: {', '.join(missing)}")
    if len(df) == 0:
        raise DatasetError("no rows")
    if len(df) > MAX_ROWS:
        raise DatasetError(f"too many rows ({len(df)} > {MAX_ROWS})")
    keep = [c for c in [*COLUMNS.values(), TARGET] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    bad = df[TARGET].isna() | ~df[TARGET].isin([0, 1])
    if bad.mean() > 0.05:
        raise DatasetError("Outcome must be 0 or 1")
    df = df[~bad]
    for col in COLUMNS.values():
        if col not in df.columns:
            df[col] = 0.0  # not measured
    df = df.fillna(0.0)
    if (df[list(COLUMNS.values())] < 0).any().any():
        raise DatasetError("negative values are not allowed")
    return df[[*COLUMNS.values(), TARGET]].astype(float).astype({TARGET: int})


def dataset_path(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise DatasetError(f"unknown dataset {dataset!r}; use one of {sorted(DATASETS)}")
    return DATASETS[dataset]["path"]


def load_dataset(dataset: str = "pima") -> pd.DataFrame:
    return validate_frame(pd.read_csv(dataset_path(dataset)))


def merge_upload(new_rows: pd.DataFrame, dataset: str = "pima") -> dict:
    """Append validated rows to the dataset (deduplicated). Keeps a copy of the upload."""
    path = dataset_path(dataset)
    base = load_dataset(dataset)
    before = len(base)
    merged = pd.concat([base, new_rows], ignore_index=True).drop_duplicates(ignore_index=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    new_rows.to_csv(UPLOADS_DIR / f"{dataset}-upload-{stamp}.csv", index=False)
    shutil.copy(path, UPLOADS_DIR / f"{dataset}-before-{stamp}.csv")  # for rollback
    merged.to_csv(path, index=False)
    return {"rows_before": before, "rows_uploaded": len(new_rows), "rows_after": len(merged), "duplicates_skipped": before + len(new_rows) - len(merged)}


# ---------------------------------------------------------------- preprocessing
def fit_preprocessing(train: pd.DataFrame, features: list[str]) -> dict:
    """Medians and IQR bounds computed on the TRAIN split only."""
    cols = [COLUMNS[f] for f in features]
    medians, bounds = {}, {}
    for col in cols:
        s = train[col].replace(0, np.nan) if col in ZERO_IS_MISSING else train[col]
        medians[col] = float(s.median()) if s.notna().any() else 0.0
        s = s.fillna(medians[col])
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        bounds[col] = [float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))]
    interactions = []
    if "BMI" in cols and "Age" in cols:
        interactions.append(["BMI", "Age"])
    if "Glucose" in cols and "Insulin" in cols:
        interactions.append(["Glucose", "Insulin"])
    return {"columns": cols, "medians": medians, "bounds": bounds, "interactions": interactions}


def transform(df: pd.DataFrame, prep: dict) -> np.ndarray:
    X = df[prep["columns"]].astype(float).copy()
    for col in prep["columns"]:
        if col in ZERO_IS_MISSING:
            X.loc[X[col] == 0, col] = prep["medians"][col]
        lo, hi = prep["bounds"][col]
        X[col] = X[col].clip(lo, hi)
    for a, b in prep["interactions"]:
        X[f"{a}_x_{b}"] = X[a] * X[b]
    return X.to_numpy()


def make_pipeline(name: str) -> Pipeline:
    clf = GaussianNB() if name == "naive_bayes" else MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def metrics(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if len(set(y_true)) > 1 else None,
    }


def cv_f1(train: pd.DataFrame, features: list[str], model: str, folds: int = 5) -> dict:
    """Stratified k-fold F1 on the training split. Preprocessing is re-fitted
    inside each fold, so no information leaks from the validation fold."""
    from sklearn.model_selection import StratifiedKFold

    scores = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SPLIT_SEED)
    for tr_idx, va_idx in skf.split(train, train[TARGET]):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        prep = fit_preprocessing(tr, features)
        pipe = make_pipeline(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(transform(tr, prep), tr[TARGET].to_numpy())
            pred = pipe.predict(transform(va, prep))
        scores.append(f1_score(va[TARGET], pred, zero_division=0))
    return {"cv_f1_mean": round(float(np.mean(scores)), 4), "cv_f1_std": round(float(np.std(scores)), 4)}


def bootstrap_f1_diff(y, proba_a, proba_b, n: int = 1000) -> list[float]:
    """95% CI of F1(a) - F1(b) on paired bootstrap resamples of the test rows."""
    rng = np.random.default_rng(SPLIT_SEED)
    y = np.asarray(y)
    pa, pb = (np.asarray(proba_a) >= 0.5).astype(int), (np.asarray(proba_b) >= 0.5).astype(int)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[i], pa[i], zero_division=0) - f1_score(y[i], pb[i], zero_division=0))
    return [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]


def split(df: pd.DataFrame):
    return train_test_split(df, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=df[TARGET])


# ---------------------------------------------------------------- registry
def read_registry() -> dict:
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"active": {}, "versions": []}
    if not isinstance(reg.get("active"), dict):  # migrate the old single-dataset format
        reg["active"] = {"pima": reg["active"]} if reg.get("active") else {}
    for v in reg["versions"]:
        v.setdefault("dataset", "pima")
    return reg


def active_version(dataset: str) -> str | None:
    return read_registry()["active"].get(dataset)


def available_datasets() -> list[str]:
    """Datasets that have an active model version, default first."""
    register_legacy_v1()  # a fresh checkout still has the original models
    active = read_registry()["active"]
    return sorted((d for d in DATASETS if active.get(d)), key=lambda d: d != DEFAULT_DATASET)


def write_registry(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=2))


def version_dir(version: str) -> Path:
    return VERSIONS_DIR / version


def read_manifest(version: str) -> dict:
    return json.loads((version_dir(version) / "manifest.json").read_text())


def load_version(version: str) -> tuple[dict, dict]:
    """Return (manifest, {(tier, model): pipeline})."""
    manifest = read_manifest(version)
    pipes = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tier, info in manifest["tiers"].items():
            for model, file in info["files"].items():
                with open(version_dir(version) / file, "rb") as f:
                    pipes[(tier, model)] = pickle.load(f)
    return manifest, pipes


def predict_proba_version(version: str, test: pd.DataFrame, tier: str = "standard", model: str = "mlp"):
    manifest, pipes = load_version(version)
    if (tier, model) not in pipes:
        return None
    prep = manifest["tiers"][tier]["preprocessing"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipes[(tier, model)].predict_proba(transform(test, prep))[:, 1]


def evaluate_version_on(version: str, test: pd.DataFrame, tier: str = "standard", model: str = "mlp") -> dict | None:
    proba = predict_proba_version(version, test, tier, model)
    return None if proba is None else metrics(test[TARGET].to_numpy(), proba)


# ---------------------------------------------------------------- legacy v1
LEGACY_PREP = {
    "columns": ["Glucose", "Insulin", "BMI", "Age"],
    "medians": {"Glucose": 117.0, "Insulin": 125.0, "BMI": 32.3, "Age": 29.0},
    "bounds": {"Glucose": [39.0, 201.0], "Insulin": [112.875, 135.875], "BMI": [13.85, 50.25], "Age": [-1.5, 66.5]},
    "interactions": [["BMI", "Age"], ["Glucose", "Insulin"]],
}


def register_legacy_v1() -> None:
    """Wrap the original assignment pickles as version v1 (standard tier only;
    missing insulin is imputed with the training median, as in training)."""
    reg = read_registry()
    if any(v["version"] == "v1" for v in reg["versions"]):
        return
    d = version_dir("v1")
    d.mkdir(parents=True, exist_ok=True)
    for model, src in {"mlp": "mlp_pipeline_v1.pkl", "naive_bayes": "nb_pipeline_v1.pkl"}.items():
        shutil.copy(LEGACY_MODELS_DIR / src, d / f"standard_{model}.pkl")
    manifest = {
        "version": "v1",
        "dataset": "pima",
        "created": "2025-11-27 (original assignment)",
        "dataset_rows": 768,
        "notes": "Original 4-feature models. Missing insulin is imputed with the training median.",
        "impute_missing": ["insulin"],
        "tiers": {
            "standard": {
                "features": ["age", "glucose", "bmi", "insulin"],
                "preprocessing": LEGACY_PREP,
                "files": {"mlp": "standard_mlp.pkl", "naive_bayes": "standard_naive_bayes.pkl"},
                "test_metrics": {},
            }
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    reg["versions"].append({"version": "v1", "dataset": "pima", "created": manifest["created"], "rows": 768, "tiers": ["standard"]})
    reg["active"].setdefault("pima", "v1")
    write_registry(reg)


# ---------------------------------------------------------------- training
def train_version(dataset: str = "pima", df: pd.DataFrame | None = None, force: bool = False, note: str = "") -> dict:
    """Train all tiers on one dataset, save as a new version, promote if not worse."""
    with _train_lock:
        register_legacy_v1()
        df = load_dataset(dataset) if df is None else validate_frame(df)
        train, test = split(df)
        reg = read_registry()
        n = max(int(v["version"][1:]) for v in reg["versions"]) + 1
        version = f"v{n}"
        d = version_dir(version)
        d.mkdir(parents=True, exist_ok=True)

        tiers_out = {}
        for tier, features in TIERS.items():
            cols = [COLUMNS[f] for f in features]
            # Only train a tier if its columns were actually measured: for "0 = not
            # measured" columns most rows must be non-zero; other columns (e.g.
            # Pregnancies, where 0 is a real value) just must not be all zero.
            zero_missing = [c for c in cols if c in ZERO_IS_MISSING]
            if zero_missing and ((train[zero_missing] != 0).mean() < 0.5).any():
                continue
            if any((train[c] == 0).all() for c in cols if c not in ZERO_IS_MISSING):
                continue
            prep = fit_preprocessing(train, features)
            Xtr, Xte = transform(train, prep), transform(test, prep)
            info = {"features": features, "preprocessing": prep, "files": {}, "test_metrics": {}}
            for model in MODEL_NAMES:
                pipe = make_pipeline(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xtr, train[TARGET].to_numpy())
                    proba = pipe.predict_proba(Xte)[:, 1]
                info["test_metrics"][model] = {**metrics(test[TARGET].to_numpy(), proba), **cv_f1(train, features, model)}
                info.setdefault("test_proba", {})[model] = proba.round(5).tolist()
                file = f"{tier}_{model}.pkl"
                with open(d / file, "wb") as f:
                    pickle.dump(pipe, f)
                info["files"][model] = file
            tiers_out[tier] = info

        manifest = {
            "version": version,
            "dataset": dataset,
            "dataset_label": DATASETS[dataset]["label"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_rows": len(df),
            "train_rows": len(train),
            "test_rows": len(test),
            "split": {"test_size": TEST_SIZE, "seed": SPLIT_SEED, "stratified": True},
            "required": REQUIRED,
            "notes": note,
            "tiers": tiers_out,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Promotion: compare with the active version on the SAME test rows.
        active = reg["active"].get(dataset)
        cand = tiers_out.get("standard", {}).get("test_metrics", {}).get("mlp")
        current = evaluate_version_on(active, test) if active else None
        ci = None
        if current and cand:
            active_proba = predict_proba_version(active, test)
            ci = bootstrap_f1_diff(test[TARGET], tiers_out["standard"]["test_proba"]["mlp"], active_proba)
        promoted = bool(force or current is None or (cand and cand["f1"] >= current["f1"]))
        decision = {
            "candidate_standard_mlp_f1": cand["f1"] if cand else None,
            "active_version": active,
            "active_standard_mlp_f1_same_test_rows": current["f1"] if current else None,
            "f1_difference_95ci": ci,
            "difference_is_noise": bool(ci and ci[0] < 0 < ci[1]),
            "promoted": promoted,
            "rule": "promote if candidate F1 >= active F1 on the same test rows" + (" (forced)" if force else ""),
        }
        manifest["promotion"] = decision
        for info in tiers_out.values():
            info.pop("test_proba", None)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))

        reg["versions"].append({"version": version, "dataset": dataset, "created": manifest["created"], "rows": len(df), "tiers": list(tiers_out)})
        if promoted:
            reg["active"][dataset] = version
        write_registry(reg)
        return {"version": version, "dataset": dataset, "promoted": promoted, "decision": decision, "tiers": {t: i["test_metrics"] for t, i in tiers_out.items()}}


def activate(version: str) -> str:
    """Make a version active for its dataset (rollback). Returns the dataset."""
    reg = read_registry()
    entry = next((v for v in reg["versions"] if v["version"] == version), None)
    if entry is None:
        raise ValueError(f"unknown version {version}")
    reg["active"][entry["dataset"]] = version
    write_registry(reg)
    return entry["dataset"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pima", choices=sorted(DATASETS))
    ap.add_argument("--force", action="store_true", help="activate even if F1 drops")
    args = ap.parse_args()
    out = train_version(dataset=args.dataset, force=args.force, note="trained from CLI")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
