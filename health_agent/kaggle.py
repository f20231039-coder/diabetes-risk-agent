"""Build a training CSV from the Kaggle "Diabetes prediction dataset"
(kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset, 100,000 rows).

Data-quality notes (checked before use):
- Origin is undocumented and it looks synthetic: glucose and HbA1c each take
  only 18 distinct values.
- The label follows a hard rule: every row with HbA1c > 6.6 or glucose > 200 is
  diabetic. HbA1c is therefore NOT used as a feature (it would leak the label);
  glucose is used because the app needs it, so expect optimistic scores.
- BMI 27.32 is a placeholder for missing BMI (25,495 rows) -> those rows are
  dropped, since BMI is a required input.
- Includes children -> only adults (20+) are kept, like NHANES.
- 3,854 exact duplicates -> dropped.
- No insulin, blood pressure value or pregnancies -> only the "minimum" tier
  (age, glucose, BMI) can be trained. The glucose test type is not stated.

Columns are written in the Pima layout so the same training code works.

CLI:  python -m health_agent.kaggle [path/to/diabetes_prediction_dataset.csv.zip]
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

from .training import DATA_DIR

RAW = DATA_DIR / "kaggle" / "raw" / "diabetes_prediction_dataset.csv.zip"
OUT = DATA_DIR / "kaggle.csv"
EXPECTED = {"gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"}
BMI_PLACEHOLDER = 27.32


def read_raw(path: Path = RAW) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(io.BytesIO(z.read(name)))
    else:
        df = pd.read_csv(path)
    missing = EXPECTED - set(df.columns)
    if missing:
        raise ValueError(f"not the expected Kaggle file; missing columns: {sorted(missing)}")
    return df


def build(path: Path = RAW) -> pd.DataFrame:
    raw = read_raw(path)
    n0 = len(raw)
    df = raw.drop_duplicates()
    n_dupes = n0 - len(df)
    df = df[df["age"] >= 20]
    n_adults = len(df)
    df = df[df["bmi"] != BMI_PLACEHOLDER]
    out = pd.DataFrame(
        {
            "Pregnancies": 0.0,  # not available (0 = not measured; column is all zero so tiers using it are skipped)
            "Glucose": df["blood_glucose_level"].astype(float),
            "BloodPressure": 0.0,
            "SkinThickness": 0.0,
            "Insulin": 0.0,
            "BMI": df["bmi"].astype(float),
            "DiabetesPedigreeFunction": 0.0,
            "Age": df["age"].astype(float),
            "Outcome": df["diabetes"].astype(int),
        }
    ).drop_duplicates()
    out.to_csv(OUT, index=False)
    print(
        f"Kaggle: {n0} rows -> {n_dupes} exact duplicates dropped -> {n_adults} adults (20+) -> "
        f"{len(out)} after dropping BMI placeholder rows and new duplicates; diabetes {out['Outcome'].mean():.1%}. "
        f"Saved {OUT.relative_to(DATA_DIR.parent)} (HbA1c deliberately not used)."
    )
    return out


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else RAW)
