"""Build a training dataset from NHANES (US CDC survey, public domain).

Five cycles 2009-2018. One row per adult (20+) in the fasting lab subsample.
Columns match the Pima layout so the same training code works:

  Glucose        fasting plasma glucose, mg/dL      (LBXGLU)   <- note: FASTING, unlike Pima's 2-hour test
  Insulin        fasting insulin, uU/mL             (LBXIN)
  BMI            body mass index                    (BMXBMI)
  Age            years (top-coded at 80)            (RIDAGEYR)
  BloodPressure  mean diastolic of valid readings   (BPXDI1-4)
  Pregnancies    times pregnant (women), 0 for men  (RHQ160)
  SkinThickness, DiabetesPedigreeFunction: not available -> 0 (not measured)

Outcome = 1 if a doctor ever said the person has diabetes (DIQ010 == 1) or
HbA1c >= 6.5%; 0 if told no/borderline and HbA1c < 6.5% (or missing).
Fasting glucose is NOT part of the label, so the model cannot just read
its own input back. Pregnant women are excluded.

CLI:  python -m health_agent.nhanes            # download (cached) + build data/nhanes.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .training import DATA_DIR

RAW_DIR = DATA_DIR / "nhanes" / "raw"
OUT = DATA_DIR / "nhanes.csv"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
CYCLES = {"2009": "F", "2011": "G", "2013": "H", "2015": "I", "2017": "J"}
FILES = ["DEMO", "BMX", "BPX", "GLU", "GHB", "DIQ", "RHQ"]
INS_SEPARATE = {"H", "I", "J"}  # from 2013 insulin moved out of GLU into INS


def download(force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for year, letter in CYCLES.items():
            names = FILES + (["INS"] if letter in INS_SEPARATE else [])
            for f in names:
                name = f"{f}_{letter}"
                path = RAW_DIR / f"{name}.xpt"
                if force or not path.exists() or path.stat().st_size == 0:
                    resp = client.get(BASE.format(year=year, name=name))
                    resp.raise_for_status()
                    if resp.content[:6] != b"HEADER":  # SAS transport files start with "HEADER RECORD"
                        raise RuntimeError(f"{name}: not a SAS transport file")
                    path.write_bytes(resp.content)
                    print(f"  downloaded {name}.xpt ({len(resp.content) / 1e6:.1f} MB)", flush=True)
                paths.append(path)
    return paths


def _read(name: str, cols: list[str]) -> pd.DataFrame:
    df = pd.read_sas(RAW_DIR / f"{name}.xpt", format="xport")
    return df[[c for c in ["SEQN", *cols] if c in df.columns]]


def build_cycle(letter: str) -> pd.DataFrame:
    demo = _read(f"DEMO_{letter}", ["RIDAGEYR", "RIAGENDR", "RIDEXPRG"])
    bmx = _read(f"BMX_{letter}", ["BMXBMI"])
    bpx = _read(f"BPX_{letter}", ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
    glu = _read(f"GLU_{letter}", ["LBXGLU", "LBXIN"])
    ins = _read(f"INS_{letter}", ["LBXIN"]) if letter in INS_SEPARATE else None
    ghb = _read(f"GHB_{letter}", ["LBXGH"])
    diq = _read(f"DIQ_{letter}", ["DIQ010"])
    rhq = _read(f"RHQ_{letter}", ["RHQ160"])

    df = demo.merge(glu, on="SEQN", how="inner")  # fasting subsample only
    if ins is not None:
        df = df.drop(columns=[c for c in ["LBXIN"] if c in df.columns]).merge(ins, on="SEQN", how="left")
    for part in (bmx, bpx, ghb, diq, rhq):
        df = df.merge(part, on="SEQN", how="left")

    di = df[[c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in df.columns]].where(lambda x: x > 0)
    preg = df["RHQ160"].where(df["RHQ160"] < 77)  # 77/99 = refused / don't know
    preg = np.where(df["RIAGENDR"] == 1, 0, preg)  # men

    out = pd.DataFrame(
        {
            "Pregnancies": preg,
            "Glucose": df["LBXGLU"],
            "BloodPressure": di.mean(axis=1),
            "SkinThickness": 0.0,
            "Insulin": df.get("LBXIN"),
            "BMI": df["BMXBMI"],
            "DiabetesPedigreeFunction": 0.0,
            "Age": df["RIDAGEYR"],
            "a1c": df["LBXGH"],
            "diq": df["DIQ010"],
            "pregnant": df.get("RIDEXPRG"),
            "cycle": letter,
        }
    )
    return out


def build() -> pd.DataFrame:
    frames = [build_cycle(letter) for letter in CYCLES.values()]
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df[(df["Age"] >= 20) & df["Glucose"].notna() & df["BMI"].notna()]
    df = df[df["pregnant"] != 1]
    diagnosed = df["diq"] == 1
    high_a1c = df["a1c"] >= 6.5
    known_no = df["diq"].isin([2, 3])
    df = df[diagnosed | high_a1c | (known_no & df["a1c"].notna())].copy()  # label must be determinable
    df["Outcome"] = (diagnosed | high_a1c).astype(int)
    # Pregnancies unknown (refused) for some women -> drop those rows to keep 0 meaning "zero".
    df = df[df["Pregnancies"].notna()]
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    final = df[cols].fillna(0.0)  # missing insulin / BP -> 0 = not measured (same convention as Pima)
    final.to_csv(OUT, index=False)
    print(
        f"NHANES: {n0} fasting-subsample people -> {len(final)} adults kept; "
        f"diabetes {final['Outcome'].mean():.1%}; insulin measured {(final['Insulin'] > 0).mean():.1%}; "
        f"BP measured {(final['BloodPressure'] > 0).mean():.1%}. Saved {OUT.relative_to(DATA_DIR.parent)}"
    )
    return final


if __name__ == "__main__":
    download(force="--force" in sys.argv)
    build()
