"""
Generate lightweight demo assets for Streamlit Cloud deployment.

models/best_model.pkl (259 MB) and the full processed/interim CSVs are too
large / not committed to the repo. This script uses them locally (one-off)
to produce two small, git-committable files under data/demo/:

  1. vmp_predictions.csv   -- pre-computed predictions for all unique VMPs,
                               using the full trained model. Powers the
                               Carbon Explorer display tables/charts without
                               needing the 259 MB model in production.
  2. scmd_demo_sample.csv  -- a ~5,000-row stratified sample (by BNF chapter)
                               of the full dispensing data. Powers the
                               Overview page and trains a small demo
                               RandomForestRegressor for the What-If tool.

Run from the project root:
  python src/data/generate_demo_assets.py
"""

import os
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "scmd_with_all_features.csv")
BASELINE_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "scmd_with_defra_baseline.csv")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
DEMO_DIR      = os.path.join(PROJECT_ROOT, "data", "demo")
PRED_OUT      = os.path.join(DEMO_DIR, "vmp_predictions.csv")
SAMPLE_OUT    = os.path.join(DEMO_DIR, "scmd_demo_sample.csv")

RANDOM_STATE   = 42
SAMPLE_TARGET  = 5000

SAMPLE_COLUMNS = [
    "YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE", "VMP_PRODUCT_NAME",
    "dosage_form", "is_biologic", "is_inhaler_hfc", "is_cold_chain",
    "supplier_crp_matched", "price_per_unit_median", "bnf_chapter_num",
    "bnf_chapter_name", "INDICATIVE_COST", "estimated_kgco2e_consumption",
]


def build_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Mirrors src/models/train_and_evaluate.py::build_features exactly."""
    dummies = pd.get_dummies(df["dosage_form"], prefix="form", dtype=int)
    parts   = [dummies]

    for col in ["is_biologic", "is_inhaler_hfc", "is_cold_chain", "supplier_crp_matched"]:
        parts.append(df[[col]].astype(int))

    ppu = df["price_per_unit_median"].copy()
    ppu = ppu.fillna(ppu.median())
    parts.append(np.log1p(ppu).rename("log_price_per_unit"))

    parts.append(df[["bnf_chapter_num"]].astype(int))

    X = pd.concat(parts, axis=1)
    return X.reindex(columns=feature_names, fill_value=0)


def main():
    os.makedirs(DEMO_DIR, exist_ok=True)

    print("Loading full processed data...")
    af = pd.read_csv(FEATURES_PATH, low_memory=False)
    db = pd.read_csv(
        BASELINE_PATH, low_memory=False,
        usecols=["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE",
                 "bnf_chapter_num", "bnf_chapter_name",
                 "estimated_kgco2e_consumption"],
    )
    df = af.merge(db, on=["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE"], how="left")
    print(f"  Merged: {len(df):,} rows, {df['VMP_SNOMED_CODE'].nunique():,} unique VMPs")

    # ── 1. vmp_predictions.csv ────────────────────────────────────────────
    print("\nLoading full trained model (259 MB)...")
    with open(MODEL_PATH, "rb") as f:
        pkg = pickle.load(f)

    keep = [c for c in [
        "VMP_SNOMED_CODE", "VMP_PRODUCT_NAME", "dosage_form",
        "is_biologic", "is_inhaler_hfc", "is_cold_chain",
        "supplier_crp_matched", "price_per_unit_median", "bnf_chapter_num",
    ] if c in df.columns]

    vmps = df.drop_duplicates("VMP_SNOMED_CODE")[keep].copy().reset_index(drop=True)
    X    = build_features(vmps, pkg["feature_names"])
    pred = pkg["model"].predict(X)
    vmps["predicted_kgco2e_per_unit"] = np.expm1(pred) if pkg["apply_log"] else pred

    spend = df.groupby("VMP_SNOMED_CODE")["INDICATIVE_COST"].sum().rename("total_spend")
    vmps  = vmps.merge(spend, on="VMP_SNOMED_CODE", how="left")

    vmps.to_csv(PRED_OUT, index=False)
    size_kb = os.path.getsize(PRED_OUT) / 1024
    print(f"  Saved: {PRED_OUT}  ({len(vmps):,} rows, {size_kb:,.1f} KB)")

    # ── 2. scmd_demo_sample.csv ───────────────────────────────────────────
    print("\nBuilding stratified demo sample (~5,000 rows, by BNF chapter)...")
    from sklearn.model_selection import train_test_split
    sample, _ = train_test_split(
        df, train_size=SAMPLE_TARGET, random_state=RANDOM_STATE,
        stratify=df["bnf_chapter_num"],
    )
    sample = sample[SAMPLE_COLUMNS].reset_index(drop=True)
    sample.to_csv(SAMPLE_OUT, index=False)
    size_kb = os.path.getsize(SAMPLE_OUT) / 1024
    print(f"  Saved: {SAMPLE_OUT}  ({len(sample):,} rows, {size_kb:,.1f} KB)")
    print(f"\n  Dosage form coverage in sample:")
    print(sample["dosage_form"].value_counts())


if __name__ == "__main__":
    main()
