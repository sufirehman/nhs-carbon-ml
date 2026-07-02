"""
Final feature matrix construction, model training, and evaluation.

Inputs:
  data/processed/scmd_with_all_features.csv   -- pharma + supplier features
  data/interim/scmd_with_defra_baseline.csv   -- target + BNF chapter

Outputs:
  models/best_model.pkl
  reports/model_results.csv
  reports/figures/shap_feature_importance.png
  reports/figures/shap_summary.png

Models trained:
  1. DEFRA flat baseline  (spend × 0.24, no ML)
  2. XGBoost              (Optuna-tuned, 50 trials)
  3. LightGBM             (Optuna-tuned, 50 trials)
  4. Random Forest        (sklearn, default + tuned)

Target: estimated_kgco2e_consumption (spend × 0.2402 DEFRA consumption factor).
log1p transformation applied when target is heavily right-skewed (mean/median > 5).
"""

import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "scmd_with_all_features.csv")
BASELINE_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "scmd_with_defra_baseline.csv")
MODEL_OUT     = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
RESULTS_OUT   = os.path.join(PROJECT_ROOT, "reports", "model_results.csv")
FIG_DIR       = os.path.join(PROJECT_ROOT, "reports", "figures")
RANDOM_STATE  = 42


# -- 1. Load and merge ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("Loading data...")
    af = pd.read_csv(FEATURES_PATH, low_memory=False)
    db = pd.read_csv(BASELINE_PATH, low_memory=False,
                     usecols=["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE",
                               "bnf_chapter_num", "bnf_chapter_name",
                               "estimated_kgco2e", "estimated_kgco2e_consumption"])

    # Join on the three keys that uniquely identify a row
    df = af.merge(db, on=["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE"], how="left")
    print(f"  Merged: {len(df):,} rows, {len(df.columns)} columns")
    return df


# -- 2. Target distribution and filtering --------------------------------------

def inspect_and_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    y_raw = df["estimated_kgco2e_consumption"]

    print("\n--- Target variable: estimated_kgco2e_consumption -------------------")
    print(f"  Total rows:       {len(y_raw):>10,}")
    print(f"  Non-null:         {y_raw.notna().sum():>10,}")
    print(f"  Positive:         {(y_raw > 0).sum():>10,}")
    print(f"  Zero:             {(y_raw == 0).sum():>10,}")
    print(f"  Negative:         {(y_raw < 0).sum():>10,}  "
          "(returns/credits — excluded from training)")
    print(f"  Null:             {y_raw.isna().sum():>10,}  "
          "(missing spend — excluded from training)")
    pos = y_raw[y_raw > 0]
    print(f"\n  Distribution (positive values only, n={len(pos):,}):")
    print(f"    mean   = {pos.mean():>12,.2f} kgCO2e")
    print(f"    median = {pos.median():>12,.2f} kgCO2e")
    print(f"    std    = {pos.std():>12,.2f} kgCO2e")
    print(f"    min    = {pos.min():>12,.4f} kgCO2e")
    print(f"    max    = {pos.max():>12,.2f} kgCO2e")
    skew = pos.skew()
    print(f"    skewness = {skew:.2f}  (>1 = right-skewed, >5 = severely right-skewed)")
    mean_median_ratio = pos.mean() / pos.median()
    print(f"    mean/median ratio = {mean_median_ratio:.1f}x")

    apply_log = mean_median_ratio > 5
    print(f"\n  log1p transformation: {'APPLIED' if apply_log else 'NOT applied'}")
    if apply_log:
        print("  Reason: mean/median ratio > 5x indicates severe right skew that")
        print("  would bias tree-model splits toward extreme values. log1p compresses")
        print("  the range while preserving rank order; predictions are inverse-")
        print("  transformed (expm1) back to kgCO2e for evaluation.")

    # Filter to positive target rows only
    mask = y_raw > 0
    df_clean = df[mask].copy().reset_index(drop=True)
    print(f"\n  Rows retained for training/evaluation: {len(df_clean):,}")
    return df_clean, apply_log


# -- 3. Feature matrix ---------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    print("\n--- Building feature matrix -----------------------------------------")

    parts = []

    # dosage_form — one-hot (all categories kept; no drop_first to be explicit)
    dummies = pd.get_dummies(df["dosage_form"], prefix="form", dtype=int)
    parts.append(dummies)
    dosage_cols = dummies.columns.tolist()

    # Binary flags
    for col in ["is_biologic", "is_inhaler_hfc", "is_cold_chain", "supplier_crp_matched"]:
        parts.append(df[[col]].astype(int))

    # price_per_unit — log1p, null → median
    ppu = df["price_per_unit_median"].copy()
    ppu_median = ppu.median()
    ppu = ppu.fillna(ppu_median)
    ppu_log = np.log1p(ppu).rename("log_price_per_unit")
    parts.append(ppu_log)

    # bnf_chapter — integer label (already numeric; tree models handle ordinality)
    parts.append(df[["bnf_chapter_num"]].astype(int))

    X = pd.concat(parts, axis=1)
    feature_names = X.columns.tolist()

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features ({len(feature_names)}):")
    for fn in feature_names:
        nulls = X[fn].isna().sum()
        print(f"    {fn:<35}  null={nulls}")

    return X, feature_names


# -- 4. Evaluation helpers -----------------------------------------------------

def within_20pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of predictions within ±20% of ground truth."""
    ratio = np.abs(y_pred - y_true) / np.where(y_true == 0, 1e-9, np.abs(y_true))
    return 100.0 * (ratio <= 0.20).mean()


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray,
             defra_mae: float) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    w20  = within_20pct(y_true, y_pred)
    pct_improvement = 100.0 * (defra_mae - mae) / defra_mae if defra_mae > 0 else float("nan")
    return {
        "model":               name,
        "MAE (kgCO2e)":        round(mae, 2),
        "RMSE (kgCO2e)":       round(rmse, 2),
        "R²":                  round(r2, 4),
        "Within ±20% (%)":     round(w20, 1),
        "MAE improvement vs DEFRA (%)": round(pct_improvement, 1),
    }


# -- 5. Optuna objective builders ----------------------------------------------

def xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state":      RANDOM_STATE,
        "tree_method":       "hist",
        "verbosity":         0,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


def lgb_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state":      RANDOM_STATE,
        "verbosity":         -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


# -- 6. Main -------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    # Load
    df = load_data()
    df_clean, apply_log = inspect_and_filter(df)
    X, feature_names = build_features(df_clean)

    # Target
    y_raw = df_clean["estimated_kgco2e_consumption"].values.astype(float)
    y_log = np.log1p(y_raw) if apply_log else y_raw

    # DEFRA baseline prediction (uses INDICATIVE_COST — not in feature matrix)
    spend = df_clean["INDICATIVE_COST"].values.astype(float)
    defra_pred_raw = spend * 0.24

    # Production-basis benchmark (spend × 0.17)
    prod_pred_raw = spend * 0.17

    # -- Train / test split stratified by BNF chapter --------------------------
    strat = df_clean["bnf_chapter_num"].values
    (X_train, X_test,
     y_log_train, y_log_test,
     y_raw_train, y_raw_test,
     defra_train, defra_test,
     prod_train, prod_test) = train_test_split(
        X, y_log, y_raw, defra_pred_raw, prod_pred_raw,
        test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )

    print(f"\n--- Train/test split (stratified by bnf_chapter_num) ---------------")
    print(f"  Train: {len(X_train):,} rows   Test: {len(X_test):,} rows")

    # -- DEFRA baseline evaluation ---------------------------------------------
    print("\n--- Model 1: DEFRA flat baseline (spend × 0.24) --------------------")
    defra_mae = mean_absolute_error(y_raw_test, defra_test)
    print(f"  Test MAE: {defra_mae:,.2f} kgCO2e  (reference for % improvement)")
    results = [evaluate("DEFRA flat baseline", y_raw_test, defra_test, defra_mae)]

    # Production-basis reference (not a model — for documentation only)
    print("\n--- Reference: Production-basis (spend × 0.17) ---------------------")
    prod_mae = mean_absolute_error(y_raw_test, prod_test)
    print(f"  Test MAE: {prod_mae:,.2f} kgCO2e  (benchmark column, not a trained model)")

    # Split train further for Optuna validation
    (X_opt_tr, X_opt_val,
     y_opt_tr, y_opt_val) = train_test_split(
        X_train, y_log_train,
        test_size=0.2, random_state=RANDOM_STATE
    )

    # -- XGBoost + Optuna ------------------------------------------------------
    print("\n--- Model 2: XGBoost (Optuna, 50 trials) ---------------------------")
    study_xgb = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(
        lambda t: xgb_objective(t, X_opt_tr, y_opt_tr, X_opt_val, y_opt_val),
        n_trials=50, show_progress_bar=False
    )
    best_xgb_params = {**study_xgb.best_params,
                       "random_state": RANDOM_STATE, "tree_method": "hist", "verbosity": 0}
    xgb_model = xgb.XGBRegressor(**best_xgb_params)
    xgb_model.fit(X_train, y_log_train)
    xgb_pred_log = xgb_model.predict(X_test)
    xgb_pred_raw = np.expm1(xgb_pred_log) if apply_log else xgb_pred_log
    print(f"  Best Optuna MAE (log-space, val): {study_xgb.best_value:.5f}")
    print(f"  Best params: {study_xgb.best_params}")
    results.append(evaluate("XGBoost (Optuna)", y_raw_test, xgb_pred_raw, defra_mae))

    # -- LightGBM + Optuna -----------------------------------------------------
    print("\n--- Model 3: LightGBM (Optuna, 50 trials) --------------------------")
    study_lgb = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(
        lambda t: lgb_objective(t, X_opt_tr, y_opt_tr, X_opt_val, y_opt_val),
        n_trials=50, show_progress_bar=False
    )
    best_lgb_params = {**study_lgb.best_params, "random_state": RANDOM_STATE, "verbosity": -1}
    lgb_model = lgb.LGBMRegressor(**best_lgb_params)
    lgb_model.fit(X_train, y_log_train,
                  callbacks=[lgb.log_evaluation(period=-1)])
    lgb_pred_log = lgb_model.predict(X_test)
    lgb_pred_raw = np.expm1(lgb_pred_log) if apply_log else lgb_pred_log
    print(f"  Best Optuna MAE (log-space, val): {study_lgb.best_value:.5f}")
    print(f"  Best params: {study_lgb.best_params}")
    results.append(evaluate("LightGBM (Optuna)", y_raw_test, lgb_pred_raw, defra_mae))

    # -- Random Forest ---------------------------------------------------------
    print("\n--- Model 4: Random Forest (default + tuned) -----------------------")
    # Default first
    rf_default = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE,
                                        n_jobs=-1)
    rf_default.fit(X_train, y_log_train)
    rf_pred_log_def = rf_default.predict(X_test)
    rf_pred_raw_def = np.expm1(rf_pred_log_def) if apply_log else rf_pred_log_def
    rf_default_mae = mean_absolute_error(y_raw_test, rf_pred_raw_def)
    print(f"  Default RF MAE (original space): {rf_default_mae:,.2f} kgCO2e")

    # Tuned: increase n_estimators, tune max_features and min_samples_leaf via Optuna
    def rf_objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            "max_depth":         trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical(
                                     "max_features", ["sqrt", "log2", 0.5, 0.7]),
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        }
        rf = RandomForestRegressor(**params)
        rf.fit(X_opt_tr, y_opt_tr)
        pred = rf.predict(X_opt_val)
        return mean_absolute_error(y_opt_val, pred)

    study_rf = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(rf_objective, n_trials=30, show_progress_bar=False)
    best_rf_params = {**study_rf.best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
    rf_model = RandomForestRegressor(**best_rf_params)
    rf_model.fit(X_train, y_log_train)
    rf_pred_log = rf_model.predict(X_test)
    rf_pred_raw = np.expm1(rf_pred_log) if apply_log else rf_pred_log
    print(f"  Best Optuna MAE (log-space, val): {study_rf.best_value:.5f}")
    print(f"  Best params: {study_rf.best_params}")
    results.append(evaluate("Random Forest (tuned)", y_raw_test, rf_pred_raw, defra_mae))

    # -- Select best model (lowest MAE) ----------------------------------------
    ml_results = results[1:]  # exclude DEFRA baseline
    best_result = min(ml_results, key=lambda r: r["MAE (kgCO2e)"])
    best_name = best_result["model"]
    print(f"\n--- Best ML model: {best_name} --------------------------------------")

    model_map = {
        "XGBoost (Optuna)":      (xgb_model, xgb_pred_raw),
        "LightGBM (Optuna)":     (lgb_model, lgb_pred_raw),
        "Random Forest (tuned)": (rf_model,  rf_pred_raw),
    }
    best_model, best_pred_raw = model_map[best_name]

    # -- Save best model --------------------------------------------------------
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": best_model, "feature_names": feature_names,
                     "apply_log": apply_log, "model_name": best_name}, f)
    print(f"  Saved: {MODEL_OUT}")

    # -- SHAP analysis ----------------------------------------------------------
    print(f"\n--- SHAP analysis ({best_name}) -------------------------------------")
    # Use a sample for SHAP to keep runtime manageable
    shap_sample_n = min(5000, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    shap_idx = rng.choice(len(X_test), size=shap_sample_n, replace=False)
    X_shap = X_test.iloc[shap_idx]

    if "XGBoost" in best_name:
        explainer = shap.TreeExplainer(best_model)
    elif "LightGBM" in best_name:
        explainer = shap.TreeExplainer(best_model)
    else:  # Random Forest
        explainer = shap.TreeExplainer(best_model)

    shap_values = explainer.shap_values(X_shap)

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)

    print("\n  Top 5 features by mean |SHAP|:")
    for feat, val in mean_abs_shap.head(5).items():
        print(f"    {feat:<35}  mean |SHAP| = {val:.5f}")

    # Global feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    top_n = min(15, len(mean_abs_shap))
    top_feats = mean_abs_shap.head(top_n)
    ax.barh(top_feats.index[::-1], top_feats.values[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value| (log-space)")
    ax.set_title(f"Feature Importance — {best_name}\n(SHAP, top {top_n} features)")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    bar_path = os.path.join(FIG_DIR, "shap_feature_importance.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {bar_path}")

    # SHAP summary plot (beeswarm)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                      show=False, max_display=15, plot_size=None)
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    summary_path = os.path.join(FIG_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {summary_path}")

    # -- Results table ----------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_OUT, index=False)
    print(f"\n--- Saved: {RESULTS_OUT}")

    # -- Print results ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("MODEL RESULTS — HELD-OUT TEST SET (20%, stratified by BNF chapter)")
    print(f"n_test = {len(y_raw_test):,}   Target: estimated_kgco2e_consumption")
    print(f"log1p transformation applied: {apply_log}")
    print("=" * 78)
    col_w = [28, 14, 16, 8, 17, 28]
    headers = ["Model", "MAE (kgCO2e)", "RMSE (kgCO2e)", "R²",
               "Within ±20%", "MAE impr. vs DEFRA (%)"]
    header_row = "".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_row)
    print("-" * 78)
    for r in results:
        vals = [
            r["model"],
            f"{r['MAE (kgCO2e)']:,.0f}",
            f"{r['RMSE (kgCO2e)']:,.0f}",
            f"{r['R²']:.4f}",
            f"{r['Within ±20% (%)']:.1f}%",
            f"{r['MAE improvement vs DEFRA (%)']:+.1f}%",
        ]
        print("".join(v.ljust(w) for v, w in zip(vals, col_w)))
    print("=" * 78)
    print(f"\n  Production-basis reference (spend x 0.17):")
    print(f"    MAE = {prod_mae:,.2f} kgCO2e   (not a trained model — DEFRA-based benchmark)")
    print(f"\n  Best ML model:  {best_name}")
    print(f"  SHAP top feature: {mean_abs_shap.index[0]}  "
          f"(mean |SHAP| = {mean_abs_shap.iloc[0]:.5f})")
    print()


if __name__ == "__main__":
    main()
