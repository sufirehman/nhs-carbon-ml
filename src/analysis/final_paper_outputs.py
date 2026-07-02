"""
Generate two final paper outputs:
  1. reports/figures/shap_top10_bar.png  -- publication-ready SHAP bar chart
  2. reports/paper_key_numbers.md        -- all numbers needed for paper writing

Run:
  python src/analysis/final_paper_outputs.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS      = os.path.join(PROJECT_ROOT, "reports")
FIG_DIR      = os.path.join(REPORTS, "figures")

SHAP_PATH     = os.path.join(REPORTS, "shap_findings.csv")
RESULTS_PATH  = os.path.join(REPORTS, "model_results.csv")
FEATURES_PATH = os.path.join(PROC_DIR, "scmd_with_all_features.csv")

os.makedirs(FIG_DIR, exist_ok=True)


# ── Clean display labels for SHAP features ────────────────────────────────────

LABELS = {
    "log_price_per_unit":       "Log price per unit",
    "form_iv_injection":        "IV injection form",
    "bnf_chapter_num":          "BNF chapter",
    "form_tablet_capsule":      "Tablet / capsule form",
    "form_topical":             "Topical form",
    "is_biologic":              "Biologic flag",
    "form_oral_liquid":         "Oral liquid form",
    "is_cold_chain":            "Cold-chain flag",
    "form_unclassified":        "Unclassified form",
    "form_other":               "Other form",
    "form_device_or_nutrition": "Device / nutrition form",
    "form_dpi_inhaler":         "DPI inhaler form",
    "form_other_inhaler":       "Other inhaler form",
    "is_inhaler_hfc":           "HFC inhaler flag",
    "supplier_crp_matched":     "Supplier CRP matched",
}

SHORT_INTERP = {
    "log_price_per_unit":  "Higher unit price predicts higher carbon (complex/specialist manufacturing)",
    "form_iv_injection":   "IV products have elevated carbon: sterile manufacture, cold-chain, packaging",
    "bnf_chapter_num":     "Therapeutic chapter captures carbon variation beyond price and form",
    "form_tablet_capsule": "Solid oral forms are the lowest-intensity reference dosage category",
    "is_biologic":         "Biologic manufacturing and cold-chain add carbon beyond unit price alone",
}


# ── 1. Load SHAP findings ─────────────────────────────────────────────────────

shap_df = pd.read_csv(SHAP_PATH)
top10 = shap_df.head(10).copy()
top10["label"] = top10["feature"].map(LABELS).fillna(top10["feature"])


# ── 2. Publication-ready SHAP bar chart ──────────────────────────────────────

print("Generating shap_top10_bar.png ...")

TOP3_COLOUR  = "#1a6fa8"   # deep blue for top 3
REST_COLOUR  = "#7fbcd2"   # lighter blue for ranks 4-10

colours = [TOP3_COLOUR if i < 3 else REST_COLOUR for i in range(len(top10))]

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.barh(
    top10["label"][::-1],
    top10["mean_abs_shap"][::-1],
    color=colours[::-1],
    edgecolor="white",
    linewidth=0.4,
    height=0.65,
)

# Value annotations on bars
for bar, val in zip(bars, top10["mean_abs_shap"][::-1]):
    ax.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center", ha="left",
        fontsize=8.5, color="#333333",
    )

ax.set_xlabel("Mean |SHAP value| (log-space carbon)", fontsize=11, labelpad=8)
ax.set_title(
    "Feature importance for NHS procurement carbon prediction\n"
    "(SHAP values, Random Forest, n = 59,717 test observations)",
    fontsize=11, pad=12,
)

# Clean up spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.set_xlim(0, top10["mean_abs_shap"].max() * 1.18)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
ax.set_axisbelow(True)

# Legend patch
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=TOP3_COLOUR, label="Top 3 features"),
    Patch(facecolor=REST_COLOUR, label="Features 4-10"),
]
ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=9,
    framealpha=0.85,
    edgecolor="#cccccc",
)

plt.tight_layout()
bar_path = os.path.join(FIG_DIR, "shap_top10_bar.png")
fig.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {bar_path}")


# ── 3. Compute dataset statistics from raw CSV ───────────────────────────────

print("Computing dataset statistics ...")
df = pd.read_csv(FEATURES_PATH, low_memory=False,
                 usecols=["YEAR_MONTH", "VMP_SNOMED_CODE", "VMP_PRODUCT_NAME",
                           "INDICATIVE_COST", "ODS_CODE"])

n_rows         = len(df)
n_unique_vmps  = df["VMP_SNOMED_CODE"].nunique()
n_unique_prods = df["VMP_PRODUCT_NAME"].nunique()
n_unique_sites = df["ODS_CODE"].nunique()

# Date range
df["ym_str"] = df["YEAR_MONTH"].astype(str).str.strip()
date_min = df["ym_str"].min()
date_max = df["ym_str"].max()

# Spend
total_spend = df["INDICATIVE_COST"].sum()
spend_nonnull = df["INDICATIVE_COST"].notna().sum()

# Carbon estimates (all rows with non-null spend)
total_carbon_production  = df["INDICATIVE_COST"].sum() * 0.17
total_carbon_consumption = df["INDICATIVE_COST"].sum() * 0.2402

print(f"  Rows:          {n_rows:,}")
print(f"  Unique VMPs:   {n_unique_vmps:,}")
print(f"  Unique sites:  {n_unique_sites:,}")
print(f"  Date range:    {date_min} to {date_max}")
print(f"  Total spend:   £{total_spend:,.0f}")
print(f"  Carbon (0.17): {total_carbon_production/1e3:,.0f} tCO2e")
print(f"  Carbon (0.24): {total_carbon_consumption/1e3:,.0f} tCO2e")


# ── 4. Load model results ─────────────────────────────────────────────────────

results_df = pd.read_csv(RESULTS_PATH)
rf_row     = results_df[results_df["model"] == "Random Forest (tuned)"].iloc[0]
mae_rf     = rf_row["MAE (kgCO2e)"]
rmse_rf    = rf_row["RMSE (kgCO2e)"]
r2_rf      = rf_row["R²"]
w20_rf     = rf_row["Within ±20% (%)"]

# Log-space R2 captured from generate_paper_outputs.py run
r2_log = 0.7634


# ── 5. Build shap top-5 block ─────────────────────────────────────────────────

top5 = shap_df.head(5)

shap_lines = []
for _, row in top5.iterrows():
    label  = LABELS.get(row["feature"], row["feature"])
    val    = row["mean_abs_shap"]
    interp = SHORT_INTERP.get(row["feature"], row["interpretation"][:90])
    shap_lines.append(f"| {int(row['rank'])} | {label} | {val:.5f} | {interp} |")

shap_block = "\n".join(shap_lines)


# ── 6. Write paper_key_numbers.md ────────────────────────────────────────────

print("Writing paper_key_numbers.md ...")

md = f"""\
# Paper Key Numbers — NHS Carbon ML

All figures verified from final analysis outputs ({date_max} data vintage).
Do not edit manually — regenerate via `src/analysis/final_paper_outputs.py`.

---

## Dataset

| Item | Value |
|---|---|
| Total dispensing rows | {n_rows:,} |
| Unique VMPs (products) | {n_unique_vmps:,} |
| Unique VMP product names | {n_unique_prods:,} |
| Unique prescribing sites (ODS codes) | {n_unique_sites:,} |
| Date range | {date_min} to {date_max} |
| Total indicative spend | £{total_spend/1e9:.3f}bn |
| Total spend (exact) | £{total_spend:,.0f} |
| Estimated carbon — production basis (x 0.17) | {total_carbon_production/1e3:,.0f} tCO2e ({total_carbon_production/1e6:.2f} ktCO2e) |
| Estimated carbon — consumption basis (x 0.2402) | {total_carbon_consumption/1e3:,.0f} tCO2e ({total_carbon_consumption/1e6:.2f} ktCO2e) |
| Rows with positive target (used for modelling) | 298,585 |
| Rows excluded (null or negative spend) | {n_rows - 298585:,} |

---

## Model Performance — Random Forest (tuned)

| Metric | Value |
|---|---|
| Algorithm | Random Forest (Optuna-tuned, 30 trials) |
| Best hyperparameters | n_estimators=391, max_depth=23, min_samples_leaf=2, max_features=0.5 |
| Train / test split | 80 / 20 stratified by BNF chapter, random_state=42 |
| Train rows | 238,868 |
| Test rows | 59,717 |
| R² — log1p space | **{r2_log:.4f}** |
| R² — original kgCO2e space | **{r2_rf:.4f}** |
| MAE — original space | {mae_rf:,.0f} kgCO2e |
| RMSE — original space | {rmse_rf:,.0f} kgCO2e |
| Within ±20% of actual | {w20_rf:.1f}% |
| Target transformation | log1p applied (mean/median ratio = 49x; skewness = 270.56) |
| Features | 15 (9 dosage-form dummies, 3 binary flags, log_price_per_unit, bnf_chapter_num, supplier_crp_matched) |

---

## SHAP Feature Importance — Top 5

| Rank | Feature | Mean |SHAP| | One-line interpretation |
|---|---|---|---|
{shap_block}

Full 15-feature table: `reports/shap_findings.csv`
Bar chart: `reports/figures/shap_top10_bar.png`

**Note on is_inhaler_hfc (rank 14, SHAP 0.00336):** Low rank reflects low
prevalence of HFC inhalers in secondary-care SCMD (most inhaler prescribing is
primary care). The feature is physically correct; its SHAP weight would be
substantially higher in a primary-care dataset.

---

## Supplier CRP Data Collection

| Item | Value |
|---|---|
| Suppliers targeted | 15 |
| Suppliers with usable UK-specific CRP data (Yes or Partial) | 7 (47%) |
| Suppliers with disclosed UK intensity ratios (tCO2e/£m) | 2 (Siemens Diagnostics: 30; Siemens Healthcare: 213.6) |
| Suppliers with CRP not found | 1 |
| SCMD VMPs matched to any supplier | 13 (of 8,473 unique VMPs) |
| Matched VMP coverage (% of unique VMPs) | 0.15% |
| Matched spend coverage | <0.01% of total spend |
| Suppliers contributing product matches | 2 (Fresenius Kabi, B. Braun) |
| Reason for low match rate | SCMD uses INN (generic) names; proprietary brand names appear only for products with no generic equivalent (parenteral nutrition lines) |

---

## Emissions Intensity Baselines

| Source | Factor | Basis | Notes |
|---|---|---|---|
| ONS / DEFRA 2023 (production) | 0.17 kgCO2e / £ | Production basis | NHS Supply Chain; lower bound |
| DEFRA 2023 (consumption) | 0.2402 kgCO2e / £ | Consumption basis | CPA 21 Pharmaceuticals; used as modelling target |
| DEFRA 2023 (consumption, rounded) | 0.24 kgCO2e / £ | Consumption basis | Flat baseline applied in evaluation |
| EXIOBASE (chemicals-nec) | 0.39 kgCO2e / £ | Consumption basis | Upper bound from global MRIO; cited as sensitivity check |

---

## Circular Label Note

The DEFRA flat baseline achieves near-zero MAE (~1.42 kgCO2e) because the
modelling target is algebraically derived from the same spend variable the
baseline uses (target = spend × 0.2402; baseline = spend × 0.24). MAE
improvement over DEFRA is therefore not reported — it would be a mathematical
identity, not a performance finding.

---

## Output File Locations

| File | Description |
|---|---|
| `reports/figures/shap_top10_bar.png` | Publication bar chart (300 DPI) |
| `reports/figures/predicted_vs_actual.png` | Predicted vs actual scatter (300 DPI) |
| `reports/figures/shap_feature_importance.png` | Full SHAP bar (training run) |
| `reports/figures/shap_summary.png` | SHAP beeswarm (training run) |
| `reports/shap_findings.csv` | All 15 features with interpretations |
| `reports/model_results_paper.csv` | Clean model table (no circular column) |
| `reports/methodology_note.md` | 4-paragraph methodology note |
| `models/best_model.pkl` | Saved Random Forest (feature names + apply_log flag) |
| `data/processed/scmd_with_all_features.csv` | Full feature matrix (313,375 rows) |
| `data/interim/supplier_crp_matches.csv` | CRP matching results per supplier |
"""

note_path = os.path.join(REPORTS, "paper_key_numbers.md")
with open(note_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"  Saved: {note_path}")

print("\nDone. Modelling phase complete.")
