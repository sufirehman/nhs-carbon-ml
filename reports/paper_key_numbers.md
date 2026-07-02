# Paper Key Numbers — NHS Carbon ML

All figures verified from final analysis outputs (202502 data vintage).
Do not edit manually — regenerate via `src/analysis/final_paper_outputs.py`.

---

## Dataset

| Item | Value |
|---|---|
| Total dispensing rows | 313,375 |
| Unique VMPs (products) | 8,473 |
| Unique VMP product names | 8,473 |
| Unique prescribing sites (ODS codes) | 194 |
| Date range | 202502 to 202502 |
| Total indicative spend | £1.813bn |
| Total spend (exact) | £1,813,459,454 |
| Estimated carbon — production basis (x 0.17) | 308,288 tCO2e (308.29 ktCO2e) |
| Estimated carbon — consumption basis (x 0.2402) | 435,593 tCO2e (435.59 ktCO2e) |
| Rows with positive target (used for modelling) | 298,585 |
| Rows excluded (null or negative spend) | 14,790 |

---

## Model Performance — Random Forest (tuned)

| Metric | Value |
|---|---|
| Algorithm | Random Forest (Optuna-tuned, 30 trials) |
| Best hyperparameters | n_estimators=391, max_depth=23, min_samples_leaf=2, max_features=0.5 |
| Train / test split | 80 / 20 stratified by BNF chapter, random_state=42 |
| Train rows | 238,868 |
| Test rows | 59,717 |
| R² — log1p space | **0.7634** |
| R² — original kgCO2e space | **0.2504** |
| MAE — original space | 1,028 kgCO2e |
| RMSE — original space | 10,032 kgCO2e |
| Within ±20% of actual | 12.5% |
| Target transformation | log1p applied (mean/median ratio = 49x; skewness = 270.56) |
| Features | 15 (9 dosage-form dummies, 3 binary flags, log_price_per_unit, bnf_chapter_num, supplier_crp_matched) |

---

## SHAP Feature Importance — Top 5

| Rank | Feature | Mean |SHAP| | One-line interpretation |
|---|---|---|---|
| 1 | Log price per unit | 1.23371 | Higher unit price predicts higher carbon (complex/specialist manufacturing) |
| 2 | IV injection form | 0.34721 | IV products have elevated carbon: sterile manufacture, cold-chain, packaging |
| 3 | BNF chapter | 0.25785 | Therapeutic chapter captures carbon variation beyond price and form |
| 4 | Tablet / capsule form | 0.18236 | Solid oral forms are the lowest-intensity reference dosage category |
| 5 | Topical form | 0.08960 | Topical creams, ointments, and gels have moderate-to-low carbon intensity; most are ambien |

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
