# NHS-Carbon-ML: Predicting Carbon Cost of NHS Supply Chains Using ML

This repository contains the code and research data for a study of carbon intensity in NHS secondary-care medicine procurement. The pipeline takes the NHSBSA Secondary Care Medicines Data (SCMD) as input, applies pharmacological feature engineering, trains a Random Forest regressor, and uses SHAP to explain which product attributes drive variation in estimated carbon intensity. The dataset covers 313,375 dispensing records across 8,473 unique virtual medicinal products (VMPs) from 194 NHS prescribing sites, representing approximately GBP 1.81 billion of spend (February 2025). In addition to the ML pipeline, the project includes a systematic collection and analysis of Carbon Reduction Plan (CRP) disclosures from 15 NHS suppliers, which finds that SCMD's use of International Nonproprietary Names (INN) prevents product-level supplier attribution for 99.85% of medicines by VMP count.

## Research Contribution

- Full reproducible pipeline from raw NHSBSA SCMD to SHAP-based feature importance, using publicly available data sources
- A 15-feature pharmacological representation (dosage form, biologic status, HFC propellant flag, cold-chain requirement, BNF chapter, log price per unit) as predictors of product-level carbon intensity
- Systematic CRP collection from 15 NHS suppliers with structured coverage and usability analysis, reported as a reusable dataset in `data/raw/supplier_crps_manual.csv`
- Identification and documentation of a circular label problem in spend-based DEFRA carbon estimates, with a methodology for evaluating ML models honestly in this setting (see `reports/methodology_note.md`)
- SHAP evidence that unit price (mean |SHAP| = 1.234), IV injection form (0.347), and BNF therapeutic chapter (0.258) are the primary drivers of carbon variation, providing a basis for disaggregating the DEFRA flat multiplier by product category

Paper under review. This project builds on prior work in carbon-aware machine learning for NHS procurement [CAML-TC, IEEE, link to be added].

## Repository Structure

```
NHS-Carbon-ML/
    data/
        raw/
            supplier_crps_manual.csv     # manually collected CRP data from 15 NHS suppliers
            MANUAL_CRP_COLLECTION.md     # data collection protocol and source notes
            [other raw files]            # downloaded from original sources, not committed
        interim/
            bnf_to_defra_mapping.csv     # BNF chapter to DEFRA category crosswalk
            defra_mapping_SOURCES.md     # sources and decisions behind the BNF mapping
            exiobase_consumption_basis.md  # EXIOBASE intensity derivation notes
            [processed intermediates]    # regenerable, not committed
        processed/                       # final feature matrices (not committed)
    models/                              # trained model artefacts (not committed, ~259 MB)
    reports/
        figures/                         # SHAP charts and scatter plots (PNG, 300 DPI)
        paper_key_numbers.md             # all quantitative results needed for writing
        methodology_note.md              # circular label problem and evaluation framing
        model_results.csv                # model performance table
        model_results_paper.csv          # clean version without circular comparison column
        shap_findings.csv                # all 15 features with SHAP values and interpretations
    src/
        data/                            # data acquisition scripts
        features/                        # feature engineering pipeline
        models/                          # model training and evaluation
        analysis/                        # paper output generation
    requirements.txt
    README.md
```

## Data Sources

| Source | URL | Licence | How to download |
|---|---|---|---|
| NHSBSA Secondary Care Medicines Data (SCMD) | https://opendata.nhsbsa.net/dataset/secondary-care-medicines-data | Open Government Licence v3 | Download monthly CSV from the portal; rename to `data/raw/scmd_SCMD_FINAL_YYYYMM.csv` |
| DEFRA/DESNZ 2025 GHG Conversion Factors | https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting | Open Government Licence v3 | Run `python src/data/fetch_ghg_factors.py` |
| ONS UK Environmental Accounts | https://www.ons.gov.uk/economy/environmentalaccounts | Open Government Licence v3 | Download manually from ONS; place in `data/raw/` |
| EXIOBASE 3 IOT_2022_pxp | https://zenodo.org/record/5589597 | Creative Commons Attribution 4.0 | Download `IOT_2022_pxp.zip` and place in `data/raw/` |
| NHS Supplier Carbon Reduction Plans | Included: `data/raw/supplier_crps_manual.csv` | See `MANUAL_CRP_COLLECTION.md` | Already in this repository |

## Key Findings

- Log price per unit is the strongest predictor of carbon intensity (mean |SHAP| = 1.234, three times larger than the next feature), confirming that product manufacturing complexity is the main driver of carbon variation across NHS medicines, not dosage form alone.
- IV injection form (SHAP = 0.347) and BNF therapeutic chapter (SHAP = 0.258) independently predict carbon intensity beyond what price captures, supporting the case for a product-category-differentiated DEFRA factor rather than a single flat multiplier.
- 7 of 15 targeted suppliers provided usable UK-specific CRP data (47%). Only 13 VMPs (0.15% of all VMPs in SCMD) could be matched to a supplier, all from Fresenius Kabi and B. Braun parenteral nutrition lines. SCMD's INN naming convention is the structural barrier: generic drug names carry no manufacturer identity, so CRP data cannot be linked to individual dispensing records for 99.85% of medicines.
- The DEFRA flat-rate baseline achieves near-zero MAE by construction because the modelling target is derived from the same spend variable the baseline uses (target = spend x 0.2402, baseline = spend x 0.24). MAE comparison against the DEFRA baseline is not a valid performance metric. See `reports/methodology_note.md` for the full discussion.

## Quickstart

```bash
pip install -r requirements.txt
```

Then reproduce the pipeline in this order:

1. Download SCMD and DEFRA GHG data (see Data Sources above)
2. `python src/data/fetch_ghg_factors.py` to fetch and cache DEFRA conversion factors
3. `python src/features/build_defra_mapping.py` to map BNF chapters to DEFRA categories
4. `python src/features/apply_carbon_labels.py` to attach carbon estimates to SCMD records
5. `python src/features/compute_exiobase_intensity.py` to derive EXIOBASE intensity reference values
6. `python src/features/build_pharma_features.py` to build the pharmacological feature set
7. `python src/features/match_supplier_crp.py` to match CRP data to SCMD product names
8. `python src/models/train_and_evaluate.py` to train all models and run SHAP analysis
9. `python src/analysis/generate_paper_outputs.py` to generate SHAP tables and scatter plot
10. `python src/analysis/final_paper_outputs.py` to generate the publication bar chart and key numbers

All outputs land in `reports/` and `reports/figures/`.

## Methodology Notes

`reports/methodology_note.md` contains a four-paragraph explanation of:

- why the DEFRA baseline MAE comparison is circular and is not reported as a performance metric
- what SHAP-based feature importance means in this context and how to read it
- what external validation is available from the engineering and CRP literature
- the SCMD INN naming convention as a structural barrier to supplier-level carbon attribution

## Citation

```bibtex
@article{ulrehman2025nhs,
  title   = {Predicting Carbon Cost of {NHS} Supply Chains Using Machine Learning},
  author  = {{Ul Rehman}, Sufiyan},
  year    = {2025},
  note    = {Under review}
}
```

## Author

Sufiyan Ul Rehman, Ulster University / Solent University (QA HE).
Contact: sufirehman12@gmail.com

Prior work: [CAML-TC, IEEE, link to be added]
