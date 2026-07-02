# Methodology Note: Circular Labels and SHAP-Based Evaluation

## 1. Why the DEFRA Baseline MAE Comparison is Not a Valid Performance Metric

The target variable in this study, `estimated_kgco2e_consumption`, is
constructed by multiplying each dispensing record's `INDICATIVE_COST` by the
DEFRA 2023 consumption-basis emissions intensity factor of 0.2402 kgCO2e per
pound sterling (NHS Supply Chain category, Table 5). The DEFRA flat-rate
baseline prediction applies the rounded equivalent factor of 0.24 kgCO2e/GBP
to the same `INDICATIVE_COST` column. The difference between the
label and the baseline prediction for any individual row is therefore
algebraically bounded by the rounding discrepancy (0.0002 x spend), producing
a near-zero mean absolute error (~1.42 kgCO2e) and an R-squared of 1.0000.
This result carries no empirical content: it is a mathematical identity arising
from the label construction, not evidence that the DEFRA factor accurately
represents true supply-chain carbon at the product level. Reporting ML model
MAE as a percentage improvement over this baseline would therefore be circular
and misleading. The DEFRA flat baseline is retained in Table 1 as a
methodological reference point to make this structural feature explicit, but
the "MAE improvement vs DEFRA" column has been removed from all paper tables.
MAE, RMSE, and within-20% statistics for ML models are reported as descriptive
statistics of product-feature fit quality, not as comparative improvements.

## 2. Correct Framing: SHAP-Based Feature Importance as the Primary Contribution

The ML models in this study are trained without access to `INDICATIVE_COST` or
`TOTAL_QUANTITY`, both of which are withheld from the feature matrix to avoid
replicating the circular label relationship. The models therefore predict
carbon intensity from pharmacological and clinical product attributes alone:
dosage form (nine categories), biological status, HFC propellant status,
cold-chain requirement, price-per-unit at the Virtual Medicinal Product (VMP)
level, and BNF therapeutic chapter. The Random Forest achieves an R-squared of
0.25 in original carbon space (R2 = 0.763 in log1p space), reflecting
the model's ability to recover product-level carbon variation from categorical
and structural features rather than from spend directly. The primary scientific
contribution is the SHAP (SHapley Additive exPlanations) analysis, which
identifies which product attributes carry independent information about carbon
intensity. The dominant features are `log_price_per_unit` (mean |SHAP| =
1.234), `form_iv_injection` (0.347), `bnf_chapter_num` (0.258),
and `form_tablet_capsule` (0.182). These rankings provide a principled basis
for disaggregating the DEFRA flat intensity factor into product-category-
specific ranges, which is the actionable output for NHS procurement.

## 3. External Validation

Three external sources partially validate the SHAP-derived feature rankings.
First, the engineering and life-cycle assessment literature on pharmaceutical
manufacturing consistently identifies biological manufacturing (microbial
fermentation, mammalian cell culture, downstream purification) and intravenous
sterile manufacturing (aseptic fill-finish, cold-chain distribution) as the
highest-intensity production pathways per unit of active substance, consistent
with the model's high SHAP weights on `is_biologic`, `form_iv_injection`, and
`is_cold_chain`. Second, published NHS Sustainable Development Unit analyses
and the Cambridge Healthcare Research climate-impact report identify HFC-
propellant pressurised metered-dose inhalers as the single largest product-
category carbon hotspot in NHS prescribing, estimated at approximately 3.5% of
total NHS carbon footprint; the model independently recovers this signal in
secondary-care dispensing via `is_inhaler_hfc`, without price as a mediating
variable. Third, the Carbon Reduction Plan disclosures collected from seven UK
NHS suppliers (Fresenius Kabi, B. Braun, Becton Dickinson, Johnson and
Johnson, Lohmann and Rauscher, Smith and Nephew, Siemens Healthcare Diagnostics
and Siemens Healthcare Limited) provide supply-chain intensity ratios that,
where parseable as UK-specific GBP-denominated figures, are directionally
consistent with the SHAP feature ordering for their product categories.

## 4. SCMD INN Naming Convention as a Structural Limitation

A methodological finding with independent significance is that SCMD uses
International Nonproprietary Names (INN, generic names) throughout, assigning
no manufacturer attribution to the large majority of dispensing records. Only
products with no approved generic equivalent retain identifiable supplier
signatures in the VMP name field; in practice this is restricted to parenteral
nutrition formulations marketed exclusively under proprietary brand names
(Kabiven, SMOFlipid, Clinimix, Structokabiven, Prontosan, Calea). Systematic
matching of CRP data from seven suppliers against 8,473 unique SCMD VMPs
yielded 13 matched VMPs across two suppliers (Fresenius Kabi and B. Braun),
covering 0.15% of unique VMPs and less than 0.01% of total estimated spend.
This finding -- that SCMD's INN naming convention structurally prevents
manufacturer attribution for approximately 99.85% of secondary-care medicine
VMPs -- should be cited as a barrier to product-level supplier CRP integration
in NHS carbon accounting, independent of the quality or availability of CRP
disclosures themselves. The `supplier_crp_matched` binary flag is retained in
the feature matrix as a marker of this identifiable proprietary-brand cluster;
its low SHAP rank (below all dosage form categories and price) confirms that
the matched cluster is too small to act as a meaningful carbon predictor at
dataset scale.
