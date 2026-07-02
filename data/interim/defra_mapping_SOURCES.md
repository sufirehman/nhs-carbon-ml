# Emission Factor Sources — BNF-to-Carbon Mapping

**Project:** NHS-Carbon-ML  
**Created:** 2026-06-25  
**Status:** v2 — all factors must trace to a citable source

---

## Summary Table

| BNF Chapters | Factor (kgCO2e/£) | Source | Basis | Confidence |
|---|---|---|---|---|
| Ch 0–15 (all) | **0.17** | ONS 2026, SIC 21, 2024 data | UK production-basis | sourced-lower-bound |

**There is currently ONE factor applied to all 16 BNF chapters.**  
No public source permits differentiation between pharmaceutical sub-categories.

---

## What Was Searched, Found, and Why

### SOURCE 1 — USED (primary factor)

**ONS (2026). UK Environmental Accounts: Atmospheric emissions: greenhouse gas emissions intensity by industry.**  
Dataset: `04atmosphericemissionsghgintensity.xlsx`  
Publisher: Office for National Statistics / Ricardo Energy and Environment  
Published: 5 June 2026  
Coverage: 1990–2024, ~130 UK SIC industry groups  
URL: https://www.ons.gov.uk/economy/environmentalaccounts/datasets/ukenvironmentalaccountsatmosphericemissionsgreenhousegasemissionsintensitybyeconomicsectorunitedkingdom  
File saved locally: `data/raw/ons_ghg_intensity_by_industry.xlsx`

**Relevant extracted values (2024, kgCO2e per GBP):**

| SIC | Industry | 2022 | 2023 | 2024 |
|-----|----------|------|------|------|
| SIC 21 | Basic pharmaceutical products and preparations | 0.11 | 0.16 | **0.17** |
| SIC C | Total manufacturing | 0.42 | 0.51 | 0.54 |
| SIC Q | Human health and social work activities | 0.02 | 0.02 | 0.02 |
| SIC 86 | Human health services | 0.02 | 0.02 | 0.02 |
| SIC 32 | Other manufactured goods (incl. medical instruments) | SUPPRESSED | SUPPRESSED | SUPPRESSED |
| SIC 46 | Wholesale trade excl. motor vehicles | 0.06 | 0.05 | 0.05 |
| SIC 49.3-5 | Land transport excl. rail | 0.81 | 0.52 | 0.47 |
| SIC 52 | Warehousing and support transport services | 14.85 | 8.44 | 8.37 |
| Total | Total economy excl. consumer expenditure | 0.04 | 0.03 | 0.03 |

**Factor used:** SIC 21, 2024 = **0.17 kgCO2e/GBP**

**Citation for paper:**  
> Office for National Statistics (2026). *UK Environmental Accounts: Atmospheric emissions: greenhouse gas emissions intensity by industry, United Kingdom, 1990–2024* [dataset]. Published 5 June 2026. Available at: https://www.ons.gov.uk/economy/environmentalaccounts/datasets/ukenvironmentalaccountsatmosphericemissionsgreenhousegasemissionsintensitybyeconomicsectorunitedkingdom [Accessed 25 June 2026].

**Critical caveat — UK production-basis only:**  
This figure represents direct and indirect GHG emissions from UK pharmaceutical manufacturers per £ of GVA. It does NOT include:
- Emissions embedded in imported active pharmaceutical ingredients and excipients
- Upstream supply chain effects across global production (important for branded medicines and biologics, most of which are manufactured outside the UK)
- The full consumption-basis footprint as modelled in MRIO approaches

The true Scope 3 Category 1 intensity for NHS pharmaceutical procurement is **higher** than 0.17 kgCO2e/£. This figure is a confirmed lower bound.

---

### SOURCE 2 — FOUND (methodology only, no downloadable multiplier table)

**NHS England (2022). *Delivering a 'Net Zero' National Health Service* (July 2022 update).**  
URL: https://www.england.nhs.uk/greenernhs/wp-content/uploads/sites/51/2022/07/B1728-delivering-a-net-zero-nhs-july-2022.pdf  
File saved locally: `data/raw/nhs_delivering_net_zero_2022.pdf`  
Pages consulted: 56–63 (methodology section)

**What it says:**  
Pages 56–58 confirm that NHS supply chain (Scope 3) emissions are calculated using a UK Multi-Region Input-Output (MRIO) model developed by researchers at University of Leeds for Defra, combining UK national input-output tables with EXIOBASE (an international MRIO). The model applies "EEIO carbon intensities per unit spend (kgCO2e/£) for 105 economic sectors" (p.56). The 2020 Carbon Footprint update uses the 2016 EEIO model.

Table 2 (pp.61–62) confirms "Supply chain: Top-down, Source: EEIO."

**What it does NOT contain:**  
The 105-sector multiplier table itself is not published in this document or in any linked annex. Footnote 47 references the MRIO model but no publicly downloadable version of the multiplier table was found.

**Citation for paper:**  
> NHS England (2022). *Delivering a 'Net Zero' National Health Service*. July 2022. Available at: https://www.england.nhs.uk/greenernhs/publication/delivering-a-net-zero-national-health-service/ [Accessed 25 June 2026].

---

### SOURCE 3 — NOT FOUND (paywalled)

**Tennison I, Roschnik S, Ashby B, et al. (2021). Health care's response to climate change: a carbon footprint assessment of the NHS in England. *Lancet Planetary Health* 5(2): e84–e92.**  
DOI: 10.1016/S2542-5196(21)00005-X

Attempted access: returns HTTP 403 Forbidden. Supplementary data not accessible.  
This paper uses the same NHS MRIO methodology. Its supplementary data likely contains category-level emission breakdowns, but cannot be verified without institutional journal access.

**Action required:** Obtain through institutional library access or request supplementary data from corresponding author (Imogen Tennison, NHS England Greener NHS).

---

### SOURCE 4 — NOT FOUND (DEFRA EEIO tables, no public download)

The Defra EEIO model (referenced as footnote 47 in the NHS Net Zero report) is the specific model that generated the 105-sector multipliers used in NHS carbon accounting. Multiple gov.uk URL patterns were tried; all return HTTP 404. The tables may have been removed from public access or were never publicly hosted as a standalone download.

**Attempted URLs (all 404):**
- `gov.uk/government/publications/uk-environmentally-extended-input-output-eeio-analytical-tables`
- `gov.uk/government/publications/uk-environmentally-extended-input-output-analytical-tables-2013`
- `gov.uk/guidance/defra-eeio-model`
- `gov.uk/government/collections/uk-eeio-model`
- `gov.uk/government/statistics/uk-eeio-model`

**Action required:** Contact Defra directly, or access via University of Leeds research data repository (the model was developed there).

---

### SOURCE 5 — NOT APPLICABLE (DEFRA 2025 GHG Conversion Factors)

The downloaded DEFRA/DESNZ 2025 GHG Conversion Factors spreadsheet (`ghg_conversion_factors_2025_full_set.xlsx`) was inspected in full. It contains zero spend-based emission factors. All Scope 3 factors are in physical units (per tonne of material, per km of travel, per kWh, etc.). This source is not suitable for spend-based Scope 3 Category 1 accounting.

---

## What the Revised Mapping CANNOT Do (and Why)

| Desired differentiation | Data available? | Why not |
|---|---|---|
| Oncology biologics vs generic oral tablets | No | No public source provides per-class SIC 21 sub-factors |
| Small molecules vs monoclonal antibodies | No | ONS does not sub-divide SIC 21 |
| UK-manufactured vs imported products | No | ONS is UK production-basis; import adjustment not public |
| BNF chapter-specific factors (ch1–ch15) | No | All medicines map to same SIC 21; no finer public data |

---

## Research Path to Better Factors

In priority order:

1. **Tennison et al. (2021) supplementary data** — obtain institutional access and extract category-level factors (these likely reflect the NHS MRIO multipliers in aggregate form)

2. **NHS England direct request** — the Greener NHS team may be able to share the 105-sector EEIO multiplier table used in the Carbon Footprint model under a data sharing agreement

3. **EXIOBASE MRIO** — open-data international MRIO at https://www.exiobase.eu. The NHS model was "adapted" from EXIOBASE; querying it for pharmaceutical sector emission intensities at UK level would give a consumption-basis figure

4. **University of Leeds UK MRIO** — contact the research group (Dr Jing Meng or successor) to access the adapted model

5. **ONS Supply and Use tables + ONS emissions data** — with both, it is possible to compute a rough consumption-basis multiplier by industry. This is methodologically valid but technically involved.

---

## Version History

| Version | Date | Change |
|---|---|---|
| v1 | 2026-06-25 | Initial mapping — REPLACED. Factors sourced from general knowledge, not citable sources. Ranged 0.58–1.15 kgCO2e/£ by chapter. All figures removed. |
| v2 | 2026-06-25 | All factors replaced with ONS SIC 21 = 0.17 kgCO2e/£ (2024, production-basis). Single factor, fully cited. |
