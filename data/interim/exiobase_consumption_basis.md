# UK Pharmaceutical GHG Intensity: Production vs Consumption Basis

## Summary Table

| Figure | kgCO2e/GBP | Basis | Source | Status |
|--------|-----------|-------|--------|--------|
| 0.17 | Production-basis | ONS Atmospheric Emissions, SIC 21, 2024 | Used as prior baseline |
| **0.24** | **Consumption-basis** | **DEFRA/BEIS UK Carbon Footprint 2022, CPA 21** | **PRIMARY ANCHOR — see rationale below** |
| 0.39 | Consumption-basis | EXIOBASE 3 IOT_2022_pxp, 'Chemicals nec' proxy | Upper-bound sanity check only |
| 0.55 | Consumption-basis | DEFRA/BEIS UK Carbon Footprint 2019, CPA 21 | Pre-COVID reference; not used — see caveat |

## Primary Anchor: 0.24 kgCO2e/GBP (DEFRA 2022, CPA 21)

**Source:** UK Government (DEFRA/BEIS), "UK and England's carbon footprint to 2022."
Published May 2025. Conversion factors by CPA code, file: `Conversion_factors_kgCO2_per_£_spent_by_SIC_code.ods`.
CPA row 21 "Basic pharmaceutical products and pharmaceutical preparations": **0.2402 kgCO2e/GBP** (GHG total); 0.1808 kgCO2/GBP (CO2 only).

**Why this figure was chosen as primary anchor:**

This is the dedicated pharmaceutical-sector figure from the UK government's own consumption-basis EEIO model, built from the ONS Supply and Use tables with import content adjustments. Crucially, it refers specifically to CPA 21 (pharmaceuticals), not a broader chemicals aggregate. It is therefore not subject to the sectoral dilution that affects the EXIOBASE estimate (see below). It is also the figure that NHS procurement sustainability reporting, industry Scope 3 Category 1 calculations, and peer reviewers would implicitly reference when comparing our results — which makes alignment with it a priority for the paper's credibility. Data year is 2022, consistent with the SCMD dataset.

**Why not the EXIOBASE Chemicals nec figure (0.39)?**

EXIOBASE 3's 200-product-by-product classification does not contain a dedicated pharmaceutical sector. CPA 21 is absorbed into 'Chemicals nec', which also covers a range of higher-intensity industrial and specialty chemicals. The resulting sector average (0.39 kgCO2e/GBP after correcting a GWP double-counting error for HFC/PFC) is systematically upward-biased relative to pharmaceuticals alone. It is retained as an independent upper-bound sanity check — it confirms the DEFRA figure is in the right order of magnitude — but should not be cited as the pharmaceutical intensity.

**Why not the DEFRA 2019 figure (0.55)?**

This figure is not used as primary anchor due to its large downward revision by 2022. See the COVID-era hypothesis below. It is retained as a pre-COVID structural reference.

## DEFRA 2019 to 2022 Drop: An Unverified Hypothesis

The DEFRA spend-based factor for CPA 21 pharmaceuticals fell from **0.549 kgCO2e/GBP (2019)** to **0.240 kgCO2e/GBP (2022)** — a 56% decline in three years. This is implausibly fast as a genuine decarbonisation signal for global pharmaceutical supply chains.

**Unverified hypothesis:** This drop is largely a denominator effect driven by COVID-era pharmaceutical spending. Between 2019 and 2022, NHS pharmaceutical procurement expanded substantially (COVID vaccines, antivirals, PPE-adjacent spend reclassified into pharmaceutical accounts). In an EEIO spend-based model, the kgCO2e/£ factor is calculated as total supply-chain emissions / total monetary spend. If pharmaceutical spend grew faster than supply-chain emissions — which is plausible given vaccine roll-out volumes priced at marginal cost — the factor would deflate even with no underlying change in physical emission intensity.

This is explicitly labelled as a hypothesis, not a confirmed finding. We have not decomposed the DEFRA model's numerator (supply-chain CO2e) and denominator (£ spend) series separately to verify it. It is also possible that genuine efficiency improvements or grid decarbonisation contributed to the decline. The hypothesis is stated here to justify why the 2019 figure is retained as a reference rather than discarded, and to flag the 2022 figure as potentially representing a temporary price anomaly rather than the structural intensity of pharmaceutical procurement.

## EEIO Volatility as Motivation for the ML Approach

The observed 56% swing in the DEFRA CPA 21 factor between 2019 and 2022 is itself a citable illustration of a fundamental limitation of spend-based EEIO accounting: emission intensity factors are price-denominated and therefore conflate genuine supply-chain decarbonisation with purchasing price changes, making them an unstable ground truth for procurement-level carbon decisions. This directly motivates the ML approach taken in this paper — rather than applying a volatile sector-average multiplier uniformly to all procurement lines, the model learns product-level features (dosage form, biologic status, price-per-unit, supplier CRP data) that predict where individual lines deviate from the sector average, providing a more stable and actionable ranking of procurement alternatives.

## Full Benchmarking Narrative

Starting from first principles:

**Production basis (ONS, 0.17 kgCO2e/GBP):** Captures only UK domestic pharmaceutical manufacturing emissions per pound of output. Does not include imported API synthesis, overseas formulation, or international transport. Represents a strict lower bound on the true supply chain intensity for NHS procurement, which sources most of its APIs from overseas manufacturers. Used as the original baseline before this work.

**Consumption basis, DEFRA 2022 (0.24 kgCO2e/GBP):** The UK government's own EEIO-derived estimate for the full supply chain intensity of pharmaceutical final demand, including imported content. The ratio to production-basis (0.24/0.17 = 1.41x) implies imported supply chain adds approximately 41% to the UK-domestic emission intensity — a smaller uplift than expected, likely reflecting the denominator effect discussed above. Primary anchor for all subsequent analysis.

**Consumption basis, EXIOBASE 2022 (0.39 kgCO2e/GBP):** Independent cross-check derived from EXIOBASE 3 IOT_2022_pxp using the 'Chemicals nec' sector as a proxy for pharmaceuticals (no dedicated pharma sector exists in the pxp 200-product classification). Ratio to DEFRA 2022: 0.39/0.24 = 1.63x — consistent with the expected upward bias from chemical sector dilution. Confirms the DEFRA figure is not anomalously high; EXIOBASE sets a credible upper bound.

**Consumption basis, DEFRA 2019 (0.55 kgCO2e/GBP):** Pre-COVID structural reference. Ratio to 2022: 0.55/0.24 = 2.29x. Retained as evidence that the 2022 figure may be price-distorted. If the true structural intensity is closer to the 2019 level, our 0.24-based consumption estimates are a lower bound on consumption-basis carbon.

## Dataset Columns Added

Both baseline columns are now present in `data/interim/scmd_with_defra_baseline.csv`:

| Column | Factor | Basis | Notes |
|--------|--------|-------|-------|
| `estimated_kgco2e` | 0.17 kgCO2e/GBP | ONS production-basis | Original baseline column |
| `estimated_kgco2e_consumption` | 0.24 kgCO2e/GBP | DEFRA 2022 consumption-basis | New — applied uniformly to all rows |

The consumption-basis factor is applied uniformly regardless of BNF chapter, consistent with its source: a sector-level average for CPA 21 that does not differentiate by drug class.

## Citations

**Primary anchor (DEFRA 2022):**
Department for Environment, Food & Rural Affairs (DEFRA) / Department for Energy Security and Net Zero (DESNZ) (2025). UK and England's carbon footprint to 2022. Published 14 May 2025. Table: Conversion factors kgCO2 per £ spent, by SIC code. CPA row 21: 0.2402 kgCO2e/GBP. Available: https://www.gov.uk/government/statistics/uks-carbon-footprint

**Production basis (ONS):**
Office for National Statistics (2025). UK Environmental Accounts: Atmospheric Emissions — Greenhouse Gas Emissions Intensity by Economic Sector. SIC 21, year 2024 = 0.17 kgCO2e/GBP. Available: https://www.ons.gov.uk/economy/environmentalaccounts/datasets/ukenvironmentalaccountsatmosphericemissionsgreenhousegasemissionsintensitybyeconomicsectorunitedkingdom

**EXIOBASE upper-bound cross-check:**
Stadler K, Wood R, Bulavskaya T, et al. (2018). EXIOBASE 3: Developing a Time Series of Detailed Environmentally Extended Multi-Regional Input-Output Tables. Journal of Industrial Ecology, 22(3), 502–515. DOI: https://doi.org/10.1111/jiec.12715
Data: EXIOBASE consortium (2024). Zenodo record 15689391. IOT_2022_pxp.zip. License CC-BY-SA-NC 4.0.
Computation: Leontief demand-pull on 'Chemicals nec' (GB) using GWP100-weighted GHG stressor vector (IPCC AR5). HFC and PFC stressors are reported in kg CO2-eq in EXIOBASE and were NOT additionally weighted by GWP — they are passed through with weight = 1.0.

**DEFRA 2019 reference:**
Same source series, 2019 data year. Previously available via Climatiq emission factor ID 57fa89c4-3ad3-4b5e-823c-bd3f870853aa (deprecated in Climatiq Data Version 14).
