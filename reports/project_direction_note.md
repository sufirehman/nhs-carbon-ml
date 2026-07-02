# Project Direction Note
**Date:** 2026-06-25  
**Status:** Written after source audit — before any modelling

---

## EXIOBASE Access Check

**Short answer:** EXIOBASE is free and open for non-commercial/academic use, but
it does not give you a ready-to-use multiplier table.

**Details:**
- License: CC-BY-SA-NC (free for academic institutions, non-profit research,
  environmental agencies; commercial use requires separate licensing)
- Hosted on Zenodo: https://zenodo.org/records/15689391
- 44 countries + 5 rest-of-world regions, 1995–2022
- Files: ~300–600 MB per year (28.1 GB total across all years)
- **Critical gap:** Pre-computed Leontief inverses and emission intensity
  multipliers are NOT included. They were dropped from the download to reduce
  storage. You must compute them yourself from the A matrix + environmental
  satellite vectors using the PyMRIO Python package.

**What it would take to get a consumption-basis pharma multiplier from EXIOBASE:**
1. Download one yearly file (e.g. `IOT_2022_pxp.zip`, ~400–500 MB)
2. Load with PyMRIO, compute Leontief inverse: `L = (I - A)^-1`
3. Multiply by the GHG satellite vector: `M = emissions_vector @ L`
4. Extract the row corresponding to the pharmaceutical sector
   (EXIOBASE uses CPA/NACE classification; pharma ≈ CPA 21)
5. Pull out the UK column to get kgCO2e per euro of UK pharmaceutical final demand

This is feasible (PyMRIO is already a dependency candidate, the maths is
standard) but it is a meaningful piece of additional work — roughly one
session to do properly with validation. The result would give us a
single consumption-basis scalar comparable to ONS's 0.17 (expected to be
higher, likely 0.4–0.8 range based on other countries' EEIO analyses).

**Decision for now:** Defer EXIOBASE computation. Even with a better single
number, the structural problem below still applies.

---

## The Core Problem: A Flat Multiplier Has No Model Signal

With every BNF chapter at 0.17 kgCO2e/£, the "carbon estimate" per row is:

```
estimated_kgco2e = INDICATIVE_COST × 0.17
```

This is a linear transformation of spend. Sorting rows by `estimated_kgco2e`
is identical to sorting by `INDICATIVE_COST`. An ML model trained on this
label learns nothing that a spreadsheet formula cannot already tell you.

The model is only useful if the TARGET VARIABLE has real variation that is
NOT explained by spend alone. That variation must come from features that
capture genuine differences in emission intensity between procurement lines.

---

## What the Model Actually Needs to Learn From

For the model to add value, it must predict WHERE actual carbon intensity
deviates from the flat baseline. Here is a concrete inventory of features
that create defensible variation, organized by data availability:

### Tier 1 — Derivable from SCMD now (no new data needed)

| Feature | Derivation | Carbon signal |
|---------|-----------|---------------|
| **Dosage form** | Extract from `VMP_PRODUCT_NAME`: "tablets", "solution for infusion", "inhaler", "injection vial", "patch", "cream" | Inhalers (pMDI) carry HFC propellant burden (~50× oral tablet). IV solutions require sterile manufacturing. Oral solid doses are the lowest. Strong engineering basis. |
| **Price per unit** (`INDICATIVE_COST / TOTAL_QUANITY_IN_VMP_UNIT`) | Direct calculation | Proxy for branded vs generic status and manufacturing complexity. Biologics cost £1,000–£10,000/unit vs £0.01–£0.10 for oral generics. Highly correlated with manufacturing emission intensity. |
| **Is biologic / large molecule** | Flag if product name contains: monoclonal antibody suffixes (-mab, -umab, -zumab), "-alfa", "-beta", "-ergin", "immunoglobulin", "epoetin", "filgrastim", "insulin" | Biologics have 3–10× the manufacturing energy of small molecules. Well-documented in published LCA literature (Jimenez-Gonzalez et al. 2011 and others). |
| **Is inhaler** | Flag if "inhaler", "aerosol", "pMDI", "MDI", "Evohaler", "Autohaler", "Easyhaler" in name | HFC propellants in pMDIs have GWP100 of 1,430–3,220 (HFC-134a, HFC-227ea). Well-quantified; NHS has specific bottom-up estimates for this category (NHS Net Zero 2022, p.60). |
| **Is cold chain** | Flag if product is insulin, biologic, vaccine, blood product, or specified refrigerated | Cold chain distribution adds meaningful transport and refrigeration emissions. |
| **Order volume** | `TOTAL_QUANITY_IN_VMP_UNIT` per `ODS_CODE` per product | High-frequency small orders vs consolidated bulk orders affects transport emission per unit. |
| **Trust type proxy** | Cluster ODS codes by total spend profile; teaching hospitals have different procurement mixes to district generals | Contextual signal for model; not a direct carbon driver but useful for trust-level recommendations. |

### Tier 2 — Derivable with one additional lookup (dm+d or ATC mapping)

| Feature | Derivation | Carbon signal |
|---------|-----------|---------------|
| **ATC code / drug class** | SNOMED → INN → WHO ATC classification (open data, no registration) | ATC class enables mapping to published pharmaceutical LCA studies. Several papers have class-specific emission factors (e.g. inhalational anaesthetics, oncology biologics, solid oral generics). |
| **Country of manufacture / import origin** | ATC class + known manufacturer + published supply chain data | Manufacturing in India vs Switzerland vs US vs UK implies different grid carbon intensity and transport distance. This is the biggest single driver of consumption-basis variation but requires external lookups. |
| **Generic vs branded flag** | Price per unit threshold + name pattern (check for brand name without INN) | Generics are typically manufactured at higher volume with more optimized processes, closer to point of sale. Significant emission intensity difference. |

### Tier 3 — Supplier CRP data (primary source of ground-truth variation)

This is now the critical data source for the model, not a validation check.

| Feature | Source | Carbon signal |
|---------|--------|---------------|
| **Supplier-reported Scope 1+2 intensity (kgCO2e/£ revenue)** | CRP PDFs (to be manually collected; see `data/raw/MANUAL_CRP_COLLECTION.md`) | Self-reported baseline. Not perfectly comparable across suppliers (scope boundaries vary) but currently the only product-level ground truth available |
| **Supplier Scope 3 reporting** | CRP PDFs | A minority of suppliers report Category 1 purchased goods intensity; these are the most directly applicable |
| **Supplier's 2030/2035/2050 reduction target** | CRP PDFs | Indicates future trajectory; relevant for recommender: prefer suppliers with steeper near-term declines |
| **Supplier net-zero commitment year** | CRP PDFs | Binary feature: committed vs uncommitted; credibility proxy |
| **Supplier's Science Based Target (SBT) status** | Science Based Targets initiative (sciencebasedtargets.org, open) | SBT-validated commitments are more credible than self-declared targets |

---

## Revised Model Architecture Given These Constraints

The ML problem is no longer:
> "Predict carbon per procurement line from product features"

It is instead:
> "Given spend-based baseline (confirmed lower bound), estimate the UPLIFT
> or DISCOUNT factor for each line based on product emission-intensity features,
> calibrated against supplier CRP ground truth where available"

**Practically this means:**

1. The target variable `y` should be a **relative intensity ratio**:  
   `actual_kgco2e_per_gbp / baseline_0.17` — derived from supplier CRPs  
   where available, imputed from Tier 1/2 features elsewhere.

2. The model's job is to rank procurement lines by **deviation from baseline**,
   not to predict an absolute carbon number. This is a ranking/scoring problem
   as much as a regression problem.

3. The recommender output is: *"For product X, supplier B has a lower
   reported intensity than supplier A — switch recommendation generates
   Y tCO2e saving at Z% spend premium."*

4. The 42.3% unclassified rate affects feature quality but NOT the fundamental
   model structure — Tier 1 features (dosage form, price/unit, biologic flag)
   do not require BNF classification.

---

## Priority Order for Next Work

1. **Fix classification (42% unclassified)** — derive Tier 1 features cleanly  
2. **Build Tier 1 feature set** — dosage form, biologic flag, inhaler flag, price/unit  
3. **Collect 15–20 supplier CRPs** (manual) — this is the label source  
4. **Parse CRPs** — extract scope 1+2 intensity, target year, SBT status  
5. **Join CRP data to SCMD** on supplier name / product name  
6. **Optional: compute EXIOBASE UK pharma consumption-basis multiplier**  
   (replace 0.17 lower bound with a better single anchor; one session of work)

The model cannot be meaningfully trained until step 4 is complete.
Classification can be improved in parallel with CRP collection.
