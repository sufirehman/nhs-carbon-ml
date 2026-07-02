"""
Compute UK pharmaceutical sector consumption-basis GHG intensity from EXIOBASE 3.

Method: Leontief demand-pull multiplier for the UK pharmaceutical sector.
For a unit of final demand in sector j of region r, total GHG emitted across
the global supply chain = sum over all (sector, region) pairs of:
    M[ghg, s, r2] * L[s, r2, j, r]
where L = (I - A)^-1 is the Leontief inverse and M is the stressor coefficient matrix.
Equivalently: intensity_j = (e_ghg @ S @ L)[:, idx_j]   summed over all regions.

Since we only need ONE column of L (the UK pharma sector), we solve:
    (I - A) @ x = e_j   =>   x = L @ e_j
and then intensity = S_ghg @ x, summed.
This avoids computing the full Leontief inverse.

EXIOBASE version:  3.8.2 (IOT_2022_pxp.zip, Zenodo record 15689391)
Reference year:    2022
Currency:          EUR (2022 prices)
License:           CC-BY-SA-NC 4.0
Zenodo DOI:        https://zenodo.org/records/15689391

Output: data/interim/exiobase_consumption_basis.md
"""

import os
import sys
import warnings
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR  = os.path.join(PROJECT_ROOT, "data", "raw")
INT_DIR  = os.path.join(PROJECT_ROOT, "data", "interim")

EXIOBASE_ZIP = os.path.join(RAW_DIR, "IOT_2022_pxp.zip")

# EUR/GBP average exchange rate for 2022
# Source: Bank of England statistical data, annual average 2022
# GBP/EUR = 1.1725, so EUR/GBP = 0.8529
# => 1 GBP = 1.1725 EUR; so kgCO2e/EUR * 1.1725 = kgCO2e/GBP
EUR_TO_GBP_2022 = 1.0 / 1.1725  # = 0.8529 EUR per GBP
GBP_PER_EUR = 1.0 / EUR_TO_GBP_2022  # = 1.1725 GBP per EUR


def load_exiobase(zip_path):
    try:
        import pymrio
    except ImportError:
        sys.exit("pymrio not installed. Run: pip install pymrio")

    print(f"Loading EXIOBASE from {zip_path} ...")
    print("  (This may take 1-2 minutes — large file)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exio = pymrio.parse_exiobase3(path=zip_path)
    print(f"  Regions:  {len(exio.get_regions())}")
    print(f"  Sectors:  {len(exio.get_sectors())}")
    return exio


def find_pharma_sector(exio):
    """
    Find the best available proxy for the UK pharmaceutical sector.

    EXIOBASE 3 pxp (200-product classification) does NOT include a dedicated
    pharmaceutical sector (CPA 21). Pharmaceuticals are aggregated into
    'Chemicals nec' which covers CPA 20/21 residuals. This is a documented
    limitation of the EXIOBASE pxp sector aggregation scheme.

    Reference: Stadler et al. (2018), Table A1 — no separate CPA 21 row.
    The ixi (industry-by-industry) version has the same constraint.

    Best available proxy: 'Chemicals nec' — covers basic chemical and
    pharmaceutical manufacturing outputs not elsewhere classified.
    """
    regions = list(exio.get_regions())
    sectors = list(exio.get_sectors())

    # Check for any pharma-related label first (for future-proofing)
    pharma_hits = [s for s in sectors if "pharmac" in s.lower() or "medicament" in s.lower()]
    if pharma_hits:
        pharma_sector = pharma_hits[0]
        print(f"  Found dedicated pharma sector: '{pharma_sector}'")
    else:
        # EXIOBASE pxp v3: no dedicated pharma sector.
        # 'Chemicals nec' is the closest proxy (aggregates CPA 20/21 residuals).
        pharma_sector = "Chemicals nec"
        if pharma_sector not in sectors:
            # Print all sectors for debug
            chem_hits = [s for s in sectors if "chem" in s.lower()]
            raise ValueError(
                f"'Chemicals nec' not found. Chemical sectors available: {chem_hits}"
            )
        print(f"  WARNING: No dedicated pharmaceutical sector in EXIOBASE pxp 200-sector scheme.")
        print(f"  EXIOBASE CPA 21 (pharmaceuticals) is aggregated into broader chemicals.")
        print(f"  Using proxy sector: '{pharma_sector}'")
        print(f"  This is a known EXIOBASE pxp limitation — see output file for full caveat.")

    if "GB" not in regions:
        uk_candidates = [r for r in regions if "UK" in r or "GB" in r or "United Kingdom" in r]
        print(f"  GB not found; UK candidates: {uk_candidates}")
        uk_region = uk_candidates[0] if uk_candidates else None
    else:
        uk_region = "GB"
    print(f"  UK region code: '{uk_region}'")

    return pharma_sector, uk_region


def build_ghg_weights():
    """
    GWP100 weights for EXIOBASE air_emissions stressors.
    Values from IPCC AR5 (2013), consistent with most EXIOBASE 3.x publications.
    Biogenic CO2 = 0 (short-cycle, excluded from national GHG inventories).

    Returns dict mapping stressor name substring -> GWP100 scalar.
    Order matters: more specific patterns first.
    """
    return {
        # CO2 (fossil)
        "CO2 - combustion": 1.0,
        "CO2 - non combustion": 1.0,
        "CO2 - agriculture - peat": 1.0,
        "CO2 - waste - fossil": 1.0,
        # Biogenic CO2 — excluded (short-cycle carbon; = 0 in national inventories)
        "CO2_bio": 0.0,
        "CO2 - waste - biogenic": 0.0,
        # CH4 — IPCC AR5 GWP100 = 28 (fossil+biogenic combined, no climate feedback)
        "CH4": 28.0,
        # N2O — IPCC AR5 GWP100 = 265
        "N2O": 265.0,
        # SF6 — IPCC AR5 GWP100 = 23,500
        "SF6": 23500.0,
        # HFC — mixed basket; IPCC AR5 ~1,430 (HFC-134a dominant in EU)
        "HFC": 1430.0,
        # PFC — mixed basket; IPCC AR5 ~7,390 (CF4/C2F6 mixture)
        "PFC": 7390.0,
    }


def build_ghg_intensity_vector(exio):
    """
    Aggregate GHG stressors from air_emissions.S into a single kgCO2e S-vector.

    Uses pymrio-computed S matrix (via calc_extensions), which correctly handles
    zero-output sectors (S set to 0 where x=0, avoiding division-by-zero inflation).
    S = F / x, where F is total physical emissions and x is output in million EUR.
    S is therefore in kg_gas / million EUR of output.

    We convert to kgCO2e / million EUR using GWP100 weights, then to kgCO2e / EUR
    by dividing by 1,000,000 (since x is in million EUR units).
    """
    ext = exio.air_emissions
    S = ext.S  # pymrio-computed: stressors × sectors, in units / million EUR

    gwp = build_ghg_weights()
    stressor_names = list(S.index)

    print("\nBuilding GHG CO2e vector from air_emissions.S (pymrio-computed)...")
    print("  GWP100 source: IPCC AR5 (2013)")
    print(f"  S matrix shape: {S.shape}  (stressors x sectors)")

    ghg_S_total = np.zeros(S.shape[1])
    matched = {}

    for stressor in stressor_names:
        weight = None
        for pattern, gwp_val in gwp.items():
            if pattern in stressor:
                weight = gwp_val
                break
        if weight is not None and weight > 0:
            s_row = S.loc[stressor].values.astype(float)
            # Replace NaN (zero-output sectors that pymrio sets to NaN)
            s_row = np.nan_to_num(s_row, nan=0.0, posinf=0.0, neginf=0.0)
            ghg_S_total += weight * s_row
            matched[stressor] = weight

    # S is in kg_gas per million EUR — convert to kgCO2e per EUR
    ghg_S_total /= 1_000_000.0

    print(f"  Stressors included: {len(matched)}")
    for s, w in list(matched.items())[:8]:
        print(f"    GWP={w:>6}  {s}")
    if len(matched) > 8:
        print(f"    ... and {len(matched)-8} more")

    # ghg_S finite stats (exclude zero-output sectors)
    finite_vals = ghg_S_total[ghg_S_total > 0]
    print(f"  ghg_S (non-zero) range: {finite_vals.min():.6f} to {finite_vals.max():.4f} kgCO2e/EUR")

    return ghg_S_total


def compute_intensity(exio, pharma_sector, uk_region):
    """
    Compute GHG intensity for the proxy sector via Leontief demand-pull.

    Steps:
    1. Build GHG intensity vector (kgCO2e per EUR output per sector) by
       summing air_emissions stressors weighted by GWP100.
    2. Compute A matrix (technical coefficients).
    3. Solve (I-A) x = e_j for the UK proxy sector column j.
       x = Leontief column = supply chain requirements per unit final demand.
    4. intensity = ghg_vector @ x  (dot product = total kgCO2e per EUR FD).
    """
    # Step 1: compute A, L (system) and extension S matrices
    print("\nRunning calc_system() + calc_extensions()...")
    print("  (This computes A, L, and extension S = F/x — may take 2-5 min)")
    exio.calc_system()
    exio.calc_extensions()
    A = exio.A.values
    n = A.shape[0]
    print(f"  A matrix shape: {n} x {n}")

    # Step 2: GHG S-vector from air_emissions.S (pymrio-normalised)
    ghg_vec = build_ghg_intensity_vector(exio)

    # Step 3: locate the UK proxy sector
    idx = exio.A.index  # MultiIndex (region, sector)
    try:
        col_idx = idx.get_loc((uk_region, pharma_sector))
    except KeyError:
        try:
            col_idx = idx.get_loc((pharma_sector, uk_region))
        except KeyError:
            raise ValueError(f"Cannot find ({uk_region}, {pharma_sector}) in A index")
    print(f"  UK '{pharma_sector}' sector index in A: {col_idx}")

    e_j = np.zeros(n)
    e_j[col_idx] = 1.0

    print("  Solving (I-A) x = e_j  [Leontief column, ~9800x9800 system]...")
    print("  (This may take 30-120 seconds)")
    I_minus_A = np.eye(n) - A
    x = np.linalg.solve(I_minus_A, e_j)
    print(f"  Done. Supply chain sum: {x.sum():.4f}")

    # Step 4: dot product
    # ghg_vec is in kgCO2e/EUR, x (Leontief column) is dimensionless (EUR/EUR)
    intensity_eur = float(ghg_vec @ x)

    intensity_gbp = intensity_eur * GBP_PER_EUR
    print(f"\n  Result: {intensity_eur:.4f} kgCO2e per EUR")
    print(f"          {intensity_gbp:.4f} kgCO2e per GBP  (rate: {EUR_TO_GBP_2022:.4f} EUR/GBP)")

    return intensity_eur, intensity_gbp


def write_output(intensity_eur, intensity_gbp, pharma_sector, uk_region):
    ons_production_basis = 0.17  # kgCO2e/GBP, ONS 2024
    ratio = intensity_gbp / ons_production_basis

    lines = [
        "# EXIOBASE Consumption-Basis GHG Intensity: UK Pharmaceuticals",
        "",
        "## Result",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Consumption-basis intensity (EXIOBASE)** | **{intensity_gbp:.4f} kgCO2e per GBP** |",
        f"| Production-basis intensity (ONS SIC 21, 2024) | 0.1700 kgCO2e per GBP |",
        f"| **Ratio (consumption / production)** | **{ratio:.2f}x** |",
        f"| Intensity in EUR units | {intensity_eur:.4f} kgCO2e per EUR |",
        f"| EUR/GBP rate applied (2022 annual avg) | {EUR_TO_GBP_2022:.4f} |",
        "",
        "## Interpretation",
        "",
        f"The consumption-basis figure of **{intensity_gbp:.2f} kgCO2e per GBP** captures the full",
        "global supply chain embedded in UK pharmaceutical final demand, including:",
        "- Emissions from importing active pharmaceutical ingredients from overseas",
        "- Upstream chemical and energy inputs to manufacturing",
        "- Packaging, transport, and distribution",
        "",
        f"It is **{ratio:.1f}x higher** than the ONS production-basis figure of 0.17 kgCO2e/GBP.",
        "This ratio is itself a citable finding: it quantifies the import/supply-chain",
        "correction factor needed when moving from domestic production-basis to full",
        "consumption-basis Scope 3 accounting for NHS pharmaceutical procurement.",
        "",
        "## Full Citation",
        "",
        "**Dataset:**",
        "Stadler K, Wood R, Bulavskaya T, et al. (2018). EXIOBASE 3: Developing a",
        "Time Series of Detailed Environmentally Extended Multi-Regional Input-Output",
        "Tables. Journal of Industrial Ecology, 22(3), 502-515.",
        "DOI: https://doi.org/10.1111/jiec.12715",
        "",
        "**Data record (version used):**",
        "EXIOBASE consortium (2024). EXIOBASE 3 [dataset]. Zenodo.",
        "Record: https://zenodo.org/records/15689391",
        "File used: IOT_2022_pxp.zip (year 2022, product-by-product)",
        "License: CC-BY-SA-NC 4.0 (non-commercial use)",
        "",
        "**Exchange rate:**",
        "Bank of England (2023). Annual average EUR/GBP exchange rate 2022.",
        "Series: XUMAUSS / XUMAEUS. GBP per EUR = 1.1725 (annual average 2022).",
        "",
        "## Computation Method",
        "",
        "1. Loaded EXIOBASE 3 IOT_2022_pxp.zip using PyMRIO 0.6.x",
        f"2. Located UK proxy sector: region='{uk_region}', sector='{pharma_sector}'",
        "3. Aggregated air_emissions GHG stressors into single kgCO2e vector using",
        "   IPCC AR5 GWP100 weights (CO2=1, CH4=28, N2O=265, SF6=23500, HFC=1430, PFC=7390)",
        "4. Computed technical coefficient matrix A = Z / x (flow / output)",
        "5. Solved Leontief column: (I - A) @ x_col = e_proxy_GB",
        "   (avoids full matrix inversion; equivalent to extracting one column of L)",
        "6. Computed intensity = ghg_vector @ x_col  (dot product)",
        "   = total kgCO2e per EUR of UK chemicals-nec final demand",
        "7. Converted EUR to GBP using 2022 annual average exchange rate",
        "",
        "## CRITICAL CAVEAT: Sector Proxy — Not a Dedicated Pharmaceutical Estimate",
        "",
        "**EXIOBASE 3 pxp (200-sector classification) does NOT include a separate",
        "pharmaceutical sector.** CPA 21 (basic pharmaceutical products and",
        "pharmaceutical preparations) is aggregated into the 'Chemicals nec' sector,",
        "which also covers basic industrial chemicals not elsewhere classified.",
        "",
        "This means the intensity figure above reflects the GHG intensity of a MIXED",
        "chemicals sector, not pharmaceuticals specifically. Depending on the",
        "composition of 'Chemicals nec' in the UK IO table, this could be:",
        "- An overestimate if basic industrial chemicals (lower intensity) dominate",
        "- An underestimate if pharmaceutical-specific processes (biologics, API",
        "  synthesis) are less carbon-intensive than bulk chemicals in that region",
        "",
        "**DO NOT cite this figure as the UK pharmaceutical consumption-basis intensity**",
        "without this caveat. It is a chemicals sector proxy.",
        "",
        "**To get a true dedicated pharmaceutical figure, options are:**",
        "1. Use EXIOBASE ixi (industry-by-industry) and check if NACE 21 is separate",
        "2. Use the UK ONS Supply and Use tables (purchasers' price basis) and build",
        "   a UK-only EEIO model — NACE 21 is a distinct row in the SUT",
        "3. Use published literature: Tennison et al. (2021) Lancet Planetary Health",
        "   report NHS-specific pharmaceutical supply chain emissions",
        "4. Contact the University of Leeds MRIO group who built the NHS model",
        "",
        "## Other Caveats",
        "",
        "- EXIOBASE 2022 data is nowcast (estimated); 2019 is the last fully",
        "  reconciled year for some regions",
        "- EUR/GBP rate introduces ~1-2% uncertainty given annual volatility",
        "- Does not disaggregate by molecule type (small molecule vs biologic)",
        "- CC-BY-SA-NC license: results are freely citable in academic papers",
        "  but derived commercial products require separate licensing",
    ]

    out_path = os.path.join(INT_DIR, "exiobase_consumption_basis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {out_path}")

    # Also print summary to console
    print()
    print("=" * 55)
    print("EXIOBASE RESULT SUMMARY")
    print("=" * 55)
    print(f"  Consumption-basis: {intensity_gbp:.4f} kgCO2e/GBP")
    print(f"  Production-basis:  0.1700 kgCO2e/GBP  (ONS SIC 21, 2024)")
    print(f"  Ratio:             {ratio:.2f}x")
    print(f"  -> The full supply chain adds {ratio:.1f}x the UK production emissions")
    print("=" * 55)


def main():
    if not os.path.exists(EXIOBASE_ZIP):
        sys.exit(f"EXIOBASE zip not found: {EXIOBASE_ZIP}\nRun fetch script first.")

    exio = load_exiobase(EXIOBASE_ZIP)
    pharma_sector, uk_region = find_pharma_sector(exio)
    intensity_eur, intensity_gbp = compute_intensity(exio, pharma_sector, uk_region)
    write_output(intensity_eur, intensity_gbp, pharma_sector, uk_region)


if __name__ == "__main__":
    main()
