"""
Match supplier Carbon Reduction Plans (CRPs) to SCMD product names.

Inputs:
  data/raw/supplier_crps_manual.csv
  data/processed/scmd_with_pharma_features.csv

Outputs:
  data/interim/supplier_crp_matches.csv   -- matching results per supplier
  data/processed/scmd_with_all_features.csv -- pharma features + 2 new columns

New columns added to pharma features:
  supplier_carbon_intensity  -- kgCO2e/£ (from intensity ratio; null if unmatched)
  supplier_crp_matched       -- 1/0 flag

Run:
  python src/features/match_supplier_crp.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
INT_DIR      = os.path.join(PROJECT_ROOT, "data", "interim")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")

CRP_PATH      = os.path.join(RAW_DIR,  "supplier_crps_manual.csv")
FEATURES_PATH = os.path.join(PROC_DIR, "scmd_with_pharma_features.csv")
MATCHES_PATH  = os.path.join(INT_DIR,  "supplier_crp_matches.csv")
OUTPUT_PATH   = os.path.join(PROC_DIR, "scmd_with_all_features.csv")


# ── Per-supplier matching config ──────────────────────────────────────────────
#
# Each entry maps a substring of supplier_name (lower-case) to:
#   patterns     : list of regex strings searched against VMP_PRODUCT_NAME
#   match_method : human description for the CSV
#   category     : "pharma" | "device" | "logistics"
#
# SCMD covers medicines only. Device/logistics suppliers are expected to yield
# zero product-level matches — that is recorded honestly, not forced.
#
# Fresenius Kabi and B. Braun are the only pharma/IV manufacturers here; their
# proprietary brand names (Kabiven, Clinimix, etc.) DO appear in SCMD VMPs
# because they are the only licensed products for those formulations.
#
# L&R / UrgoKTwo products appear in SCMD wound-care lines (device_or_nutrition
# dosage form) under the Urgo brand — L&R and Urgo merged 2023; "urgo" search
# is the correct heuristic.

SUPPLIER_CONFIG = [
    {
        "key": "fresenius kabi",
        "patterns": [
            r"fresenius",
            r"\bkabiven\b",
            r"\bclinimix\b",
            r"\bclinoleic\b",
            r"\bsmoflipid\b",
            r"\bstructokabiven\b",
            r"\bcalea\b",
        ],
        "match_method": "brand/manufacturer keyword search (fresenius, kabiven, "
                        "clinimix, clinoleic, smoflipid, structokabiven, calea)",
        "category": "pharma",
    },
    {
        "key": "b. braun",
        "patterns": [
            r"\bbraun\b",
            r"b\.braun",
            r"\bprontosan\b",
            r"\batrauman\b",
        ],
        "match_method": "manufacturer/brand keyword search (braun, prontosan, atrauman)",
        "category": "pharma",
    },
    {
        "key": "becton dickinson",
        "patterns": [r"\bbd\b", r"becton"],
        "match_method": "manufacturer keyword search — device supplier, "
                        "expected 0 pharma matches in SCMD",
        "category": "device",
    },
    {
        "key": "johnson & johnson",
        "patterns": [r"johnson", r"\bj&j\b", r"\bjj\b"],
        "match_method": "manufacturer keyword search — MedTech supplier, "
                        "expected 0 pharma matches in SCMD",
        "category": "device",
    },
    {
        "key": "lohmann",
        "patterns": [
            r"lohmann",
            r"\bl&r\b",
            r"\burgo\b",
            r"vliwasoft",
            r"suprasorb",
        ],
        "match_method": "manufacturer/brand keyword search (lohmann, l&r, urgo — "
                        "L&R and Urgo merged 2023; wound-care products in SCMD "
                        "device_or_nutrition category)",
        "category": "device",
    },
    {
        "key": "dhl",
        "patterns": [],
        "match_method": "N/A — logistics/distribution supplier; no pharmaceutical "
                        "product names in SCMD attributable to a logistics company",
        "category": "logistics",
    },
    {
        "key": "smith and nephew",
        "patterns": [r"smith.*nephew", r"\ballevyn\b", r"\bacticoat\b", r"\bopsite\b"],
        "match_method": "brand keyword search — MedTech/wound-care supplier, "
                        "expected 0 pharma matches in SCMD",
        "category": "device",
    },
    {
        "key": "siemens healthcare diagnostics",
        "patterns": [r"\bsiemens\b"],
        "match_method": "manufacturer keyword search — diagnostics/lab-equipment "
                        "supplier; expected 0 pharma matches; intensity ratio used "
                        "as supplier-tier feature",
        "category": "device",
    },
    {
        "key": "siemens healthcare limited",
        "patterns": [r"\bsiemens\b"],
        "match_method": "manufacturer keyword search — imaging/MRI supplier; "
                        "expected 0 pharma matches; intensity ratio used as "
                        "supplier-tier feature",
        "category": "device",
    },
]


def _extract_first_float(text) -> float | None:
    """Pull the first numeric value out of a free-text string."""
    if pd.isna(text):
        return None
    m = re.search(r"[\d,]+\.?\d*", str(text).replace(",", ""))
    return float(m.group().replace(",", "")) if m else None


def _extract_intensity(text) -> float | None:
    """
    Extract the primary (first-quoted) tCO2e/£m intensity from the
    carbon_intensity_tco2e_per_gbpm field, which is free text.
    Returns None when:
      - field is NaN or unparseable
      - value is explicitly flagged as global (not UK-specific)
      - value is denominated in non-GBP currency (e.g. $m)
    """
    if pd.isna(text):
        return None
    s = str(text)
    # Exclude global figures and non-GBP denominators — not usable as UK label
    if re.search(r"\bglobal\b", s, re.IGNORECASE):
        return None
    if "$" in s:
        return None
    m = re.search(r"([\d]+\.?\d*)", s)
    return float(m.group(1)) if m else None


def match_supplier(supplier_name: str, product_names: pd.Series,
                   config_entry: dict) -> set[str]:
    """Return set of VMP_PRODUCT_NAME strings matching any pattern for this supplier."""
    if not config_entry["patterns"]:
        return set()
    combined = re.compile(
        "|".join(config_entry["patterns"]), re.IGNORECASE
    )
    matched = product_names[product_names.str.contains(combined, na=False)]
    return set(matched.tolist())


def main() -> None:
    os.makedirs(INT_DIR, exist_ok=True)

    # ── 1. Load and filter CRP CSV ──────────────────────────────────────────
    crp = pd.read_csv(CRP_PATH, encoding="utf-8-sig")
    usable_mask = crp["usable_as_direct_label"].str.startswith(
        ("Yes", "Partial"), na=False
    )
    usable = crp[usable_mask].reset_index(drop=True)

    print(f"Total CRP rows:   {len(crp)}")
    print(f"Usable (Yes/Partial): {len(usable)}")
    print()

    # ── 2. Load SCMD product names ──────────────────────────────────────────
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    unique_vmps = (
        features.drop_duplicates(subset="VMP_SNOMED_CODE")
        [["VMP_SNOMED_CODE", "VMP_PRODUCT_NAME"]]
        .reset_index(drop=True)
    )
    product_names = unique_vmps["VMP_PRODUCT_NAME"]
    n_vmps = len(unique_vmps)
    print(f"Unique SCMD VMPs available for matching: {n_vmps:,}\n")

    # ── 3. Match each usable supplier ───────────────────────────────────────
    rows = []
    # Map: VMP_PRODUCT_NAME -> (carbon_intensity_kgco2e_per_gbp, supplier_name)
    # Only pharma suppliers with product matches contribute to this.
    product_intensity_map: dict[str, tuple[float, str]] = {}

    for _, crp_row in usable.iterrows():
        sname = crp_row["supplier_name"]
        sname_lower = sname.lower()

        # Find matching config entry
        cfg = None
        for entry in SUPPLIER_CONFIG:
            if entry["key"] in sname_lower:
                cfg = entry
                break

        if cfg is None:
            print(f"  WARNING: no config entry for supplier '{sname}' — skipping")
            continue

        # Run product-name search
        matched_names = match_supplier(sname, product_names, cfg)
        n_matched = len(matched_names)

        # Extract intensity ratio (tCO2e/£m) from free-text column
        raw_intensity = crp_row.get("carbon_intensity_tco2e_per_gbpm")
        intensity_tco2e_per_gbpm = _extract_intensity(raw_intensity)

        # Convert tCO2e/£m → kgCO2e/£:
        #   1 tCO2e = 1000 kgCO2e; 1 £m = 1,000,000 £
        #   → kgCO2e/£ = tCO2e/£m × 1000 / 1,000,000 = tCO2e/£m × 0.001
        intensity_kgco2e_per_gbp = (
            intensity_tco2e_per_gbpm * 0.001
            if intensity_tco2e_per_gbpm is not None
            else None
        )

        # Extract total_tco2e (first number in field)
        total_tco2e = _extract_first_float(crp_row.get("total_tco2e"))

        # Notes: category + special flags
        notes_parts = [f"category={cfg['category']}"]
        if cfg["category"] == "device":
            notes_parts.append("SCMD covers medicines — device/diagnostic product "
                                "names not expected in VMP list")
        if cfg["category"] == "logistics":
            notes_parts.append("logistics supplier — no pharmaceutical VMPs attributable")
        if intensity_kgco2e_per_gbp is not None:
            notes_parts.append(
                f"intensity ratio used as supplier-tier feature: "
                f"{intensity_tco2e_per_gbpm} tCO2e/£m "
                f"= {intensity_kgco2e_per_gbp:.4f} kgCO2e/£"
            )

        rows.append({
            "supplier_name":                   sname,
            "n_scmd_products_matched":         n_matched,
            "match_method":                    cfg["match_method"],
            "carbon_intensity_tco2e_per_gbpm": intensity_tco2e_per_gbpm,
            "intensity_kgco2e_per_gbp":        intensity_kgco2e_per_gbp,
            "total_tco2e":                     total_tco2e,
            "usable_as_direct_label":          crp_row["usable_as_direct_label"][:60],
            "notes":                           "; ".join(notes_parts),
        })

        # Register matched product names for the supplier_crp_matched flag.
        # supplier_carbon_intensity is only populated when a valid UK-specific
        # intensity ratio is also available for that supplier.
        if n_matched > 0:
            for pname in matched_names:
                if pname not in product_intensity_map:
                    # Store (intensity_or_None, supplier_name)
                    product_intensity_map[pname] = (intensity_kgco2e_per_gbp, sname)

    matches_df = pd.DataFrame(rows)

    # ── Print matching summary table ─────────────────────────────────────────
    print("=" * 72)
    print("SUPPLIER CRP MATCHING SUMMARY")
    print("=" * 72)

    display_cols = [
        "supplier_name",
        "n_scmd_products_matched",
        "carbon_intensity_tco2e_per_gbpm",
        "intensity_kgco2e_per_gbp",
        "total_tco2e",
    ]
    # Truncate supplier name for display
    display = matches_df[display_cols].copy()
    display["supplier_name"] = display["supplier_name"].str[:45]
    print(display.to_string(index=False))
    print()

    # ── Save interim matches CSV ─────────────────────────────────────────────
    save_cols = [
        "supplier_name", "n_scmd_products_matched", "match_method",
        "carbon_intensity_tco2e_per_gbpm", "intensity_kgco2e_per_gbp",
        "total_tco2e", "notes",
    ]
    matches_df[save_cols].to_csv(MATCHES_PATH, index=False)
    print(f"Saved: {MATCHES_PATH}")
    print()

    # ── 3. Add columns to pharma features ────────────────────────────────────
    print("Adding supplier_carbon_intensity and supplier_crp_matched columns...")

    # Build VMP-level intensity lookup from matched products.
    # supplier_crp_matched = 1 for any product in product_intensity_map,
    # regardless of whether an intensity value is available.
    # supplier_carbon_intensity is null when the supplier has no intensity ratio.
    vmp_intensity = (
        unique_vmps["VMP_PRODUCT_NAME"]
        .map(lambda n: product_intensity_map.get(n, (None, None))[0])
    )
    vmp_matched = (
        unique_vmps["VMP_PRODUCT_NAME"]
        .map(lambda n: 1 if n in product_intensity_map else 0)
    )

    # Build lookup: VMP_SNOMED_CODE → (intensity, matched_flag)
    intensity_map = dict(zip(
        unique_vmps["VMP_SNOMED_CODE"], vmp_intensity
    ))
    matched_map = dict(zip(
        unique_vmps["VMP_SNOMED_CODE"], vmp_matched
    ))

    features["supplier_carbon_intensity"] = features["VMP_SNOMED_CODE"].map(intensity_map)
    features["supplier_crp_matched"]      = features["VMP_SNOMED_CODE"].map(matched_map).fillna(0).astype(int)

    # ── 4. Save full features ────────────────────────────────────────────────
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}  ({len(features):,} rows, {len(features.columns)} columns)")
    print()

    # ── 5. Print coverage summary ────────────────────────────────────────────
    n_matched_rows  = int(features["supplier_crp_matched"].sum())
    n_total_rows    = len(features)
    total_spend     = features["INDICATIVE_COST"].sum()
    matched_spend   = features.loc[
        features["supplier_crp_matched"] == 1, "INDICATIVE_COST"
    ].sum()

    print("=" * 72)
    print("COVERAGE SUMMARY")
    print("=" * 72)
    print(f"  SCMD rows with supplier_crp_matched = 1:  "
          f"{n_matched_rows:>8,}  of {n_total_rows:,}  "
          f"({100 * n_matched_rows / n_total_rows:.2f}%)")
    print(f"  Total SCMD spend:                         "
          f"£{total_spend:>14,.2f}")
    print(f"  Matched supplier spend:                   "
          f"£{matched_spend:>14,.2f}  "
          f"({100 * matched_spend / total_spend:.2f}% of total)")
    print()

    # Additional detail: which suppliers actually contributed matches
    matched_suppliers = [
        r for r in rows if r["n_scmd_products_matched"] > 0
    ]
    if matched_suppliers:
        print("  Suppliers with >=1 SCMD product match:")
        for r in matched_suppliers:
            print(f"    {r['supplier_name'][:50]:<50}  "
                  f"{r['n_scmd_products_matched']:>4} products")
    else:
        print("  No suppliers matched any SCMD product names.")
    print()
    print("  Note: Siemens intensity ratios (30 and 213.6 tCO2e/£m) are recorded")
    print("  in supplier_crp_matches.csv as supplier-tier features even though")
    print("  no SCMD pharmaceutical VMPs were matched to Siemens product lines.")


if __name__ == "__main__":
    main()
