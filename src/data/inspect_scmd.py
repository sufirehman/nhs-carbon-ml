"""
Inspect and document the NHSBSA SCMD dataset.

Outputs a markdown summary to reports/scmd_data_summary.md.

NOTE: SCMD does not include BNF chapter/section/paragraph columns.
It uses VMP_SNOMED_CODE + VMP_PRODUCT_NAME. BNF classification is
derived in src/features/build_defra_mapping.py using product-name
pattern matching. See that script for the full methodology note.
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")

SCMD_GLOB_PATTERN = "scmd_"


def find_scmd_file():
    for f in os.listdir(RAW_DIR):
        if f.lower().startswith("scmd_") and f.lower().endswith(".csv"):
            return os.path.join(RAW_DIR, f)
    raise FileNotFoundError(
        "No SCMD CSV found in data/raw/. Run src/data/fetch_scmd.py first."
    )


def main():
    scmd_path = find_scmd_file()
    print(f"Loading: {scmd_path}")
    df = pd.read_csv(scmd_path, encoding="utf-8-sig", low_memory=False)

    print("\n=== COLUMNS & DTYPES ===")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<40} {dtype}")

    print(f"\n=== SHAPE ===")
    print(f"  Rows: {len(df):,}    Columns: {len(df.columns)}")

    print("\n=== DATE RANGE ===")
    months = sorted(df["YEAR_MONTH"].dropna().unique())
    print(f"  Months present: {months}")

    print("\n=== NULL COUNTS ===")
    nulls = df.isnull().sum()
    for col, n in nulls[nulls > 0].items():
        pct = 100 * n / len(df)
        print(f"  {col:<40} {n:>8,}  ({pct:.1f}%)")
    if nulls.sum() == 0:
        print("  None — dataset is complete.")

    print("\n=== UNIQUE COUNTS ===")
    print(f"  NHS organisations (ODS_CODE):   {df['ODS_CODE'].nunique():>6,}")
    print(f"  VMP products (SNOMED code):     {df['VMP_SNOMED_CODE'].nunique():>6,}")
    print(f"  VMP product names:              {df['VMP_PRODUCT_NAME'].nunique():>6,}")
    print(f"  Units of measure:               {df['UNIT_OF_MEASURE_NAME'].nunique():>6,}")

    print("\n  NOTE: SCMD has NO BNF chapter/section/paragraph columns.")
    print("  Classification is derived from product names in build_defra_mapping.py.")

    print("\n=== COST SUMMARY (INDICATIVE_COST, £) ===")
    cost = df["INDICATIVE_COST"].dropna()
    print(f"  Total spend:  £{cost.sum():>15,.0f}")
    print(f"  Mean per row: £{cost.mean():>10,.2f}")
    print(f"  Median:       £{cost.median():>10,.2f}")
    print(f"  Max:          £{cost.max():>10,.2f}")
    print(f"  Rows with £0:  {(cost == 0).sum():,}")

    print("\n=== TOP 20 PRODUCTS BY TOTAL INDICATIVE COST ===")
    top_products = (
        df.groupby("VMP_PRODUCT_NAME")["INDICATIVE_COST"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    for name, val in top_products.items():
        print(f"  £{val:>12,.0f}  {name}")

    print("\n=== TOP 20 NHS ORGANISATIONS BY TOTAL SPEND ===")
    top_orgs = (
        df.groupby("ODS_CODE")["INDICATIVE_COST"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    for ods, val in top_orgs.items():
        print(f"  {ods:<10}  £{val:>12,.0f}")

    print("\n=== QUANTITY DISTRIBUTION ===")
    qty = df["TOTAL_QUANITY_IN_VMP_UNIT"].dropna()
    print(f"  Min: {qty.min():.1f}    Max: {qty.max():,.0f}    Mean: {qty.mean():,.1f}")

    # ── Markdown report ──────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "scmd_data_summary.md")

    lines = [
        "# SCMD Data Summary",
        "",
        f"**Source file:** `{os.path.basename(scmd_path)}`  ",
        f"**Rows:** {len(df):,}  ",
        f"**Columns:** {len(df.columns)}  ",
        f"**Date range:** {months[0]} to {months[-1]}  ",
        "",
        "## Columns",
        "",
        "| Column | dtype | Non-null count |",
        "|--------|-------|---------------|",
    ]
    for col in df.columns:
        nn = df[col].count()
        lines.append(f"| `{col}` | {df[col].dtype} | {nn:,} |")

    lines += [
        "",
        "## Key Counts",
        "",
        f"- **NHS organisations (ODS_CODE):** {df['ODS_CODE'].nunique():,}",
        f"- **Unique VMP products (SNOMED):** {df['VMP_SNOMED_CODE'].nunique():,}",
        f"- **Total indicative spend:** £{cost.sum():,.0f}",
        "",
        "## Important: Missing BNF Classification",
        "",
        "The SCMD dataset does **not** include BNF chapter, section, or paragraph",
        "columns. The dataset uses SNOMED VMP codes for product identification.",
        "",
        "BNF-like classification is derived in `src/features/build_defra_mapping.py`",
        "using product-name pattern matching against BNF chapter keyword lists.",
        "This is an approximation; a more accurate mapping would use the NHS dm+d",
        "(Dictionary of Medicines and Devices) SNOMED-to-BNF lookup, available from",
        "NHS TRUD (requires free registration at https://isd.digital.nhs.uk/trud).",
        "",
        "## Top 20 Products by Total Indicative Cost",
        "",
        "| Product | Total £ |",
        "|---------|---------|",
    ]
    for name, val in top_products.items():
        lines.append(f"| {name} | £{val:,.0f} |")

    lines += [
        "",
        "## Top 20 NHS Organisations by Spend",
        "",
        "| ODS Code | Total £ |",
        "|----------|---------|",
    ]
    for ods, val in top_orgs.items():
        lines.append(f"| {ods} | £{val:,.0f} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
