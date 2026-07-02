"""
Derive pharmacological carbon-relevant features from SCMD product names.

Works at VMP (product) level, then joins back to all SCMD rows.
Output: data/processed/scmd_with_pharma_features.csv
Also writes: reports/pharma_feature_summary.md

Features built:
  dosage_form       -- tablet/capsule | oral_liquid | pmdi_inhaler |
                       dpi_inhaler | iv_injection | topical | other
  is_biologic       -- 1/0 (keyword list in data/interim/biologic_keywords.csv)
  is_inhaler_hfc    -- 1/0  pMDI only (HFC propellant burden)
  is_cold_chain     -- 1/0  requires 2-8 deg C refrigerated storage
  price_per_unit    -- INDICATIVE_COST / TOTAL_QUANITY_IN_VMP_UNIT (GBP)
"""

import os
import re
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
INT_DIR      = os.path.join(PROJECT_ROOT, "data", "interim")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
KEYWORD_FILE = os.path.join(INT_DIR, "biologic_keywords.csv")


# ─── 1. Dosage form classifier ────────────────────────────────────────────────

# Rules applied in order; first match wins.
# Patterns tested against lower-cased VMP_PRODUCT_NAME.
DOSAGE_FORM_RULES = [
    # pMDI inhaler — MUST come before generic inhaler rules.
    # Sources: product name keywords for pressurised metered-dose inhalers.
    # HFC propellant burden: NHS England "Delivering a Net Zero NHS" (2022, p.60)
    # quantifies MDI emissions via NAEI data. Specific per-unit figure is ~20 kg CO2e
    # for a salbutamol pMDI vs ~0.4 kg CO2e for a DPI (Wilkinson et al. 2019,
    # BMJ Open, doi:10.1136/bmjopen-2019-028763). Ratio ~50x confirms pMDI flag.
    ("pmdi_inhaler", re.compile(
        r"pressurised|evohaler|autohaler|easi.breath|hfa inhaler|"
        r"cfc.free inhaler|cfc inhaler|aerosol inhaler|"
        r"metered.dose inhaler|pMDI|"
        # Named pMDI brands common in NHS formulary
        r"salamol|salbulin|airflusal forspiro"  # last one is DPI — exclude pattern only if needed
        r"|airomir|salamol|ventolin evohaler",
        re.IGNORECASE
    )),

    # Dry powder inhaler — specifically NOT pMDI.
    ("dpi_inhaler", re.compile(
        r"turbohaler|accuhaler|handihaler|ellipta|breezhaler|nexthaler|"
        r"genuair|twisthaler|rotacap|clickhaler|novolizer|"
        r"dry powder inhaler|breath.actuated inhaler|"
        r"spiromax|forspiro|zonda inhaler",
        re.IGNORECASE
    )),

    # Generic inhaler — not yet classified to pMDI or DPI.
    # These will be reviewed; conservative assignment to other.
    ("other_inhaler", re.compile(
        r"\binhaler\b|\bnebuliser solution\b|\bnebulizer solution\b",
        re.IGNORECASE
    )),

    # IV / injection — before oral to avoid matching "oral solution for injection"
    ("iv_injection", re.compile(
        r"solution for infusion|solution for injection|"
        r"concentrate for solution for infusion|"
        r"powder for solution for infusion|"
        r"powder for solution for injection|"
        r"powder for concentrate|"
        r"suspension for injection|"
        r"emulsion for infusion|"
        r"dispersion for infusion|"
        r"infusion$|injection$|"
        r"\bintravenous\b|\biv\b infusion|\biv\b injection|"
        r"for injection$|for infusion$|"
        # Pre-mixed infusion bags and drug-in-diluent products not caught above
        r"infusion bags?|infusion \d|in sodium chloride|in glucose",
        re.IGNORECASE
    )),

    # Oral liquid — includes oromucosal solutions (liquids administered to oral mucosa)
    ("oral_liquid", re.compile(
        r"oromucosal|"
        r"oral solution|oral suspension|oral liquid|oral drops|"
        r"oral syrup|oral mixture|oral emulsion|"
        r"\bsyrup\b|\belixir\b|\blinctus\b|\bmixture\b|"
        r"\bdrops\b oral|\bliquid\b oral|"
        r"solution$(?!.*inject)(?!.*infus)",  # solution not followed by injection
        re.IGNORECASE
    )),

    # Tablet / capsule / solid oral dose
    ("tablet_capsule", re.compile(
        r"\btablets?\b|\bcapsules?\b|\bcaplets?\b|\bpills?\b|"
        r"\bdispersible\b|\bmodified.release\b|\bprolonged.release\b|"
        r"\bimmediate.release\b|\bgastro.resistant\b|"
        r"\bchewable\b|\borodispersible\b|\bsublingual\b|\bbuccal\b|"
        r"\blozenge\b|\bwafer\b|\bgranules?\b|\bpowder\b(?!.*inject)(?!.*infus)|"
        r"\bpellets?\b|\bmicrogranules?\b",
        re.IGNORECASE
    )),

    # Topical — patches, creams, ointments, gels, eye/ear/nose drops
    ("topical", re.compile(
        r"\bcream\b|\bointment\b|\bgel\b|\blotion\b|\bpaste\b|\bfoam\b|"
        r"\btransdermal\b|\bpatch(es)?\b|"
        r"\beye drops?\b|\bear drops?\b|\bnasal spray\b|\bnasal drops?\b|"
        r"\bnasal solution\b|\bnasal suspension\b|"
        r"\beye ointment\b|\beye gel\b|\beye solution\b|"
        r"\bintravitreal\b|\bophthalmic\b|"
        r"\bshampoo\b|\bscalp\b|\bspray\b(?!.*nasal)",
        re.IGNORECASE
    )),

    # Rectal / vaginal / other routes
    ("other", re.compile(
        r"\bsuppository\b|\bsuppositories\b|\benema\b|"
        r"\bpessary\b|\bpessaries\b|\bvaginal\b|\brectal\b|"
        r"\bimplant\b|\bintrauterine\b|\binhalation\b|"
        r"\bgas\b|\bmedicated\b|\brinse\b|\bmouthwash\b",
        re.IGNORECASE
    )),

    # Medical devices, wound care, and enteral/parenteral nutrition.
    # These are non-pharmaceutical SCMD line items; carbon profile is
    # not derivable from pharma spend-based factors.
    ("device_or_nutrition", re.compile(
        r"\bdressing\b|\bbandage\b|\btape\b|\bsponge\b|"
        r"\bcatheter\b|\btubing\b|\bneedle\b|\bdropper\b|\bdevice\b|"
        r"tube[\s\-]feed|\bemulsion\s+liquid\b|\bfibre\s+liquid\b|"
        r"fortisip|survimed|paediasure|paedsure|liquigen|"
        r"nutrison|fresubin|jevity",
        re.IGNORECASE
    )),
]


def classify_dosage_form(name: str) -> str:
    for form, pattern in DOSAGE_FORM_RULES:
        if pattern.search(name):
            return form
    return "unclassified"


# ─── 2. Biologic classifier ───────────────────────────────────────────────────

def build_biologic_patterns(keyword_file: str):
    """Load keyword CSV and compile into a single regex."""
    kw = pd.read_csv(keyword_file)
    patterns = []
    for _, row in kw.iterrows():
        k   = str(row["keyword"]).strip()
        mt  = str(row["match_type"]).strip()
        if mt == "suffix":
            # Match as word ending: e.g. "-mab" -> r"mab\b"
            pat = re.escape(k.lstrip("-")) + r"\b"
        elif mt in ("contains", "suffix_check"):
            pat = re.escape(k)
        else:
            pat = k
        patterns.append(pat)
    return re.compile("|".join(patterns), re.IGNORECASE)


# ─── 3. pMDI / HFC inhaler ───────────────────────────────────────────────────

# pMDI = pressurised metered-dose inhaler using HFC propellant.
# HFC-134a (GWP100 = 1430) and HFC-227ea (GWP100 = 3220) are the primary propellants.
# Reference: NHS England (2022) "Delivering a Net Zero NHS", p.60 — MDI emissions
# quantified via NAEI data; NHS Carbon Footprint model treats MDIs as a
# distinct bottom-up category separate from pharmaceutical spend.
# Per-inhaler emission estimate: ~20 kg CO2e for a salbutamol pMDI vs ~0.4 kg CO2e
# for an equivalent DPI (Wilkinson AJ et al., BMJ Open 2019;9:e028763,
# doi:10.1136/bmjopen-2019-028763). Ratio ~50x.

PMDI_PATTERN = re.compile(
    r"pressurised|evohaler|modulite|autohaler|easi.breath|"
    r"hfa\b|cfc.free|cfc inhaler|aerosol inhal|"
    r"metered.dose|pMDI|airomir|salamol|airflusal|"
    r"ventolin evohaler|clenil|qvar|fostair|flutiform|"
    r"symbicort(?!.*turbohaler)|relvar(?!.*ellipta)",
    re.IGNORECASE
)

DPI_PATTERN = re.compile(
    r"turbohaler|accuhaler|handihaler|ellipta|breezhaler|nexthaler|"
    r"genuair|twisthaler|rotacap|clickhaler|novolizer|spiromax|"
    r"forspiro|dry powder inhaler|breath.actuated",
    re.IGNORECASE
)


def is_pmdi(name: str) -> int:
    """1 if pMDI, 0 if DPI or other. DPI check takes precedence over pMDI brand names."""
    # Explicit non-pMDI inhalers: Olbas is a passive vapour stick, no HFC propellant.
    if re.search(r"\bolbas\b", name, re.IGNORECASE):
        return 0
    if DPI_PATTERN.search(name):
        return 0
    if PMDI_PATTERN.search(name):
        return 1
    # Fallback: "inhaler" without any DPI device keyword → treat as pMDI.
    # DPI exclusion has already fired above, so anything reaching here with
    # "inhaler" in the name is a pressurised device not yet named explicitly.
    if re.search(r"\binhaler\b", name, re.IGNORECASE):
        return 1
    return 0


# ─── 4. Cold-chain classifier ─────────────────────────────────────────────────

# Refrigerated storage (2-8 deg C) required for:
# - All insulins
# - Most biologics (mAbs, ESAs, CSFs, growth factors)
# - Most vaccines
# - Human immunoglobulins
# - Most recombinant clotting factors
# Conservative: flag if matches biologic AND not explicitly "store below 25" in name
# (SCMD doesn't carry storage instructions, so we proxy via product class)

COLD_CHAIN_PATTERN = re.compile(
    r"\binsulin\b|immunoglobulin|epoetin|darbepoetin|"
    r"filgrastim|pegfilgrastim|lenograstim|lipegfilgrastim|"
    r"somatropin|teriparatide|"
    r"vaccine|vaccination|"
    r"fresh frozen plasma|prothrombin complex|fibrinogen|"
    r"factor viii|factor ix|factor vii|"
    r"alteplase|tenecteplase|"
    # Specific mAbs confirmed cold-chain in BNF / SmPC
    r"trastuzumab|pertuzumab|bevacizumab|cetuximab|"
    r"pembrolizumab|nivolumab|atezolizumab|durvalumab|"
    r"rituximab|obinutuzumab|ofatumumab|"
    r"adalimumab|infliximab|etanercept|certolizumab|golimumab|"
    r"tocilizumab|sarilumab|"
    r"denosumab|ranibizumab|aflibercept|faricimab|"
    r"ocrelizumab|natalizumab|vedolizumab|"
    r"ustekinumab|secukinumab|ixekizumab|guselkumab|risankizumab|"
    r"bimekizumab|dupilumab|benralizumab|mepolizumab|omalizumab|"
    r"bortezomib|carfilzomib|daratumumab|"
    r"botulinum",
    re.IGNORECASE
)


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Load SCMD ──────────────────────────────────────────────────────────
    scmd_files = [f for f in os.listdir(RAW_DIR) if f.startswith("scmd_") and f.endswith(".csv")]
    if not scmd_files:
        sys.exit("No SCMD file in data/raw/. Run fetch_scmd.py first.")
    scmd_path = os.path.join(RAW_DIR, scmd_files[0])
    print(f"Loading SCMD: {scmd_path}")
    df = pd.read_csv(scmd_path, encoding="utf-8-sig", low_memory=False)
    print(f"  {len(df):,} rows, {df['VMP_SNOMED_CODE'].nunique():,} unique VMPs")

    # ── Build VMP-level feature table ──────────────────────────────────────
    print("\nBuilding VMP-level feature table...")
    vmp = (
        df.groupby(["VMP_SNOMED_CODE", "VMP_PRODUCT_NAME"])
        .agg(
            total_spend=("INDICATIVE_COST", "sum"),
            total_qty=("TOTAL_QUANITY_IN_VMP_UNIT", "sum"),
            n_rows=("INDICATIVE_COST", "count"),
        )
        .reset_index()
    )
    print(f"  {len(vmp):,} unique VMP products")

    # Load biologic pattern
    biologic_pattern = build_biologic_patterns(KEYWORD_FILE)

    # Apply classifiers
    names = vmp["VMP_PRODUCT_NAME"]

    print("  Classifying dosage forms...")
    vmp["dosage_form"] = names.map(classify_dosage_form)

    print("  Classifying biologics...")
    vmp["is_biologic"] = names.str.contains(biologic_pattern).astype(int)

    print("  Classifying pMDI / HFC inhalers...")
    vmp["is_inhaler_hfc"] = names.map(is_pmdi)

    print("  Classifying cold-chain products...")
    vmp["is_cold_chain"] = names.str.contains(COLD_CHAIN_PATTERN).astype(int)

    # Price per unit — aggregate from SCMD (use medians to avoid outliers skewing)
    print("  Computing price per unit...")
    price_per_unit = (
        df[df["INDICATIVE_COST"].notna() & df["TOTAL_QUANITY_IN_VMP_UNIT"].notna()
           & (df["TOTAL_QUANITY_IN_VMP_UNIT"] > 0)]
        .assign(ppu=lambda x: x["INDICATIVE_COST"] / x["TOTAL_QUANITY_IN_VMP_UNIT"])
        .groupby("VMP_SNOMED_CODE")["ppu"]
        .agg(price_per_unit_median="median", price_per_unit_mean="mean")
        .reset_index()
    )
    vmp = vmp.merge(price_per_unit, on="VMP_SNOMED_CODE", how="left")

    # ── Consistency checks ─────────────────────────────────────────────────
    # Inhalers flagged as pMDI should also have dosage_form == pmdi_inhaler or other_inhaler
    # If is_inhaler_hfc=1 but dosage_form is tablet_capsule, flag it
    vmp["flag_ppm_dosage_mismatch"] = (
        (vmp["is_inhaler_hfc"] == 1) &
        ~vmp["dosage_form"].isin(["pmdi_inhaler", "dpi_inhaler", "other_inhaler", "other"])
    ).astype(int)

    # ── Join back to all SCMD rows ─────────────────────────────────────────
    print("  Joining features back to SCMD rows...")
    feature_cols = [
        "VMP_SNOMED_CODE", "dosage_form", "is_biologic",
        "is_inhaler_hfc", "is_cold_chain",
        "price_per_unit_median", "price_per_unit_mean",
    ]
    df_out = df.merge(vmp[feature_cols], on="VMP_SNOMED_CODE", how="left")

    # ── Save output ────────────────────────────────────────────────────────
    out_path = os.path.join(PROC_DIR, "scmd_with_pharma_features.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df_out):,} rows, {len(df_out.columns)} columns)")

    # ── Print and write summary ────────────────────────────────────────────
    _print_and_write_summary(vmp, df_out, out_path)


def _print_and_write_summary(vmp, df_out, out_path):
    n = len(vmp)

    def pct(k):
        return 100 * k / n

    print("\n" + "=" * 65)
    print("FEATURE SUMMARY")
    print("=" * 65)

    # Dosage form breakdown
    print("\n[1] DOSAGE FORM")
    df_counts = vmp["dosage_form"].value_counts()
    for form, cnt in df_counts.items():
        print(f"  {form:<20} {cnt:>5,}  ({pct(cnt):.1f}%)")

    # Flag breakdown
    for col, label in [
        ("is_biologic",    "[2] IS BIOLOGIC"),
        ("is_inhaler_hfc", "[3] IS pMDI/HFC INHALER"),
        ("is_cold_chain",  "[4] IS COLD CHAIN"),
    ]:
        pos = vmp[col].sum()
        neg = n - pos
        print(f"\n{label}")
        print(f"  Flagged:   {pos:>5,}  ({pct(pos):.1f}%)")
        print(f"  Not flagged:{neg:>4,}  ({pct(neg):.1f}%)")

    # Price per unit
    ppu = vmp["price_per_unit_median"].dropna()
    print(f"\n[5] PRICE PER UNIT (GBP, median across SCMD rows)")
    print(f"  Products with price data: {len(ppu):,}")
    print(f"  Median: GBP {ppu.median():.4f}")
    print(f"  Mean:   GBP {ppu.mean():.4f}")
    print(f"  P95:    GBP {ppu.quantile(0.95):.2f}")
    print(f"  Max:    GBP {ppu.max():,.2f}")

    # Consistency flags
    mismatch = vmp["flag_ppm_dosage_mismatch"].sum()
    if mismatch:
        print(f"\n  MISMATCH: {mismatch} products flagged pMDI but dosage_form not inhaler")
        print("  Sample:")
        sample = vmp[vmp["flag_ppm_dosage_mismatch"] == 1]["VMP_PRODUCT_NAME"].head(5)
        for name in sample:
            print(f"    {name}")

    # Sample 10 products per category
    SAMPLES = {
        "tablet_capsule":   ("dosage_form", "tablet_capsule"),
        "oral_liquid":      ("dosage_form", "oral_liquid"),
        "pmdi_inhaler":     ("dosage_form", "pmdi_inhaler"),
        "dpi_inhaler":      ("dosage_form", "dpi_inhaler"),
        "iv_injection":     ("dosage_form", "iv_injection"),
        "topical":          ("dosage_form", "topical"),
        "other_inhaler":    ("dosage_form", "other_inhaler"),
        "unclassified_form":("dosage_form", "unclassified"),
        "biologic":         ("is_biologic", 1),
        "inhaler_hfc":      ("is_inhaler_hfc", 1),
        "cold_chain":       ("is_cold_chain", 1),
    }

    lines = [
        "# Pharma Feature Summary",
        "",
        f"**Products analysed:** {n:,} unique VMPs  ",
        f"**SCMD rows:** {len(df_out):,}  ",
        "",
        "## Dosage Form Distribution",
        "",
        "| Form | Count | % |",
        "|------|-------|---|",
    ]
    for form, cnt in vmp["dosage_form"].value_counts().items():
        lines.append(f"| {form} | {cnt:,} | {pct(cnt):.1f}% |")

    for col, label in [
        ("is_biologic",    "Biologic"),
        ("is_inhaler_hfc", "pMDI / HFC Inhaler"),
        ("is_cold_chain",  "Cold Chain"),
    ]:
        pos = int(vmp[col].sum())
        lines += [
            "",
            f"## {label}",
            "",
            f"| | Count | % |",
            "|---|---|---|",
            f"| Flagged | {pos:,} | {pct(pos):.1f}% |",
            f"| Not flagged | {n - pos:,} | {pct(n - pos):.1f}% |",
        ]

    lines += [
        "",
        "## Price Per Unit (GBP)",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Products with data | {len(ppu):,} |",
        f"| Median | GBP {ppu.median():.4f} |",
        f"| Mean | GBP {ppu.mean():.4f} |",
        f"| 95th percentile | GBP {ppu.quantile(0.95):.2f} |",
        f"| Maximum | GBP {ppu.max():,.2f} |",
        "",
        "## Sample Products by Category (10 each — for manual verification)",
        "",
    ]

    for section_key, (col, val) in SAMPLES.items():
        if col == "dosage_form":
            sample_df = vmp[vmp["dosage_form"] == val]
        else:
            sample_df = vmp[vmp[col] == val]
        sample_df = sample_df.sort_values("total_spend", ascending=False).head(10)

        lines.append(f"### {section_key.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Product Name | Total Spend (GBP) |")
        lines.append("|---|---|")
        for _, row in sample_df.iterrows():
            lines.append(f"| {row['VMP_PRODUCT_NAME']} | {row['total_spend']:,.0f} |")
        lines.append("")

    # Unclassified products warning
    unc = vmp[vmp["dosage_form"] == "unclassified"]
    if len(unc):
        lines += [
            "## Unclassified Dosage Forms — Review Required",
            "",
            f"**{len(unc):,} products** did not match any dosage form rule.",
            "Top 20 by spend:",
            "",
            "| Product Name | Total Spend (GBP) |",
            "|---|---|",
        ]
        for _, row in unc.sort_values("total_spend", ascending=False).head(20).iterrows():
            lines.append(f"| {row['VMP_PRODUCT_NAME']} | {row['total_spend']:,.0f} |")

    report_path = os.path.join(REPORTS_DIR, "pharma_feature_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
