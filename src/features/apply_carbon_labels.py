"""
Apply DEFRA EEIO baseline carbon labels to the SCMD dataset.

Joins SCMD with the BNF-chapter emission factor mapping, classifies each
medicine by product name, and computes two carbon estimates per row:

  estimated_kgco2e              -- production-basis: INDICATIVE_COST * 0.17
                                   Source: ONS Atmospheric Emissions, SIC 21, 2024
  estimated_kgco2e_consumption  -- consumption-basis: INDICATIVE_COST * 0.2402
                                   Source: DEFRA/BEIS UK Carbon Footprint 2022, CPA 21
                                   Primary anchor for all downstream analysis.
                                   See data/interim/exiobase_consumption_basis.md.

Saves: data/interim/scmd_with_defra_baseline.csv
"""

import os
import re
import sys
import pandas as pd

# Add project root to path so we can import the classifier
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
INT_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

MAPPING_PATH = os.path.join(INT_DIR, "bnf_to_defra_mapping.csv")

# Consumption-basis anchor: DEFRA/BEIS UK Carbon Footprint 2022, CPA 21.
# "Basic pharmaceutical products and pharmaceutical preparations"
# Source: Conversion_factors_kgCO2_per_£_spent_by_SIC_code.ods, published May 2025.
# Full rationale in data/interim/exiobase_consumption_basis.md.
DEFRA_CONSUMPTION_KGCO2E_PER_GBP = 0.240242


def find_scmd_file():
    for f in os.listdir(RAW_DIR):
        if f.lower().startswith("scmd_") and f.lower().endswith(".csv"):
            return os.path.join(RAW_DIR, f)
    raise FileNotFoundError("No SCMD CSV in data/raw/. Run fetch_scmd.py first.")


# ── Inline BNF classifier (mirrors build_defra_mapping.py) ──────────────────

BNF_CHAPTERS = [
    (1, "Gastro-intestinal",
     r"antacid|omeprazol|lansoprazol|pantoprazol|ranitidine|metoclopram|domperidon|"
     r"loperamide|mesalazine|infliximab|adalimumab.*crohn|azathioprine|"
     r"lactulose|senna|bisacodyl|macrogol|linaclotide|prucalopride|"
     r"pancreatin|ursodeoxycholic|ondansetron|granisetron|aprepitant|"
     r"laxative|antidiarrhoeal|antispasmodic|hyoscine butylbromide|"
     r"sucralfate|colestyramine|obeticholic"),

    (2, "Cardiovascular",
     r"aspirin.*card|clopidogrel|warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|"
     r"atorvastatin|rosuvastatin|simvastatin|pravastatin|ezetimibe|"
     r"amlodipine|nifedipine|diltiazem|verapamil|"
     r"bisoprolol|carvedilol|metoprolol|atenolol|"
     r"lisinopril|ramipril|perindopril|enalapril|candesartan|losartan|"
     r"furosemide|bumetanide|spironolactone|eplerenone|"
     r"digoxin|amiodarone|flecainide|"
     r"nicorandil|isosorbide|glyceryl trinitrate|"
     r"sacubitril|ivabradine|dapagliflozin.*heart|"
     r"antihypertensive|antiarrhythmic|anticoagulant|antiplatelet|statin"),

    (3, "Respiratory",
     r"salbutamol|salmeterol|formoterol|indacaterol|olodaterol|"
     r"budesonide|fluticasone|beclometasone|ciclesonide|mometasone.*inhal|"
     r"tiotropium|ipratropium|glycopyrronium|aclidinium|umeclidinium|"
     r"montelukast|zafirlukast|theophylline|aminophylline|"
     r"roflumilast|benralizumab|mepolizumab|dupilumab|"
     r"carbocisteine|erdosteine|dornase alfa|"
     r"inhaler|nebuliser solution|"
     r"bronchodilator|corticosteroid.*inhal"),

    (4, "Central Nervous System",
     r"sertraline|fluoxetine|citalopram|escitalopram|venlafaxine|duloxetine|"
     r"mirtazapine|amitriptyline|nortriptyline|clomipramine|"
     r"olanzapine|quetiapine|risperidone|aripiprazole|clozapine|"
     r"diazepam|lorazepam|clonazepam|midazolam|temazepam|"
     r"zopiclone|zolpidem|melatonin|"
     r"levodopa|co-careldopa|pramipexole|ropinirole|rasagiline|selegiline|"
     r"donepezil|rivastigmine|galantamine|memantine|"
     r"gabapentin|pregabalin|lamotrigine|levetiracetam|valproate|"
     r"carbamazepine|phenytoin|topiramate|"
     r"methylphenidate|atomoxetine|lisdexamfetamine|"
     r"sumatriptan|rizatriptan|zolmitriptan|"
     r"naltrexone|buprenorphine.*opioid|methadone|"
     r"antidepressant|antipsychotic|anxiolytic|sedative|hypnotic|anticonvulsant|"
     r"antiparkinsonian"),

    (5, "Infections",
     r"amoxicillin|co-amoxiclav|ampicillin|flucloxacillin|piperacillin|"
     r"cefalexin|cefuroxime|ceftriaxone|cefotaxime|ceftazidime|cefepime|"
     r"clarithromycin|azithromycin|erythromycin|"
     r"ciprofloxacin|levofloxacin|moxifloxacin|"
     r"trimethoprim|co-trimoxazole|nitrofurantoin|"
     r"metronidazole|tinidazole|"
     r"doxycycline|tetracycline|"
     r"vancomycin|teicoplanin|linezolid|daptomycin|"
     r"meropenem|ertapenem|imipenem|"
     r"fluconazole|itraconazole|voriconazole|posaconazole|caspofungin|"
     r"aciclovir|valaciclovir|ganciclovir|valganciclovir|"
     r"oseltamivir|remdesivir|"
     r"isoniazid|rifampicin|ethambutol|pyrazinamide|"
     r"antibiotic|antifungal|antiviral|antimicrobial|bactericidal"),

    (6, "Endocrine",
     r"insulin|metformin|glipizide|gliclazide|glibenclamide|"
     r"sitagliptin|vildagliptin|saxagliptin|alogliptin|"
     r"dapagliflozin|empagliflozin|canagliflozin|"
     r"liraglutide|semaglutide|dulaglutide|exenatide|"
     r"levothyroxine|liothyronine|propylthiouracil|carbimazole|"
     r"prednisolone|dexamethasone|hydrocortisone|fludrocortisone|"
     r"growth hormone|somatropin|"
     r"testosterone|anastrozole|letrozole|"
     r"antidiabetic|hypoglycaemic|thyroid|corticosteroid(?!.*inhal)"),

    (7, "Obstetrics, Gynaecology & Urinary",
     r"oxybutynin|solifenacin|tolterodine|mirabegron|"
     r"tamsulosin|alfuzosin|silodosin|doxazosin.*bph|finasteride|dutasteride|"
     r"oestrogen|estrogen|progesterone|norethisterone|"
     r"mifepristone|misoprostol|"
     r"sildenafil|tadalafil|vardenafil|avanafil|"
     r"oxytocin|ergometrine|atosiban|"
     r"clomifene|gonadotrophin|"
     r"contraceptive|intrauterine"),

    (8, "Malignant Disease & Immunosuppression",
     r"methotrexate|cyclophosphamide|chlorambucil|melphalan|"
     r"fluorouracil|capecitabine|gemcitabine|cytarabine|"
     r"doxorubicin|epirubicin|idarubicin|"
     r"paclitaxel|docetaxel|nab-paclitaxel|cabazitaxel|"
     r"carboplatin|cisplatin|oxaliplatin|"
     r"imatinib|dasatinib|nilotinib|bosutinib|ponatinib|"
     r"erlotinib|gefitinib|afatinib|osimertinib|"
     r"trastuzumab|pertuzumab|cetuximab|bevacizumab|"
     r"pembrolizumab|nivolumab|atezolizumab|durvalumab|"
     r"rituximab|obinutuzumab|ofatumumab|"
     r"bortezomib|carfilzomib|ixazomib|daratumumab|"
     r"lenalidomide|thalidomide|pomalidomide|"
     r"letrozole|anastrozole|exemestane|tamoxifen|fulvestrant|"
     r"leuprorelin|goserelin|triptorelin|"
     r"mycophenolate|tacrolimus|ciclosporin|sirolimus|"
     r"antineoplastic|chemotherapy|cytotoxic|immunosuppressant|"
     r"anticancer|monoclonal antibody|targeted therapy"),

    (9, "Nutrition & Blood",
     r"ferrous|ferric|iron.*inj|"
     r"vitamin b12|hydroxocobalamin|cyanocobalamin|folic acid|"
     r"epoetin|darbepoetin|erythropoietin|"
     r"filgrastim|pegfilgrastim|lenograstim|"
     r"albumin|"
     r"parenteral nutrition|total parenteral|"
     r"potassium chloride|sodium chloride.*inj|glucose.*inj|hartmann|"
     r"magnesium sulfate.*inj|calcium gluconate.*inj|"
     r"factor viii|factor ix|factor vii|"
     r"fresh frozen plasma|prothrombin|"
     r"alteplase|tenecteplase|streptokinase|"
     r"heparin|enoxaparin|dalteparin|fondaparinux|"
     r"antifibrinolytic|tranexamic|"
     r"anaemia|haemostatic|antianaemic"),

    (10, "Musculoskeletal & Joint Diseases",
     r"ibuprofen|naproxen|diclofenac|celecoxib|etoricoxib|"
     r"paracetamol(?!.*codeine).*oral|"
     r"methotrexate.*rheuma|sulfasalazine|hydroxychloroquine|"
     r"adalimumab|etanercept|certolizumab|golimumab|"
     r"tocilizumab|sarilumab|baricitinib|tofacitinib|upadacitinib|"
     r"abatacept|"
     r"alendronic|risedronate|zoledronic|ibandronic|"
     r"denosumab|teriparatide|romosozumab|"
     r"colchicine|allopurinol|febuxostat|"
     r"nsaid|dmard|disease modifying|antirheumatic"),

    (11, "Eye",
     r"latanoprost|bimatoprost|travoprost|timolol.*eye|"
     r"brimonidine|brinzolamide|dorzolamide|"
     r"ranibizumab|bevacizumab.*eye|aflibercept|faricimab|"
     r"dexamethasone.*eye|prednisolone.*eye|"
     r"chloramphenicol.*eye|fusidic acid.*eye|"
     r"eye drop|eye ointment|intravitreal"),

    (12, "Ear, Nose & Oropharynx",
     r"fluticasone.*nasal|mometasone.*nasal|beclometasone.*nasal|"
     r"azelastine.*nasal|ipratropium.*nasal|"
     r"ear drop|nasal spray|"
     r"nystatin.*oral|miconazole.*oral gel"),

    (13, "Skin",
     r"betamethasone.*cream|clobetasol|hydrocortisone.*cream|"
     r"tacrolimus.*skin|pimecrolimus|"
     r"aciclovir.*cream|fusidic acid.*cream|mupirocin|"
     r"calcipotriol|dithranol|coal tar|"
     r"isotretinoin|adapalene|benzoyl peroxide|"
     r"emollient|aqueous cream|"
     r"topical.*antibiotic|topical.*antifungal|"
     r"clotrimazole|miconazole.*skin|terbinafine.*skin"),

    (14, "Immunological Products & Vaccines",
     r"vaccine|vaccination|immunisation|"
     r"immunoglobulin|antitetanus|antitoxin|"
     r"infliximab|adalimumab(?!.*crohn)(?!.*rheuma)|"
     r"omalizumab|dupilumab(?!.*asthma)|"
     r"normal immunoglobulin|subcutaneous immunoglobulin|iv immunoglobulin"),

    (15, "Anaesthesia",
     r"propofol|thiopental|ketamine|etomidate|"
     r"suxamethonium|rocuronium|vecuronium|atracurium|mivacurium|"
     r"neostigmine|sugammadex|"
     r"fentanyl|alfentanil|remifentanil|sufentanil|"
     r"morphine|oxycodone|hydromorphone|"
     r"bupivacaine|levobupivacaine|ropivacaine|lidocaine|"
     r"sevoflurane|isoflurane|desflurane|nitrous oxide|"
     r"dexmedetomidine|clonidine.*anaes|"
     r"ondansetron.*anaes|cyclizine|dexamethasone.*anaes|"
     r"anaesthetic|analgesic.*inj|opioid.*inj|neuromuscular block"),

    (0, "Other / Unclassified", r""),
]


def classify_bnf(name: str) -> tuple[int, str]:
    name_lc = str(name).lower()
    for ch_num, ch_name, pattern in BNF_CHAPTERS:
        if ch_num == 0:
            return ch_num, ch_name
        if pattern and re.search(pattern, name_lc):
            return ch_num, ch_name
    return 0, "Other / Unclassified"


def main():
    os.makedirs(INT_DIR, exist_ok=True)

    # Load SCMD
    scmd_path = find_scmd_file()
    print(f"Loading SCMD: {scmd_path}")
    df = pd.read_csv(scmd_path, encoding="utf-8-sig", low_memory=False)
    print(f"  {len(df):,} rows loaded.")

    # Load mapping
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(
            "Mapping not found. Run src/features/build_defra_mapping.py first."
        )
    mapping = pd.read_csv(MAPPING_PATH)
    factor_lookup = dict(
        zip(mapping["bnf_chapter_num"], mapping["defra_kgco2e_per_gbp"])
    )
    conf_lookup = dict(
        zip(mapping["bnf_chapter_num"], mapping["mapping_confidence"])
    )
    name_lookup = dict(
        zip(mapping["bnf_chapter_num"], mapping["bnf_chapter_name"])
    )

    # Classify BNF chapter from product name
    print("Classifying BNF chapters from product names…")
    classifications = df["VMP_PRODUCT_NAME"].map(classify_bnf)
    df["bnf_chapter_num"] = [c[0] for c in classifications]
    df["bnf_chapter_name"] = [c[1] for c in classifications]

    # Join emission factors
    df["defra_kgco2e_per_gbp"] = df["bnf_chapter_num"].map(factor_lookup)
    df["mapping_confidence"] = df["bnf_chapter_num"].map(conf_lookup)

    # Production-basis carbon estimate (ONS SIC 21, 0.17 kgCO2e/GBP)
    df["estimated_kgco2e"] = df["INDICATIVE_COST"] * df["defra_kgco2e_per_gbp"]

    # Consumption-basis carbon estimate (DEFRA 2022 CPA 21, 0.2402 kgCO2e/GBP)
    # Applied uniformly — the DEFRA factor is a sector-level average for all
    # pharmaceutical products; no chapter-level differentiation is available.
    df["consumption_kgco2e_per_gbp"] = DEFRA_CONSUMPTION_KGCO2E_PER_GBP
    df["estimated_kgco2e_consumption"] = (
        df["INDICATIVE_COST"] * DEFRA_CONSUMPTION_KGCO2E_PER_GBP
    )

    # Data quality flags
    df["dq_zero_cost"] = df["INDICATIVE_COST"] == 0
    df["dq_null_cost"] = df["INDICATIVE_COST"].isnull()
    df["dq_unclassified"] = df["bnf_chapter_num"] == 0

    # Save
    out_path = os.path.join(INT_DIR, "scmd_with_defra_baseline.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(df):,}    Columns: {len(df.columns)}")

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n=== CARBON BASELINE SUMMARY ===")
    total_co2_prod = df["estimated_kgco2e"].sum()
    total_co2_cons = df["estimated_kgco2e_consumption"].sum()
    total_co2 = total_co2_prod   # keep reference for chapter breakdown
    total_spend = df["INDICATIVE_COST"].sum()
    print(f"  Total indicative spend:                    £{total_spend:>15,.0f}")
    print(f"  Production-basis total (0.17 kgCO2e/£):    {total_co2_prod:>15,.0f} kgCO2e")
    print(f"  Consumption-basis total (0.24 kgCO2e/£):   {total_co2_cons:>15,.0f} kgCO2e")
    print(f"  Production-basis intensity check:           {total_co2_prod/total_spend:.4f} kgCO2e/£")
    print(f"  Consumption-basis intensity check:          {total_co2_cons/total_spend:.4f} kgCO2e/£")

    print("\n=== DATA QUALITY ===")
    print(f"  Rows with £0 cost:              {df['dq_zero_cost'].sum():>8,} ({100*df['dq_zero_cost'].mean():.1f}%)")
    print(f"  Rows with null cost:            {df['dq_null_cost'].sum():>8,} ({100*df['dq_null_cost'].mean():.1f}%)")
    print(f"  Rows unclassified (Ch 0):       {df['dq_unclassified'].sum():>8,} ({100*df['dq_unclassified'].mean():.1f}%)")

    print("\n=== MAPPING CONFIDENCE DISTRIBUTION ===")
    conf_counts = df.groupby("mapping_confidence").agg(
        rows=("estimated_kgco2e", "count"),
        total_kgco2e=("estimated_kgco2e", "sum"),
        total_spend=("INDICATIVE_COST", "sum"),
    ).reset_index()
    for _, row in conf_counts.iterrows():
        spend_pct = 100 * row["total_spend"] / total_spend
        co2_pct = 100 * row["total_kgco2e"] / total_co2
        print(f"  {row['mapping_confidence']:<8}  {row['rows']:>8,} rows  "
              f"  spend: {spend_pct:.1f}%  carbon: {co2_pct:.1f}%")

    print("\n=== TOP 10 BNF CHAPTERS BY ESTIMATED CARBON (tCO2e) ===")
    by_chapter = (
        df.groupby(["bnf_chapter_num", "bnf_chapter_name", "mapping_confidence"])
        .agg(
            total_spend=("INDICATIVE_COST", "sum"),
            total_kgco2e=("estimated_kgco2e", "sum"),
            n_rows=("estimated_kgco2e", "count"),
        )
        .reset_index()
        .sort_values("total_kgco2e", ascending=False)
        .head(10)
    )
    print(f"  {'Chapter':<45} {'Conf':<7} {'tCO2e':>10} {'£M spend':>10} {'Rows':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for _, row in by_chapter.iterrows():
        flag = " [LOW]" if row["mapping_confidence"] == "low" else ""
        print(f"  {row['bnf_chapter_name']:<45} {row['mapping_confidence']:<7} "
              f"{row['total_kgco2e']/1000:>10,.1f} "
              f"{row['total_spend']/1e6:>10,.1f} "
              f"{row['n_rows']:>8,}{flag}")

    print("\n=== UNCLASSIFIED PRODUCTS (SAMPLE — up to 20) ===")
    unclass = df[df["dq_unclassified"]]["VMP_PRODUCT_NAME"].value_counts().head(20)
    for name, count in unclass.items():
        print(f"  {count:>6,}x  {name}")
    print("  (Review these — they may need additional keyword patterns or dm+d lookup)")


if __name__ == "__main__":
    main()
