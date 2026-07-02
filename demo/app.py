"""
NHS Carbon ML Dashboard

Five-page Streamlit research preview:
  1. Overview             -- dataset summary and spend breakdowns
  2. Carbon Explorer      -- model predictions and what-if tool
  3. SHAP Intelligence    -- feature importance
  4. Supplier Transparency -- CRP analysis and INN naming coverage
  5. Methodology          -- circular label problem and model evaluation

Run from the project root:
  streamlit run demo/app.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# ── Path resolution ────────────────────────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURES_PATH = os.path.join(ROOT, "data", "processed", "scmd_with_all_features.csv")
BASELINE_PATH = os.path.join(ROOT, "data", "interim",   "scmd_with_defra_baseline.csv")
DEMO_SAMPLE_PATH = os.path.join(ROOT, "data", "demo",     "scmd_demo_sample.csv")
DEMO_PRED_PATH   = os.path.join(ROOT, "data", "demo",     "vmp_predictions.csv")
SHAP_PATH     = os.path.join(ROOT, "reports",            "shap_findings.csv")
RESULTS_PATH  = os.path.join(ROOT, "reports",            "model_results_paper.csv")
CRP_PATH      = os.path.join(ROOT, "data", "raw",        "supplier_crps_manual.csv")
METHOD_PATH   = os.path.join(ROOT, "reports",            "methodology_note.md")
FIG_SHAP_BAR  = os.path.join(ROOT, "reports", "figures", "shap_top10_bar.png")
FIG_SCATTER   = os.path.join(ROOT, "reports", "figures", "predicted_vs_actual.png")


# ── NHS colour scheme ──────────────────────────────────────────────────────────

TEAL  = "#00A499"
NAVY  = "#003087"
GREY  = "#425563"
LTEAL = "#AED9D5"


# ── Page configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NHS Carbon ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    [data-testid="stMetricValue"] {{ color:{NAVY}; font-weight:700; }}
    [data-testid="stMetricLabel"] {{ color:{GREY}; font-size:0.82rem; }}
    h1 {{ color:{NAVY}; border-bottom:3px solid {TEAL};
          padding-bottom:0.4rem; margin-bottom:1rem; }}
    h2, h3 {{ color:{NAVY}; }}
</style>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

def show_footer() -> None:
    st.markdown(
        '<hr style="margin-top:40px;border:none;border-top:0.5px solid #E2E8F0">'
        '<p style="text-align:center;font-size:12px;color:#718096;margin-top:8px">'
        'Built by <a href="https://github.com/sufirehman" style="color:#00A499;'
        'text-decoration:none">Sufiyan Ul Rehman</a>'
        " — AI Researcher and Lecturer,"
        " Ulster University / Solent University (QAHE)"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading SCMD data...")
def load_scmd() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        # Cloud deployment: full processed data isn't committed (100+ MB).
        # Fall back to the pre-computed stratified demo sample instead.
        return pd.read_csv(DEMO_SAMPLE_PATH, low_memory=False)

    af = pd.read_csv(FEATURES_PATH, low_memory=False)
    if os.path.exists(BASELINE_PATH):
        db = pd.read_csv(
            BASELINE_PATH, low_memory=False,
            usecols=["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE",
                     "bnf_chapter_num", "bnf_chapter_name",
                     "estimated_kgco2e_consumption"],
        )
        af = af.merge(db, on=["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE"], how="left")
    else:
        af["bnf_chapter_num"]  = 0
        af["bnf_chapter_name"] = "Unknown"
        af["estimated_kgco2e_consumption"] = af["INDICATIVE_COST"].fillna(0) * 0.2402
    return af


@st.cache_resource(show_spinner="Training lightweight demo model...")
def load_demo_model() -> dict:
    """Small RandomForest trained on the ~5,000-row demo sample, used only
    for the What-If tool's live re-prediction. Keeps the app free of the
    259 MB research model for Streamlit Cloud deployment."""
    from sklearn.ensemble import RandomForestRegressor

    sample = pd.read_csv(DEMO_SAMPLE_PATH, low_memory=False)
    sample = sample[sample["estimated_kgco2e_consumption"] > 0].reset_index(drop=True)

    y_log = np.log1p(sample["estimated_kgco2e_consumption"].values.astype(float))
    X, feature_names = build_features_train(sample)

    model = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1,
    )
    model.fit(X, y_log)

    return {
        "model": model,
        "feature_names": feature_names,
        "apply_log": True,
        "model_name": "Lightweight Random Forest (demo, n=50 trees)",
    }


@st.cache_data
def load_shap() -> pd.DataFrame:
    return pd.read_csv(SHAP_PATH)


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_crp() -> pd.DataFrame:
    return pd.read_csv(CRP_PATH, encoding="utf-8-sig")


# ── Feature matrix builder (mirrors train_and_evaluate.py exactly) ─────────────

# Fixed dosage-form categories so one-hot columns are always consistent,
# regardless of which forms happen to appear in a given (sub-)sample.
ALL_FORMS = [
    "device_or_nutrition", "dpi_inhaler", "iv_injection", "oral_liquid",
    "other", "other_inhaler", "tablet_capsule", "topical", "unclassified",
]


def build_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    form = pd.Categorical(df["dosage_form"], categories=ALL_FORMS)
    dummies = pd.get_dummies(form, prefix="form", dtype=int)
    parts   = [dummies]

    for col in ["is_biologic", "is_inhaler_hfc", "is_cold_chain", "supplier_crp_matched"]:
        if col in df.columns:
            val = df[col].fillna(0).astype(int)
        else:
            val = pd.Series(0, index=df.index, name=col)
        parts.append(val.rename(col).to_frame())

    ppu = (
        df["price_per_unit_median"].copy()
        if "price_per_unit_median" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    ppu = ppu.fillna(ppu.median() if ppu.notna().any() else 1.0)
    parts.append(np.log1p(ppu).rename("log_price_per_unit").to_frame())

    bnf = (
        df["bnf_chapter_num"].fillna(0).astype(int)
        if "bnf_chapter_num" in df.columns
        else pd.Series(0, index=df.index)
    )
    parts.append(bnf.rename("bnf_chapter_num").to_frame())

    X = pd.concat(parts, axis=1)
    return X.reindex(columns=feature_names, fill_value=0)


def build_features_train(df: pd.DataFrame) -> tuple:
    """Same construction as build_features, but derives feature_names from
    the data itself (used to train the lightweight demo model)."""
    feature_names = (
        [f"form_{f}" for f in ALL_FORMS]
        + ["is_biologic", "is_inhaler_hfc", "is_cold_chain", "supplier_crp_matched",
           "log_price_per_unit", "bnf_chapter_num"]
    )
    return build_features(df, feature_names), feature_names


# ── Pre-computed VMP-level predictions (cached across page navigations) ────────

@st.cache_data(show_spinner="Loading product predictions...")
def vmp_predictions() -> pd.DataFrame:
    return pd.read_csv(DEMO_PRED_PATH)


# ── SHAP display labels ────────────────────────────────────────────────────────

SHAP_LABELS = {
    "log_price_per_unit":       "Log price per unit",
    "form_iv_injection":        "IV injection form",
    "bnf_chapter_num":          "BNF chapter",
    "form_tablet_capsule":      "Tablet / capsule form",
    "form_topical":             "Topical form",
    "is_biologic":              "Biologic flag",
    "form_oral_liquid":         "Oral liquid form",
    "is_cold_chain":            "Cold-chain flag",
    "form_unclassified":        "Unclassified form",
    "form_other":               "Other form",
    "form_device_or_nutrition": "Device / nutrition form",
    "form_dpi_inhaler":         "DPI inhaler form",
    "form_other_inhaler":       "Other inhaler form",
    "is_inhaler_hfc":           "HFC inhaler flag",
    "supplier_crp_matched":     "Supplier CRP matched",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="background:{NAVY};color:white;padding:1rem 1.2rem;'
        f'border-radius:6px;text-align:center;margin-bottom:1.2rem;">'
        f'<div style="font-size:1.05rem;font-weight:700;letter-spacing:0.02em;">'
        f'NHS Carbon ML</div>'
        f'<div style="font-size:0.72rem;opacity:0.75;margin-top:3px;">'
        f'Research Dashboard</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Page",
        ["Overview", "Carbon Explorer", "SHAP Intelligence",
         "Supplier Transparency", "Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: NHSBSA SCMD, Feb 2025")
    st.caption("Model: Random Forest (Optuna-tuned)")
    st.caption("Log-space R² = 0.76  |  n = 313,375")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Overview")
    st.markdown(
        "NHS secondary-care medicine procurement, February 2025 (NHSBSA SCMD). "
        "Carbon estimates use the DEFRA 2023 consumption-basis factor (0.2402 kgCO2e/£) "
        "applied to indicative spend."
    )

    df = load_scmd()

    total_spend    = df["INDICATIVE_COST"].sum()
    carbon_consump = total_spend * 0.2402
    carbon_prod    = total_spend * 0.17

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dispensing rows",           f"{len(df):,}")
    m2.metric("Unique products (VMPs)",    f"{df['VMP_SNOMED_CODE'].nunique():,}")
    m3.metric("Prescribing sites",         f"{df['ODS_CODE'].nunique():,}")
    m4.metric("Total indicative spend",    f"£{total_spend / 1e9:.2f}bn")
    m5.metric("Estimated carbon (0.2402)", f"{carbon_consump / 1e3:,.0f} tCO2e")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Spend by dosage form")
        form_spend = (
            df.groupby("dosage_form")["INDICATIVE_COST"]
            .sum().sort_values().reset_index()
        )
        form_spend.columns = ["Dosage form", "Spend (£)"]
        fig = px.bar(
            form_spend, x="Spend (£)", y="Dosage form",
            orientation="h", color_discrete_sequence=[TEAL],
        )
        fig.update_layout(
            plot_bgcolor="white", font_color=GREY, height=340,
            margin=dict(l=0, r=30, t=10, b=20),
            xaxis=dict(gridcolor="#ebebeb"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Carbon flag counts (dispensing rows)")
        flags = pd.DataFrame({
            "Category": ["HFC propellant inhalers", "Biological medicines",
                          "Cold-chain products",     "Supplier CRP matched"],
            "Rows":     [int(df["is_inhaler_hfc"].sum()),
                         int(df["is_biologic"].sum()),
                         int(df["is_cold_chain"].sum()),
                         int(df["supplier_crp_matched"].sum())],
        }).sort_values("Rows")
        fig2 = px.bar(
            flags, x="Rows", y="Category",
            orientation="h", color_discrete_sequence=[NAVY],
        )
        fig2.update_layout(
            plot_bgcolor="white", font_color=GREY, height=340,
            margin=dict(l=0, r=30, t=10, b=20),
            xaxis=dict(gridcolor="#ebebeb"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Carbon estimates under three baseline approaches")
    baseline_rows = [
        ("Production basis",    "ONS / DEFRA 0.17 kgCO2e/£",  total_spend * 0.17),
        ("Consumption basis",   "DEFRA 0.2402 kgCO2e/£",      total_spend * 0.2402),
        ("EXIOBASE upper bound","chemicals-nec 0.39 kgCO2e/£", total_spend * 0.39),
    ]
    bdf = pd.DataFrame(baseline_rows, columns=["Basis", "Source", "kgCO2e"])
    bdf["tCO2e"] = (bdf["kgCO2e"] / 1e3).map("{:,.0f}".format)
    bdf = bdf.drop(columns="kgCO2e")
    st.dataframe(bdf, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Top 20 products by indicative spend")
    top20 = (
        df.groupby(["VMP_PRODUCT_NAME", "dosage_form"])["INDICATIVE_COST"]
        .sum().sort_values(ascending=False).head(20).reset_index()
    )
    top20.columns = ["Product", "Dosage form", "Spend (£)"]
    top20["Spend (£)"] = top20["Spend (£)"].map("£{:,.0f}".format)
    st.dataframe(top20, use_container_width=True, hide_index=True)
    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CARBON EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Carbon Explorer":
    st.title("Carbon Explorer")
    st.markdown(
        "Filter NHS medicines and see the trained Random Forest's predicted carbon "
        "intensity per unit. Use the What-If tool to modify a product's attributes "
        "and observe how the prediction changes."
    )

    df   = load_scmd()
    vmps = vmp_predictions()

    all_forms = sorted(df["dosage_form"].dropna().unique().tolist())

    with st.sidebar:
        st.subheader("Filters")
        sel_forms = st.multiselect(
            "Dosage form", all_forms, default=all_forms, key="cf_forms"
        )
        if "bnf_chapter_num" in vmps.columns:
            bnf_opts = sorted(
                vmps["bnf_chapter_num"].dropna().astype(int).unique().tolist()
            )
            sel_bnf = st.multiselect(
                "BNF chapter", bnf_opts, default=bnf_opts, key="cf_bnf"
            )
        else:
            sel_bnf = None
        sel_bio  = st.checkbox("Biologics only",    value=False, key="cf_bio")
        sel_cold = st.checkbox("Cold chain only",   value=False, key="cf_cold")
        sel_hfc  = st.checkbox("HFC inhalers only", value=False, key="cf_hfc")

    mask = vmps["dosage_form"].isin(sel_forms)
    if sel_bnf is not None:
        mask &= vmps["bnf_chapter_num"].isin(sel_bnf)
    if sel_bio:  mask &= vmps["is_biologic"]    == 1
    if sel_cold: mask &= vmps["is_cold_chain"]  == 1
    if sel_hfc:  mask &= vmps["is_inhaler_hfc"] == 1
    fv = vmps[mask].copy()

    st.info(
        f"{len(fv):,} unique products match current filters "
        f"(from {len(vmps):,} total VMPs in SCMD)"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Highest predicted carbon per unit")
        top_t = fv.nlargest(15, "predicted_kgco2e_per_unit")[
            ["VMP_PRODUCT_NAME", "dosage_form",
             "predicted_kgco2e_per_unit", "total_spend"]
        ].copy()
        top_t.columns = ["Product", "Form", "Pred. kgCO2e/unit", "Spend (£)"]
        top_t["Pred. kgCO2e/unit"] = top_t["Pred. kgCO2e/unit"].map("{:,.1f}".format)
        top_t["Spend (£)"]    = top_t["Spend (£)"].map("£{:,.0f}".format)
        st.dataframe(top_t, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Price per unit vs predicted carbon")
        sc = fv[fv["price_per_unit_median"].notna()].copy()
        sc["log_price"] = np.log1p(sc["price_per_unit_median"])
        sample = sc.sample(min(2000, len(sc)), random_state=42)
        fig = px.scatter(
            sample,
            x="log_price",
            y="predicted_kgco2e_per_unit",
            color="dosage_form",
            hover_name="VMP_PRODUCT_NAME",
            hover_data={"log_price": ":.2f", "dosage_form": False},
            labels={
                "log_price":                 "Log price per unit (£)",
                "predicted_kgco2e_per_unit": "Predicted kgCO2e / unit",
                "dosage_form":               "Form",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.65))
        fig.update_layout(
            plot_bgcolor="white", font_color=GREY, height=410,
            margin=dict(l=0, r=0, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, font_size=10),
            xaxis=dict(gridcolor="#ebebeb"),
            yaxis=dict(gridcolor="#ebebeb"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # -- What-If tool --
    st.divider()
    st.subheader("What-If: modify product attributes")
    st.caption(
        "Select a product from the top 200 by spend. Change its attributes "
        "and see how a lightweight demo model's carbon prediction responds. "
        "This tool uses a small RandomForest (50 trees) trained on a 5,000-row "
        "sample for fast, interactive re-prediction — not the full research "
        "model shown in the tables and charts above."
    )

    pkg = load_demo_model()
    top200 = vmps.nlargest(200, "total_spend")["VMP_PRODUCT_NAME"].tolist()
    selected = st.selectbox("Product", top200, key="wi_product")
    sel_row  = vmps[vmps["VMP_PRODUCT_NAME"] == selected].iloc[0]

    wi_c1, wi_c2 = st.columns(2)

    with wi_c1:
        wi_form = st.selectbox(
            "Dosage form", all_forms,
            index=all_forms.index(sel_row["dosage_form"])
                  if sel_row["dosage_form"] in all_forms else 0,
            key="wi_form",
        )
        wi_bio  = st.checkbox("Biologic",    value=bool(sel_row.get("is_biologic", 0)),   key="wi_bio")
        wi_cold = st.checkbox("Cold chain",  value=bool(sel_row.get("is_cold_chain", 0)), key="wi_cold")
        wi_hfc  = st.checkbox("HFC inhaler", value=bool(sel_row.get("is_inhaler_hfc", 0)), key="wi_hfc")
        default_price = (
            float(sel_row["price_per_unit_median"])
            if "price_per_unit_median" in sel_row and pd.notna(sel_row["price_per_unit_median"])
            else 1.0
        )
        wi_price = st.number_input(
            "Price per unit (£)", min_value=0.01,
            value=default_price, step=0.50, format="%.2f", key="wi_price",
        )

    wi_row = pd.DataFrame([{
        "dosage_form":           wi_form,
        "is_biologic":           int(wi_bio),
        "is_inhaler_hfc":        int(wi_hfc),
        "is_cold_chain":         int(wi_cold),
        "supplier_crp_matched":  int(sel_row.get("supplier_crp_matched", 0)),
        "price_per_unit_median": wi_price,
        "bnf_chapter_num":       (
            int(sel_row["bnf_chapter_num"])
            if "bnf_chapter_num" in sel_row.index and pd.notna(sel_row["bnf_chapter_num"])
            else 0
        ),
    }])

    orig_row = pd.DataFrame([{
        "dosage_form":           sel_row["dosage_form"],
        "is_biologic":           int(sel_row.get("is_biologic", 0)),
        "is_inhaler_hfc":        int(sel_row.get("is_inhaler_hfc", 0)),
        "is_cold_chain":         int(sel_row.get("is_cold_chain", 0)),
        "supplier_crp_matched":  int(sel_row.get("supplier_crp_matched", 0)),
        "price_per_unit_median": default_price,
        "bnf_chapter_num":       (
            int(sel_row["bnf_chapter_num"])
            if "bnf_chapter_num" in sel_row.index and pd.notna(sel_row["bnf_chapter_num"])
            else 0
        ),
    }])

    # Both sides of the comparison use the same demo model, so an unchanged
    # selection always shows a zero delta.
    X_orig   = build_features(orig_row, pkg["feature_names"])
    orig_log = pkg["model"].predict(X_orig)[0]
    orig     = np.expm1(orig_log) if pkg["apply_log"] else orig_log

    X_wi    = build_features(wi_row, pkg["feature_names"])
    wi_log  = pkg["model"].predict(X_wi)[0]
    wi_pred = np.expm1(wi_log) if pkg["apply_log"] else wi_log
    delta   = wi_pred - orig
    d_pct   = 100.0 * delta / orig if orig > 0 else 0.0

    with wi_c2:
        st.markdown("**Prediction comparison (demo model)**")
        st.metric("Original prediction", f"{orig:,.1f} kgCO2e / unit")
        st.metric(
            "Modified prediction",
            f"{wi_pred:,.1f} kgCO2e / unit",
            delta=f"{delta:+,.1f} ({d_pct:+.1f}%)",
            delta_color="inverse",
        )
        if abs(d_pct) > 1.0:
            direction = "higher" if delta > 0 else "lower"
            st.info(
                f"Changing these attributes shifts the predicted carbon "
                f"{abs(d_pct):.0f}% {direction}."
            )
        else:
            st.info("Attributes are unchanged from the original product.")
        st.caption(
            f"For reference, the full research model (table above) predicts "
            f"{float(sel_row['predicted_kgco2e_per_unit']):,.1f} kgCO2e/unit "
            f"for this product's original attributes."
        )
    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "SHAP Intelligence":
    st.title("SHAP Intelligence")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) values measure how much each feature "
        "shifts the model's prediction away from the mean, averaged across the "
        "held-out test set (n = 59,717 records). Values are in log-space carbon units."
    )

    shap_df = load_shap()
    shap_df["label"] = shap_df["feature"].map(SHAP_LABELS).fillna(shap_df["feature"])

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("All 15 features")
        colours = [NAVY if i < 3 else TEAL for i in range(len(shap_df))]
        fig = go.Figure(go.Bar(
            x=shap_df["mean_abs_shap"][::-1].values,
            y=shap_df["label"][::-1].values,
            orientation="h",
            marker_color=colours[::-1],
            text=[f"{v:.3f}" for v in shap_df["mean_abs_shap"][::-1]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig.update_layout(
            xaxis_title="Mean |SHAP value| (log-space carbon)",
            plot_bgcolor="white", font_color=GREY, height=510,
            margin=dict(l=0, r=70, t=10, b=30),
            xaxis=dict(gridcolor="#ebebeb"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Navy = top 3 features by SHAP.  Teal = features 4-15.")

    with col2:
        st.subheader("Feature interpretations")
        for _, row in shap_df.iterrows():
            with st.expander(f"{row['label']}  ({row['mean_abs_shap']:.3f})"):
                st.write(row["interpretation"])

    st.divider()
    st.subheader("Key findings")
    st.markdown(
        "**Log price per unit** (SHAP = 1.234) is 3.6 times larger than the next "
        "feature (IV injection form: 0.347). Unit price is the strongest proxy for "
        "embedded carbon intensity: expensive medicines require complex manufacturing "
        "and longer cold chains. IV injection form and BNF therapeutic chapter add "
        "independent information beyond price: IV products carry sterile-manufacturing "
        "and packaging overhead, and BNF chapter captures treatment-area variation that "
        "dosage form alone misses."
    )
    st.markdown(
        "**HFC inhaler flag ranks 14th** (SHAP 0.003) because HFC inhalers are rare in "
        "secondary-care dispensing. The physical signal is real and well-evidenced, "
        "but low prevalence in SCMD suppresses the SHAP weight. The same feature "
        "would rank substantially higher in a primary-care dataset."
    )

    if os.path.exists(FIG_SHAP_BAR):
        st.divider()
        st.subheader("Publication figure (300 DPI)")
        st.image(FIG_SHAP_BAR, caption="Top 10 features by mean |SHAP| — Random Forest")
    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SUPPLIER TRANSPARENCY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Supplier Transparency":
    st.title("Supplier Transparency")
    st.markdown(
        "Carbon Reduction Plan (CRP) disclosures collected manually from 15 NHS "
        "suppliers. Coverage analysis reveals a structural barrier: SCMD uses "
        "International Nonproprietary Names (INN), so generic drug names carry "
        "no manufacturer identity. CRP data cannot be linked to dispensing records "
        "for 99.85% of medicines."
    )

    crp = load_crp()

    usable_mask = crp["usable_as_direct_label"].str.startswith(
        ("Yes", "Partial"), na=False
    )
    n_total     = len(crp)
    n_usable    = int(usable_mask.sum())
    n_intensity = int(crp["carbon_intensity_tco2e_per_gbpm"].notna().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suppliers targeted",          str(n_total))
    c2.metric("Usable CRP data",             f"{n_usable} of {n_total} ({100 * n_usable // n_total}%)")
    c3.metric("Disclosed intensity ratios",  str(n_intensity))
    c4.metric("VMPs matched in SCMD",        "13 of 8,473 (0.15%)")

    st.divider()
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Supplier CRP summary")
        show = ["supplier_name", "scope_geography", "reporting_year",
                "usable_as_direct_label", "carbon_intensity_tco2e_per_gbpm",
                "total_tco2e", "verification_level"]
        show = [c for c in show if c in crp.columns]
        disp = crp[show].copy()
        if "usable_as_direct_label" in disp.columns:
            disp["usable_as_direct_label"] = disp["usable_as_direct_label"].str[:55]
        if "verification_level" in disp.columns:
            disp["verification_level"] = disp["verification_level"].str[:50]
        disp.columns = [c.replace("_", " ").title() for c in disp.columns]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Scope geography")
        if "scope_geography" in crp.columns:
            geo = (
                crp["scope_geography"]
                .fillna("Not stated")
                .value_counts()
                .reset_index()
            )
            geo.columns = ["Scope", "Count"]
            fig = px.pie(
                geo, values="Count", names="Scope",
                color_discrete_sequence=[NAVY, TEAL, LTEAL, GREY, "#7B8FA1"],
                hole=0.4,
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="v", font_size=9),
                height=270,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Supplier notes")
    st.caption("Click to expand detailed notes from CRP collection.")
    if "notes" in crp.columns:
        for _, row in crp.iterrows():
            if pd.notna(row.get("notes")) and str(row["notes"]).strip():
                label = str(row["supplier_name"])[:70]
                with st.expander(label):
                    st.write(str(row["notes"]))

    st.divider()
    st.subheader("The INN naming convention finding")
    st.info(
        "SCMD uses INN (generic) names for all medicines. Only products with no "
        "approved generic equivalent retain identifiable supplier signatures in "
        "the VMP name. In practice this is restricted to parenteral nutrition "
        "formulations (Kabiven, SMOFlipid, Clinimix, Prontosan). "
        "Matching CRP data from 15 suppliers against 8,473 unique VMPs yielded "
        "**13 matched VMPs** across two suppliers (Fresenius Kabi and B. Braun), "
        "covering less than 0.01% of total spend. "
        "This is a citable structural limitation of NHS administrative data for "
        "supplier-level carbon attribution, independent of CRP quality or availability."
    )
    show_footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Methodology":
    st.title("Methodology")

    results_df = load_results()
    st.subheader("Model performance — held-out test set (n = 59,717)")
    drop_cols = [c for c in results_df.columns if "note" in c.lower()]
    st.dataframe(
        results_df.drop(columns=drop_cols, errors="ignore"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "The DEFRA baseline R² = 1.00 and MAE ≈ 1 kgCO2e "
        "reflect label circularity, not predictive accuracy. See the note below."
    )

    st.divider()

    if os.path.exists(FIG_SCATTER):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted vs actual (log space)")
            st.image(
                FIG_SCATTER,
                caption=(
                    "R² = 0.76 in log1p space. "
                    "Each point is one dispensing record in the held-out test set."
                ),
            )
        with col2:
            st.subheader("Reading the scatter")
            st.markdown(
                "The model explains 76% of variance in log-transformed carbon "
                "using 15 pharmacological features, with no access to spend or "
                "total quantity. In original kgCO2e space R² = 0.25, "
                "reflecting the difficulty of recovering a severely skewed "
                "distribution (mean/median ratio = 49x) from categorical features.\n\n"
                "The scatter shows the model captures the overall price-driven "
                "trend correctly. It cannot resolve fine-grained variation within "
                "product types, because products sharing the same dosage form and "
                "price differ in features not present in SCMD: manufacturer, "
                "country of origin, upstream energy mix.\n\n"
                "**log1p transformation** was applied because skewness = 270.56. "
                "Predictions are inverse-transformed via expm1 before reporting in "
                "kgCO2e. The transformation is applied transparently in all tables."
            )

    st.divider()
    st.subheader("Methodology note")
    if os.path.exists(METHOD_PATH):
        with open(METHOD_PATH, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning(
            "reports/methodology_note.md not found. "
            "Run src/analysis/generate_paper_outputs.py to generate it."
        )
    show_footer()
