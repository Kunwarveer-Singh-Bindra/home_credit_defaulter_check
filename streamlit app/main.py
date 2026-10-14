from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]

    @property
    def model(self) -> Path:
        return Path(os.environ.get("MODEL_PATH", self.root / "Notebooks" / "catboost_model.cbm"))

    @property
    def train_data(self) -> Path:
        return Path(os.environ.get("TRAIN_DATA_PATH", self.root / "output" / "train_main.csv"))


PATHS = Paths()

RISK_HIGH_THRESHOLD = 0.35
RISK_WATCH_THRESHOLD = 0.18
LOOKUP_CHUNKSIZE = 10_000

ORGANIZATION_TYPES = [
    "Business Entity Type 3", "Self-employed", "Other", "Medicine", "Business Entity Type 2",
    "Government", "School", "Trade: type 7", "Kindergarten", "Construction", "Business Entity Type 1",
    "Transport: type 4", "Industry: type 9", "Military", "Bank", "Security", "Housing", "Agriculture",
    "Police", "Transport: type 2", "Postal", "Restaurant", "Trade: type 3", "Security Ministries",
    "University", "Legal Services", "XNA",
]

st.set_page_config(
    page_title="CreditLens | Loan Default Risk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Safe coercion helpers — `value or fallback` breaks on NaN (NaN is truthy
# in Python), which is common in this dataset. These check pd.isna explicitly.
# ----------------------------------------------------------------------------

def safe_int(value: Any, fallback: int) -> int:
    if value is None:
        return int(fallback)
    try:
        if pd.isna(value):
            return int(fallback)
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def safe_float(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    try:
        if pd.isna(value):
            return float(fallback)
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def safe_str(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    return str(value)


def safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return fallback
    return numerator / denominator


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ink:#102a43; --muted:#486581; --teal:#0f766e; --teal-dark:#0b5a54;
  --mint:#d7f5ee; --coral:#f97362;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--ink) !important; letter-spacing: -0.03em; }

.stApp { background: #f7fafc; }
.main p, .main span, .main label, .main li,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
[data-testid="stWidgetLabel"] * {
  color: var(--ink) !important;
}
.main [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }

[data-testid="stSidebar"] { background: #102a43; }
[data-testid="stSidebar"] * { color: #e6fffa !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #9fb3c8 !important; }

.hero { padding: 1.6rem 2rem 1.4rem; border-radius: 24px; background: linear-gradient(125deg,#d7f5ee 0%,#eefbf8 48%,#fff5f1 100%); border: 1px solid #c7eee5; margin-bottom: 1.3rem; }
.eyebrow { color: var(--teal); font-size: .74rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
.hero h1 { margin: .35rem 0 .3rem; font-size: 2.55rem; }
.hero p { color: var(--muted) !important; font-size: 1.03rem; max-width: 780px; margin: 0; }

.section-label { color: var(--teal) !important; font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 1.1rem 0 .3rem; }
.section-note { color: var(--muted) !important; font-size: .82rem; margin: -.15rem 0 .5rem; }

.card { background: white; border: 1px solid #e5edf3; border-radius: 18px; padding: 1.15rem 1.25rem; box-shadow: 0 6px 20px rgba(16,42,67,.04); }
.metric-label { color: var(--muted) !important; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
.risk-pill { padding:.45rem .75rem; border-radius:999px; display:inline-block; font-weight:700; font-size:.9rem; }
.risk-low { color:#116466 !important; background:#d7f5ee; }
.risk-watch { color:#9c4221 !important; background:#ffecd9; }
.risk-high { color:#9b2c2c !important; background:#ffe0de; }
.footnote { color:#829ab1 !important; font-size:.78rem; line-height:1.45; }

div[data-testid="stMetric"] { background:white; border:1px solid #e5edf3; padding: .75rem 1rem; border-radius:16px; }
div[data-testid="stMetric"] * { color: var(--ink) !important; }

.stButton > button[kind="primary"] {
  background: var(--teal) !important; border-color: var(--teal) !important; color: white !important;
}
.stButton > button[kind="primary"]:hover { background: var(--teal-dark) !important; }
</style>
"""


def inject_styles() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data / model loading
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(PATHS.model))
    return model


@st.cache_data(show_spinner=False)
def load_baseline() -> pd.Series:
    return pd.read_csv(PATHS.train_data, nrows=1).iloc[0]


@st.cache_data(show_spinner=False)
def find_applicant(applicant_id: int) -> pd.Series | None:
    for chunk in pd.read_csv(PATHS.train_data, chunksize=LOOKUP_CHUNKSIZE):
        match = chunk.loc[chunk["SK_ID_CURR"].eq(applicant_id)]
        if not match.empty:
            return match.iloc[0]
    return None


@st.cache_data(show_spinner=False)
def top_features(_model: CatBoostClassifier, n: int = 12) -> pd.DataFrame:
    importance = np.asarray(_model.get_feature_importance())
    return (
        pd.DataFrame({"feature": _model.feature_names_, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(n)
    )


def as_catboost_frame(record: pd.Series, model: CatBoostClassifier) -> pd.DataFrame:
    frame = pd.DataFrame([record]).reindex(columns=model.feature_names_)
    cat_columns = {model.feature_names_[i] for i in model.get_cat_feature_indices()}
    for col in frame.columns:
        if col in cat_columns:
            frame[col] = frame[col].fillna("Missing").astype(str)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def recompute_engineered_features(record: pd.Series) -> pd.Series:
    """Keep the notebook's engineered ratio columns consistent with whatever
    the user just edited (AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
    AMT_INCOME_TOTAL, DAYS_BIRTH) — otherwise the model scores on stale
    ratios left over from the loaded baseline applicant.

    Formulas copied verbatim from the training feature-engineering step:
        AGE_YEARS               = DAYS_BIRTH / -365
        CREDIT_ANNUITY_RATIO    = AMT_CREDIT / AMT_ANNUITY
        CREDIT_GOODS_PRICE_RATIO= AMT_CREDIT / AMT_GOODS_PRICE
        CREDIT_DOWNPAYMENT      = AMT_GOODS_PRICE - AMT_CREDIT
        CREDIT_INCOME_RATIO     = AMT_CREDIT / AMT_INCOME_TOTAL
        ANNUITY_INCOME_RATIO    = AMT_ANNUITY / AMT_INCOME_TOTAL
    """
    credit = safe_float(record.get("AMT_CREDIT"), 0.0)
    annuity = safe_float(record.get("AMT_ANNUITY"), 0.0)
    goods_price = safe_float(record.get("AMT_GOODS_PRICE"), 0.0)
    income = safe_float(record.get("AMT_INCOME_TOTAL"), 0.0)
    days_birth = safe_float(record.get("DAYS_BIRTH"), 0.0)

    record["AGE_YEARS"] = int(days_birth / -365) if days_birth else 0
    record["CREDIT_ANNUITY_RATIO"] = safe_div(credit, annuity)
    record["CREDIT_GOODS_PRICE_RATIO"] = safe_div(credit, goods_price)
    record["CREDIT_DOWNPAYMENT"] = goods_price - credit
    record["CREDIT_INCOME_RATIO"] = safe_div(credit, income)
    record["ANNUITY_INCOME_RATIO"] = safe_div(annuity, income)
    return record


# ----------------------------------------------------------------------------
# Scoring logic
# ----------------------------------------------------------------------------

def classify_risk(probability: float) -> tuple[str, str, str]:
    if probability >= RISK_HIGH_THRESHOLD:
        return ("High risk", "risk-high",
                "A higher-than-usual probability of default. Review affordability and supporting evidence carefully.")
    if probability >= RISK_WATCH_THRESHOLD:
        return ("Watch", "risk-watch",
                "A moderate probability of default. Consider additional verification or a more conservative offer.")
    return ("Lower risk", "risk-low",
            "The model estimates a relatively lower probability of default for this profile.")


def money(value: float) -> str:
    return f"₹{value:,.0f}"


# ----------------------------------------------------------------------------
# UI building blocks
# ----------------------------------------------------------------------------

def render_sidebar() -> tuple[str, int | None]:
    with st.sidebar:
        st.markdown("## ◈ CreditLens")
        st.caption("Loan default risk intelligence")
        st.divider()
        st.markdown("### Data mode")
        mode = st.radio("Choose how to score", ["Existing applicant", "Quick assessment"], label_visibility="collapsed")
        applicant_id = None
        if mode == "Existing applicant":
            applicant_id = st.number_input("Applicant ID", min_value=100001, value=100002, step=1)
            st.caption("Loads that applicant's full engineered record; the form below edits the fields a bank would actually collect.")
        else:
            st.caption("Starts from a baseline record; fill in the form as if taking a fresh application.")
        st.divider()
        st.markdown("**Model**  ·  CatBoost classifier")
        st.markdown(f"**Features**  ·  {len(load_model().feature_names_):,} engineered inputs")
        st.markdown("**Output**  ·  Default probability")
    return mode, applicant_id


def render_hero() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Decision support · credit risk</div>'
        '<h1>See the risk before the loan.</h1>'
        '<p>A clean, explainable view of your loan-default model. Fill in an application below and get '
        'a probability-led risk signal in seconds.</p></div>',
        unsafe_allow_html=True,
    )


def applicant_form(default: pd.Series) -> dict[str, Any]:
    """Mirrors a realistic bank loan-application form, grouped into the
    sections a real intake form uses, plus the credit-bureau scores that
    dominate this model's predictions (EXT_SOURCE_1/2/3 — by far the three
    most important features; a real bank pulls these automatically on
    consent rather than asking the applicant to type them)."""

    st.markdown('<div class="section-label">Credit bureau scores</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">Normalized external credit-bureau scores. '
        'These are the single strongest predictors in the model — a real bank pulls them automatically.</p>',
        unsafe_allow_html=True,
    )
    e1, e2, e3 = st.columns(3)
    with e1:
        ext1 = st.slider("Bureau score 1", 0.0, 1.0, safe_float(default.get("EXT_SOURCE_1"), 0.5), 0.01,
                          help="Model field: EXT_SOURCE_1")
    with e2:
        ext2 = st.slider("Bureau score 2", 0.0, 1.0, safe_float(default.get("EXT_SOURCE_2"), 0.5), 0.01,
                          help="Model field: EXT_SOURCE_2")
    with e3:
        ext3 = st.slider("Bureau score 3", 0.0, 1.0, safe_float(default.get("EXT_SOURCE_3"), 0.5), 0.01,
                          help="Model field: EXT_SOURCE_3")

    st.markdown('<div class="section-label">Personal details</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["M", "F"],
                               index=0 if safe_str(default.get("CODE_GENDER"), "M") == "M" else 1,
                               help="Model field: CODE_GENDER")
    with c2:
        days_birth = safe_float(default.get("DAYS_BIRTH"), -12000)
        age = st.number_input("Age (years)", min_value=18, max_value=75,
                               value=safe_int(abs(days_birth) / 365, 33),
                               help="Model field: DAYS_BIRTH")
    with c3:
        family_status = st.selectbox(
            "Marital status",
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
            help="Model field: NAME_FAMILY_STATUS",
        )
    with c4:
        family_members = st.number_input("Household size", min_value=1, max_value=15,
                                          value=safe_int(default.get("CNT_FAM_MEMBERS"), 1),
                                          help="Model field: CNT_FAM_MEMBERS")

    c5, c6, c7 = st.columns(3)
    with c5:
        children = st.number_input("Number of children", min_value=0, max_value=20,
                                    value=safe_int(default.get("CNT_CHILDREN"), 0),
                                    help="Model field: CNT_CHILDREN")
    with c6:
        education = st.selectbox(
            "Education level",
            ["Secondary / secondary special", "Higher education", "Incomplete higher",
             "Lower secondary", "Academic degree"],
            help="Model field: NAME_EDUCATION_TYPE",
        )
    with c7:
        id_published_years = st.number_input("ID document issued (years ago)", min_value=0, max_value=50,
                                               value=safe_int(abs(safe_float(default.get("DAYS_ID_PUBLISH"), -3000)) / 365, 8),
                                               help="Model field: DAYS_ID_PUBLISH")

    st.markdown('<div class="section-label">Employment & income</div>', unsafe_allow_html=True)
    c8, c9, c10 = st.columns(3)
    with c8:
        income_type = st.selectbox(
            "Income type",
            ["Working", "Commercial associate", "Pensioner", "State servant", "Student", "Unemployed"],
            help="Model field: NAME_INCOME_TYPE",
        )
    with c9:
        occupation = st.selectbox(
            "Occupation",
            ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff",
             "Accountants", "Medicine staff", "Security staff", "Cooking staff", "Other"],
            help="Model field: OCCUPATION_TYPE",
        )
    with c10:
        organization = st.selectbox("Employer industry", ORGANIZATION_TYPES,
                                     help="Model field: ORGANIZATION_TYPE")

    c11, c12 = st.columns(2)
    with c11:
        days_employed = safe_float(default.get("DAYS_EMPLOYED"), -637)
        employed_years = st.number_input("Years at current job", min_value=0, max_value=50,
                                          value=safe_int(abs(days_employed) / 365, 2),
                                          help="Model field: DAYS_EMPLOYED")
    with c12:
        income = st.number_input("Annual income", min_value=0.0,
                                  value=safe_float(default.get("AMT_INCOME_TOTAL"), 202500.0), step=5000.0, format="%.0f",
                                  help="Model field: AMT_INCOME_TOTAL")

    st.markdown('<div class="section-label">Loan details</div>', unsafe_allow_html=True)
    c13, c14, c15 = st.columns(3)
    with c13:
        contract = st.selectbox("Loan type", ["Cash loans", "Revolving loans"],
                                 index=0 if safe_str(default.get("NAME_CONTRACT_TYPE"), "Cash loans") == "Cash loans" else 1,
                                 help="Model field: NAME_CONTRACT_TYPE")
    with c14:
        credit = st.number_input("Requested credit amount", min_value=0.0,
                                  value=safe_float(default.get("AMT_CREDIT"), 406597.5), step=5000.0, format="%.0f",
                                  help="Model field: AMT_CREDIT")
    with c15:
        goods_price = st.number_input("Price of goods/property financed", min_value=0.0,
                                       value=safe_float(default.get("AMT_GOODS_PRICE"), credit), step=5000.0, format="%.0f",
                                       help="Model field: AMT_GOODS_PRICE")

    st.markdown('<div class="section-label">Assets & housing</div>', unsafe_allow_html=True)
    c16, c17, c18, c19 = st.columns(4)
    with c16:
        housing = st.selectbox(
            "Housing type",
            ["House / apartment", "With parents", "Rented apartment",
             "Municipal apartment", "Office apartment", "Co-op apartment"],
            help="Model field: NAME_HOUSING_TYPE",
        )
    with c17:
        own_realty = st.selectbox("Owns property", ["Y", "N"],
                                   index=0 if safe_str(default.get("FLAG_OWN_REALTY"), "Y") == "Y" else 1,
                                   help="Model field: FLAG_OWN_REALTY")
    with c18:
        own_car = st.selectbox("Owns a car", ["N", "Y"],
                                index=0 if safe_str(default.get("FLAG_OWN_CAR"), "N") == "N" else 1,
                                help="Model field: FLAG_OWN_CAR")
    with c19:
        car_age = st.number_input("Car age (years)", min_value=0, max_value=50,
                                   value=safe_int(default.get("OWN_CAR_AGE"), 0),
                                   disabled=(own_car == "N"), help="Model field: OWN_CAR_AGE")

    return {
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "CODE_GENDER": gender,
        "DAYS_BIRTH": -int(age * 365),
        "NAME_FAMILY_STATUS": family_status,
        "CNT_FAM_MEMBERS": family_members,
        "CNT_CHILDREN": children,
        "NAME_EDUCATION_TYPE": education,
        "DAYS_ID_PUBLISH": -int(id_published_years * 365),
        "NAME_INCOME_TYPE": income_type,
        "OCCUPATION_TYPE": occupation,
        "ORGANIZATION_TYPE": organization,
        "DAYS_EMPLOYED": -int(employed_years * 365),
        "AMT_INCOME_TOTAL": income,
        "NAME_CONTRACT_TYPE": contract,
        "AMT_CREDIT": credit,
        "AMT_GOODS_PRICE": goods_price,
        "NAME_HOUSING_TYPE": housing,
        "FLAG_OWN_REALTY": own_realty,
        "FLAG_OWN_CAR": own_car,
        "OWN_CAR_AGE": car_age if own_car == "Y" else 0,
    }


def render_result(result: dict[str, Any], current: pd.Series) -> None:
    probability = result["probability"]
    st.markdown("### Assessment result")

    a, b, c, d = st.columns(4)
    a.metric("Default probability", f"{probability:.1%}")
    b.metric("Risk band", result["label"])
    c.metric("Requested credit", money(safe_float(current.get("AMT_CREDIT"), 0.0)))
    d.metric("Credit / income",
             f"{safe_div(safe_float(current.get('AMT_CREDIT'), 0.0), safe_float(current.get('AMT_INCOME_TOTAL'), 1.0)):.1%}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(
            f'<div class="card"><div class="metric-label">Model signal</div>'
            f'<div style="margin:.55rem 0 .4rem"><span class="risk-pill {result["css"]}">{result["label"]}</span></div>'
            f'<p style="color:var(--muted);margin:0">{result["explanation"]}</p></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(probability, 1.0), text="Estimated probability of default")
    with right:
        st.markdown(
            '<div class="card"><div class="metric-label">Important context</div>'
            '<p style="color:var(--muted);margin:.55rem 0 0">This is a model estimate, not an approval or '
            'rejection decision. Use it alongside policy rules, affordability checks, and human review.</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander("Model context & feature importance"):
        imp = top_features(result["model"])
        if not imp.empty:
            imp = imp.rename(columns={"feature": "Feature", "importance": "Importance"})
            st.dataframe(imp, hide_index=True, use_container_width=True)
        st.caption("Feature importance is a model-level diagnostic; it does not explain this individual prediction causally.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    inject_styles()

    try:
        model = load_model()
        baseline = load_baseline()
    except Exception as exc:
        st.error(f"The model could not be loaded: {exc}")
        st.stop()

    mode, applicant_id = render_sidebar()
    render_hero()

    if mode == "Existing applicant":
        with st.spinner("Looking up applicant record…"):
            record = find_applicant(int(applicant_id))
        if record is None:
            st.warning("No engineered record found for that applicant ID. Try 100002 or switch to Quick assessment.")
            st.stop()
        current = record.copy()
        st.success(f"Loaded applicant {int(applicant_id)} from the engineered dataset.")
    else:
        current = baseline.copy()

    score_key = (mode, applicant_id)
    if st.session_state.get("score_key") != score_key:
        st.session_state.pop("result", None)
        st.session_state["score_key"] = score_key

    overrides = applicant_form(current)
    for key, value in overrides.items():
        current[key] = value
    if overrides["AMT_INCOME_TOTAL"] > 0:
        current["AMT_ANNUITY"] = min(
            safe_float(current.get("AMT_ANNUITY"), 24700.5), overrides["AMT_INCOME_TOTAL"] * 0.4,
        )

    # Keep engineered ratio columns consistent with whatever was just edited
    current = recompute_engineered_features(current)

    if st.button("Run risk assessment", type="primary", use_container_width=True):
        with st.spinner("Scoring application…"):
            try:
                features = as_catboost_frame(current, model)
                probability = float(model.predict_proba(features)[0, 1])
                label, css, explanation = classify_risk(probability)
                st.session_state["result"] = {
                    "probability": probability, "label": label, "css": css,
                    "explanation": explanation, "model": model,
                }
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")

    result = st.session_state.get("result")
    if result:
        render_result(result, current)
    else:
        st.info("Complete the application and click **Run risk assessment** to see the model signal.")

    st.markdown(
        '<div class="footnote">For demonstration and decision-support use only. '
        'The trained model inherits the limitations and dataset assumptions of the original project.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()