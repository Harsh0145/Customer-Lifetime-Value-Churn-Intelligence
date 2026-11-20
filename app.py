"""Customer Lifetime Value and Churn Intelligence System."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.pipeline import (  # noqa: E402
    DATA_URL,
    build_features,
    load_dataset,
    model_bundle,
    score_customers,
    uplift_simulation,
    write_postgres,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle(frame: pd.DataFrame):
    return model_bundle(frame)


st.set_page_config(page_title="Customer Intelligence", page_icon="📡", layout="wide")
st.title("📡 Customer Lifetime Value & Churn Intelligence")
st.caption("An end-to-end, educational retention analytics workbench built on the IBM Telco sample.")

with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Use your own compatible CSV", type="csv", help="Expected IBM Telco columns; see Data Dictionary.")
    demo = st.button("Load IBM Telco sample", use_container_width=True)
    st.caption("The default sample is fetched from IBM's public GitHub mirror. You can upload the CSV if hosting has no outbound access.")
    if st.button("Clear loaded data", use_container_width=True):
        st.session_state.pop("customers", None)

if upload is not None:
    try:
        st.session_state.customers = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
if demo:
    try:
        st.session_state.customers = load_dataset(DATA_URL)
    except Exception as exc:
        st.error(f"Could not download sample: {exc}. Upload the CSV to continue.")

if "customers" not in st.session_state:
    st.info("Load the sample dataset or upload a CSV to begin.")
    st.markdown(f"[Download the IBM Telco sample CSV]({DATA_URL})")
    st.stop()

try:
    raw = st.session_state.customers.copy()
    data = build_features(raw)
except Exception as exc:
    st.error(f"The file does not match the expected dataset format: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("Retention offer assumptions")
    discount = st.slider("Monthly discount", 0, 50, 15, 1, format="$%d")
    duration = st.slider("Offer duration (months)", 1, 12, 6)
    conversion = st.slider("Assumed offer effectiveness", 0, 50, 15, 1, format="%d%%") / 100
    st.caption("Offer response is simulated; effectiveness is an assumption, not a measured causal effect.")

with st.spinner("Fitting churn, survival, and segmentation models…"):
    try:
        bundle = get_model_bundle(data)
        scored = score_customers(data, bundle)
    except Exception as exc:
        st.exception(exc)
        st.stop()

overview, drivers, survival, offers, data_tab = st.tabs(["Overview", "Drivers", "Survival & value", "Offer ROI", "Data & database"])

with overview:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(scored):,}")
    m2.metric("Observed churn", f"{(scored.Churn == 'Yes').mean():.1%}")
    m3.metric("High-risk customers", f"{(scored.churn_probability >= 0.5).sum():,}")
    m4.metric("Mean predicted CLV", f"${scored.predicted_clv.mean():,.0f}")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(scored, x="churn_probability", nbins=25, title="Predicted churn risk", labels={"churn_probability":"Churn probability"}), use_container_width=True)
    right.plotly_chart(px.histogram(scored, x="Contract", color="Churn", barmode="group", title="Churn by contract", histfunc="count"), use_container_width=True)
    st.subheader("Customers to review")
    st.dataframe(scored.sort_values("churn_probability", ascending=False)[["customerID", "Contract", "tenure", "MonthlyCharges", "churn_probability", "expected_remaining_months", "predicted_clv", "segment"]].head(20), use_container_width=True, hide_index=True)
    st.download_button("Download scored customers", scored.to_csv(index=False).encode(), "customer_scores.csv", "text/csv")

with drivers:
    st.markdown("Feature importance is computed with SHAP when available; the chart is global and does not establish causation.")
    st.metric("Holdout ROC AUC", f"{bundle['auc']:.3f}" if pd.notna(bundle["auc"]) else "Unavailable")
    st.metric("Holdout average precision", f"{bundle['average_precision']:.3f}" if pd.notna(bundle["average_precision"]) else "Unavailable")
    if bundle.get("shap_importance") is not None:
        imp = bundle["shap_importance"].head(15).sort_values("mean_abs_shap")
        st.plotly_chart(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", title="Top global SHAP drivers"), use_container_width=True)
    else:
        st.info("SHAP unavailable in this environment; showing model gain importance.")
        imp = bundle["importance"].head(15).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", title="Top model drivers"), use_container_width=True)
    st.caption("Metrics use a stratified holdout split. Dataset is a historical snapshot; validate on time-separated operational data before deployment.")

with survival:
    st.markdown("A Cox proportional hazards model estimates time until churn. Customers still active at the snapshot are right-censored. CLV is a discounted expected-revenue proxy, not profit or audited accounting value.")
    st.plotly_chart(px.histogram(scored, x="expected_remaining_months", nbins=25, title="Expected remaining customer lifetime (months)"), use_container_width=True)
    seg = scored.groupby("segment", as_index=False).agg(customers=("customerID", "count"), mean_risk=("churn_probability", "mean"), mean_clv=("predicted_clv", "mean"))
    st.dataframe(seg, use_container_width=True, hide_index=True)

with offers:
    st.markdown("### Retention budget simulator")
    st.warning("The IBM sample has no offer assignment or offer outcome. This is a transparent scenario model, not a causal uplift estimate. Use randomized campaign data to estimate real treatment effects.")
    cohort = scored[scored.churn_probability >= 0.25].copy()
    result = uplift_simulation(cohort, discount, duration, conversion)
    a, b, c, d = st.columns(4)
    a.metric("Customers targeted", f"{len(cohort):,}")
    b.metric("Expected saves (assumed)", f"{result['expected_saves']:,.0f}")
    c.metric("Offer cost", f"${result['cost']:,.0f}")
    d.metric("Expected retained revenue", f"${result['retained_revenue']:,.0f}")
    st.metric("Net value after offer cost", f"${result['net_value']:,.0f}", delta=f"ROI {result['roi']:.1%}" if result['cost'] else "No discount cost")
    st.caption("Savings/value assumes expected saved customers would otherwise churn, uses estimated CLV as retained revenue, and ignores margin, discount redemption, and contact costs.")

with data_tab:
    st.subheader("Dataset preview")
    st.dataframe(raw.head(25), use_container_width=True, hide_index=True)
    st.caption(f"{len(raw):,} records · {len(raw.columns)} source columns · {raw.isna().sum().sum()} missing cells")
    st.subheader("Optional PostgreSQL load")
    try:
        secret_db_url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        secret_db_url = ""
    db_url = st.text_input("SQLAlchemy connection URL", value=secret_db_url or os.getenv("DATABASE_URL", ""), type="password", placeholder="postgresql+psycopg://user:password@host:5432/database")
    if st.button("Write source rows to PostgreSQL", disabled=not db_url):
        try:
            with st.spinner("Writing customer_source table…"):
                write_postgres(raw, db_url)
            st.success("Loaded source data to customer_source.")
        except Exception as exc:
            st.error(f"Database load failed: {exc}")
    st.caption("Credentials are entered at runtime and are not saved. In Streamlit Community Cloud, set DATABASE_URL under app Secrets instead of committing credentials.")

with st.expander("Model and data caveats"):
    st.markdown("""
- The Telco dataset is a one-time sample, not a longitudinal usage or complaint history. Those concepts are not fabricated as if observed.
- Tenure, charges, contract and service fields are used; churn is the observed target.
- Kaplan–Meier and Cox estimates treat current non-churners as censored at observed tenure.
- CLV is modeled from monthly charges, estimated survival and a user-configurable discount rate. It excludes costs and margin.
- Offer ROI uses user-supplied response assumptions. It is not uplift modeling unless treatment and outcome are observed under a credible experiment.
""")
