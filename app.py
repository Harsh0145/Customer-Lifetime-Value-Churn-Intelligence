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
from src.uplift import HILLSTROM_CREDIT, HILLSTROM_URL, fit_uplift, load_hillstrom


@st.cache_resource(show_spinner=False)
def get_model_bundle(frame: pd.DataFrame):
    return model_bundle(frame)


@st.cache_data(show_spinner=False)
def get_uplift_result(frame: pd.DataFrame, arm: str, outcome: str, target_fraction: float):
    return fit_uplift(frame, treatment_arm=arm, outcome=outcome, target_fraction=target_fraction)


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

overview, drivers, survival, offers, uplift_tab, data_tab = st.tabs(["Overview", "Drivers", "Survival & value", "Offer ROI", "Campaign uplift", "Data & database"])

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

with uplift_tab:
    st.subheader("Randomized campaign uplift")
    st.markdown(
        "Estimate who responds to an offer using a real randomized campaign experiment. "
        "The Hillstrom email experiment is a marketing-response benchmark; its outcomes are "
        "purchases or visits, not telecom churn prevention."
    )
    st.caption(f"Source: {HILLSTROM_CREDIT}. Each email arm is compared with the no-email control.")
    campaign_upload = st.file_uploader(
        "Or upload your randomized campaign CSV",
        type="csv",
        key="uplift_campaign_upload",
        help="This version accepts the Hillstrom email experiment CSV with segment, visit, and conversion columns.",
    )
    load_campaign = st.button("Load Hillstrom randomized campaign", key="load_hillstrom_campaign")
    if campaign_upload is not None:
        try:
            st.session_state.campaign_frame = pd.read_csv(campaign_upload)
        except Exception as exc:
            st.error(f"Could not read campaign CSV: {exc}")
    if load_campaign:
        try:
            with st.spinner("Downloading the 64,000-customer randomized experiment…"):
                st.session_state.campaign_frame = load_hillstrom()
        except Exception as exc:
            st.error(f"Could not download Hillstrom: {exc}. Upload the CSV to continue.")
    if "campaign_frame" not in st.session_state:
        st.info("Choose Load Hillstrom randomized campaign or upload an experiment CSV to begin.")
        st.markdown(f"[Open the Hillstrom source dataset]({HILLSTROM_URL})")
    else:
        campaign_frame = st.session_state.campaign_frame
        if {"segment", "visit", "conversion"}.issubset(campaign_frame.columns):
            c1, c2, c3 = st.columns(3)
            arm = c1.selectbox("Email treatment arm", ["Mens E-Mail", "Womens E-Mail"])
            outcome = c2.selectbox("Response outcome", ["conversion", "visit"])
            target_fraction = c3.slider("Target highest-uplift share", 5, 50, 20, 5) / 100
            if st.button("Estimate campaign uplift", type="primary", key="fit_campaign_uplift"):
                try:
                    with st.spinner("Fitting CausalML T-learner and evaluating randomized holdout…"):
                        st.session_state.uplift_result = get_uplift_result(
                            campaign_frame, arm, outcome, target_fraction
                        )
                        st.session_state.uplift_result_key = (arm, outcome, target_fraction)
                except ImportError:
                    st.error("CausalML is not installed. Redeploy after committing the updated requirements.txt.")
                except Exception as exc:
                    st.error(f"Uplift model could not be fitted: {exc}")
            result = st.session_state.get("uplift_result")
            if result and st.session_state.get("uplift_result_key") == (arm, outcome, target_fraction):
                st.success(f"Model: {result['method']}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Experiment customers", f"{result['n_customers']:,}")
                m2.metric("Treatment / control", f"{result['treatment_n']:,} / {result['control_n']:,}")
                m3.metric(f"Observed holdout lift · top {target_fraction:.0%}", f"{result['targeted_uplift']:+.2%}", help=f"95% approximate CI: {result['targeted_ci_low']:+.2%} to {result['targeted_ci_high']:+.2%}")
                m4.metric("Overall holdout lift", f"{result['overall_uplift']:+.2%}", help=f"95% approximate CI: {result['overall_ci_low']:+.2%} to {result['overall_ci_high']:+.2%}")
                st.caption(
                    f"Lift confidence intervals use the randomized holdout and are approximate normal intervals. "
                    f"The score ranks customers; the reported top-group result is measured on holdout customers, "
                    f"not on the data used to train the model. Holdout contains {result['holdout_n']:,} customers."
                )
                if not result["holdout_curve"].empty:
                    curve = result["holdout_curve"].sort_values("ranked_group", ascending=False)
                    st.plotly_chart(
                        px.bar(curve, x="ranked_group", y="observed_uplift", title="Observed randomized lift by predicted-uplift rank group", labels={"ranked_group":"Highest predicted uplift group", "observed_uplift":"Treatment minus control response rate"}),
                        use_container_width=True,
                    )
                st.markdown("#### Campaign economics scenario")
                st.caption("This scales the measured holdout response lift to a planned contact volume. Enter value per incremental response and contact cost; it is a campaign scenario, not a churn-save estimate.")
                econ_a, econ_b, econ_c = st.columns(3)
                planned_contacts = econ_a.number_input("Planned contacts", min_value=0, value=min(5000, result["n_customers"]), step=500)
                value_per_response = econ_b.number_input("Value per incremental response ($)", min_value=0.0, value=80.0, step=5.0)
                contact_cost = econ_c.number_input("Cost per contact ($)", min_value=0.0, value=0.05, step=0.01, format="%.2f")
                incremental_responses = planned_contacts * max(0.0, result["targeted_uplift"])
                campaign_cost = planned_contacts * contact_cost
                incremental_value = incremental_responses * value_per_response
                net_campaign_value = incremental_value - campaign_cost
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Expected incremental responses", f"{incremental_responses:,.1f}")
                e2.metric("Estimated incremental value", f"${incremental_value:,.2f}")
                e3.metric("Campaign contact cost", f"${campaign_cost:,.2f}")
                e4.metric("Net scenario value", f"${net_campaign_value:,.2f}")
                st.dataframe(result["recommended_customers"].head(20), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download ranked uplift recommendations",
                    result["recommended_customers"].to_csv(index=False).encode("utf-8"),
                    "campaign_uplift_recommendations.csv",
                    "text/csv",
                )
                st.warning("The estimated lift applies to this randomized email benchmark. Do not apply it as a causal telecom retention effect; train and evaluate on your own randomized retention-offer experiment before using it for churn campaigns.")
        else:
            st.error("Uploaded campaign data must use the Hillstrom schema (segment, visit, conversion) for this version.")

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
