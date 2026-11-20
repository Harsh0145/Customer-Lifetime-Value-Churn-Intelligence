# Customer Lifetime Value & Churn Intelligence System

An interactive Streamlit application that scores telecom churn risk, estimates customer survival, approximates lifetime revenue, segments customers, and explores retention-offer economics. It uses the public IBM Telco Customer Churn sample by default and supports compatible CSV uploads.

> **Interpretation:** this is a portfolio/learning demo, not a production decision system. The supplied dataset is a single historical snapshot. It contains no offer assignment, offer outcome, complaints, or longitudinal usage history. Offer response in the simulator is an editable assumption, not an estimated causal uplift. CLV is a discounted revenue proxy, not profit.

## Features

- LightGBM churn classifier with stratified holdout ROC AUC and average precision.
- Global SHAP feature importance (model gain fallback if SHAP cannot initialize).
- Kaplan–Meier survival curve and Cox proportional hazards model; active customers are right-censored at observed tenure.
- Estimated remaining lifetime and discounted monthly-charge CLV proxy.
- K-means customer segments and a downloadable scored-customer CSV.
- Offer budget/ROI scenario simulator for the Telco sample, clearly labeled as an assumption-based scenario.
- Campaign uplift analysis using CausalML's T-learner on the randomized Hillstrom email experiment. The app compares one email treatment arm with the no-email control and reports observed lift for a top-ranked subgroup on a held-out randomized sample.
- Ranked campaign response recommendations and a campaign economics scenario based on the measured Hillstrom holdout lift.
- Optional PostgreSQL source-data load to `customer_source` using SQLAlchemy.
- IBM sample download or user-supplied IBM-compatible CSV.

## Run locally

Python 3.11 is recommended (also used by Streamlit Community Cloud).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Select **Load IBM Telco sample** in the sidebar. If the host cannot access the public dataset URL, download the CSV from the source link below and upload it in the sidebar.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository, set the main file to `app.py`, and deploy.
3. The app fetches the sample CSV at runtime. For an offline/private deployment, use **Upload CSV**.
4. PostgreSQL is optional. Configure `DATABASE_URL` in the app's Secrets when using it, for example:

```toml
DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require"
```

Never commit credentials. The database load replaces the `customer_source` table each time; use a dedicated demo database/schema.

## Data and modeling notes

Dataset: [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv), also distributed via the [Kaggle Telco Customer Churn listing](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It contains 7,043 records and a `Churn` outcome. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

The app derives average monthly spend, a service count, and a month-to-month contract flag. Complaints and usage trends are not present in the source and are not invented. Churn prediction uses a random stratified split; because the data has no event dates, the holdout is not a time-based validation. The Cox model estimates event duration from tenure and the churn indicator, with active customers censored at their observed tenure. The predicted lifetime is therefore best interpreted as exploratory.

The simulator computes `targeted customers × assumed response rate` for expected saves and compares their average predicted CLV with discount expense. It does not estimate individual treatment effects. For a valid uplift model, collect randomized offer assignment, offer details/cost, exposure date, and a defined post-offer retention outcome; evaluate uplift on a randomized holdout.

### Randomized campaign uplift

The **Campaign uplift** tab loads the [Hillstrom MineThatData email experiment](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv), credited to Kevin Hillstrom / MineThatData (2008). It contains 64,000 customers randomly assigned to a men's email, a women's email, or no email; the app compares one selected email arm with the no-email control. It fits a CausalML T-learner using pre-campaign features, then evaluates treatment-control response-rate difference for the highest-scored share on a stratified randomized holdout. The interval shown is an approximate 95% normal interval.

The benchmark outcomes are email visits and purchases. They are not telecom retention outcomes. The campaign economics panel scales the measured benchmark lift to a user-entered contact count, response value, and contact cost; these business inputs remain assumptions. To estimate retention-offer effects, replace the benchmark with an adequately powered randomized retention experiment and define a follow-up churn/retention outcome. Do not transfer the Hillstrom effect to telecom customers.

## Architecture

```mermaid
flowchart LR
  A[IBM sample CSV or upload] --> B[Pandas validation and features]
  B --> C[LightGBM churn model]
  B --> D[Kaplan–Meier and Cox survival]
  C --> E[SHAP global explanation]
  C --> F[Customer risk scores]
  D --> G[Remaining lifetime and CLV proxy]
  F --> H[K-means segments]
  F --> I[Offer scenario simulator]
  B --> J[(Optional PostgreSQL)]
  L[Hillstrom randomized campaign] --> M[CausalML T-learner]
  M --> N[Holdout subgroup lift and campaign economics]
  F --> K[Streamlit dashboard and CSV export]
  G --> K
  H --> K
  I --> K
  N --> K
```

## Repository layout

```text
app.py                 Streamlit user interface
src/pipeline.py        Telco preparation and modeling
src/uplift.py          Randomized experiment preparation and CausalML uplift
DATA_DICTIONARY.md     Source fields and derived fields
ARCHITECTURE.md        System flow and database notes
requirements.txt       Reproducible runtime dependencies
```

## Evaluation and demo checklist

- Verify the app displays the customer count, churn rate, validation metrics, SHAP/global drivers, survival and CLV estimates, segments, and offer scenarios.
- Load Hillstrom and verify treatment/control counts, holdout lift and interval, response-by-rank chart, economics scenario, and ranked-customer CSV.
- Save a short screen recording covering Overview → Drivers → Survival & value → Offer ROI.
- Holdout results observed in the deployed demo: ROC AUC **0.843** and average precision **0.649**. These are dataset-specific results from a stratified random holdout of the IBM Telco sample and should not be reported as production performance.
- Test the optional PostgreSQL action only with a disposable database; it replaces `customer_source`.

## Next steps for a real retention program

## Observed demo results

- Holdout ROC AUC: **0.843**
- Holdout average precision: **0.649**

These are results from a stratified random holdout of the IBM Telco sample. They are dataset-specific and should not be treated as production performance.

Add dated usage and complaint events, define a forward-looking churn label, validate using out-of-time cohorts, use margin and intervention costs in CLV, run a randomized retention-offer experiment (the included Hillstrom experiment is email marketing, not churn retention), and monitor calibration, drift, fairness, and campaign outcomes.
