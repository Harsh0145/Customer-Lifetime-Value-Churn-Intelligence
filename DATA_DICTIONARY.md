# Data dictionary

The app expects the IBM Telco Customer Churn sample's core schema. User uploads need at least `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, and `Churn`.

| Field | Meaning | Use |
| --- | --- | --- |
| `customerID` | Sample customer identifier | Display/export only |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demographic/account attributes | Churn and survival covariates |
| `tenure` | Observed customer tenure in months | Churn feature and observed duration |
| `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Services subscribed | Churn features and derived count |
| `Contract` | Contract type | Churn feature and dashboard breakdown |
| `PaperlessBilling`, `PaymentMethod` | Billing attributes | Churn and survival covariates |
| `MonthlyCharges` | Current monthly charge | Churn feature and revenue proxy |
| `TotalCharges` | Accumulated charges | Numeric-cleaned; used for average spend |
| `Churn` | Yes/No observed outcome | Classifier label and survival event |

## Derived fields

| Field | Definition |
| --- | --- |
| `event` | 1 when `Churn == Yes`, else 0 |
| `duration` | `tenure`, with a one-month lower bound for survival fitting |
| `avg_monthly_spend` | `TotalCharges / tenure`; current monthly charge at zero tenure |
| `services_count` | Count of available service fields equal to `Yes` |
| `is_month_to_month` | 1 when `Contract` is `Month-to-month` |
| `churn_probability` | LightGBM probability from the fitted classifier |
| `expected_remaining_months` | Cox median survival estimate less observed tenure, bounded to 1–120 months; fallback uses Kaplan–Meier median |
| `predicted_clv` | Discounted value of monthly charges over estimated remaining months at 1% monthly discount |
| `segment` | K-means group from tenure, monthly charges, risk, and CLV |

The source has no complaint, dated usage, offer assignment, or treatment response fields. No causal uplift feature is derived from it.
