"""Data preparation, churn, survival, CLV, and scenario modeling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TARGET = "Churn"
ID = "customerID"


def load_dataset(url: str = DATA_URL) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    from io import BytesIO
    return pd.read_csv(BytesIO(response.content))


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = {ID, "tenure", "MonthlyCharges", "TotalCharges", TARGET, "Contract"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = raw.copy()
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"].fillna(0))
    df["Churn"] = df["Churn"].astype(str).str.strip().str.title()
    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn values must be Yes or No.")
    df["event"] = (df["Churn"] == "Yes").astype(int)
    # In this cross-sectional sample, tenure is the observed event/censor time.
    df["duration"] = df["tenure"].clip(lower=1)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    df["services_count"] = df[[c for c in ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"] if c in df]].eq("Yes").sum(axis=1)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def _cox_fit(df: pd.DataFrame) -> tuple[CoxPHFitter | None, KaplanMeierFitter]:
    km = KaplanMeierFitter().fit(df["duration"], event_observed=df["event"], label="Customer survival")
    covariates = [c for c in ["MonthlyCharges", "SeniorCitizen", "Partner", "Dependents", "Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "services_count"] if c in df.columns]
    cox_df = df[["duration", "event", *covariates]].copy()
    categorical = cox_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cox_df = pd.get_dummies(cox_df, columns=categorical, drop_first=True, dtype=float)
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    try:
        cox = CoxPHFitter(penalizer=0.1)
        cox.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)
        return cox, km
    except Exception:
        return None, km


def model_bundle(df: pd.DataFrame) -> dict:
    # Deliberately fit a reproducible baseline model on each app session's data.
    excluded = {ID, TARGET, "event", "duration", "TotalCharges"}
    features = [c for c in df.columns if c not in excluded]
    X, y = df[features], df["event"]
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical),
    ])
    # LightGBM is preferred; sklearn fallback keeps the demo usable on unsupported platforms.
    try:
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=180, learning_rate=0.04, num_leaves=20, max_depth=-1, class_weight="balanced", verbosity=-1, random_state=42, n_jobs=1)
    except ImportError:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier = Pipeline([("pre", pre), ("model", estimator)])
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob) if y_test.nunique() == 2 else np.nan
    ap = average_precision_score(y_test, prob) if y_test.nunique() == 2 else np.nan
    model = classifier.named_steps["model"]
    names = classifier.named_steps["pre"].get_feature_names_out()
    shap_importance = None
    try:
        import shap
        # Tree SHAP where supported. Limit rows to keep hosted app memory predictable.
        matrix = classifier.named_steps["pre"].transform(X.sample(min(500, len(X)), random_state=42))
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(matrix)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        shap_importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        pass
    if shap_importance is None:
        imp = getattr(model, "feature_importances_", np.zeros(len(names)))
        if len(imp) != len(names):
            imp = np.zeros(len(names))
        importance = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    else:
        importance = shap_importance.rename(columns={"mean_abs_shap": "importance"})
    cox, km = _cox_fit(df)
    return {"classifier": classifier, "features": features, "auc": auc, "average_precision": ap, "importance": importance, "shap_importance": shap_importance, "cox": cox, "km": km}


def score_customers(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    result = df.copy()
    result["churn_probability"] = bundle["classifier"].predict_proba(result[bundle["features"]])[:, 1]
    cox = bundle.get("cox")
    if cox is not None:
        covariates = [c for c in ["MonthlyCharges", "SeniorCitizen", "Partner", "Dependents", "Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "services_count"] if c in result.columns]
        temp = result[["duration", *covariates]].copy()
        categorical = temp.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        temp = pd.get_dummies(temp, columns=categorical, drop_first=True, dtype=float).drop(columns=["duration"])
        temp = temp.reindex(columns=cox.params_.index, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
        try:
            surv = cox.predict_median(temp).replace([np.inf, -np.inf], np.nan)
            remaining = (surv - result["tenure"]).clip(lower=1).fillna(12)
            result["expected_remaining_months"] = remaining.clip(upper=120)
        except Exception:
            result["expected_remaining_months"] = 12.0
    else:
        # A transparent group-level Kaplan–Meier fallback for environments where Cox fails.
        km = bundle["km"]
        surv_median = km.median_survival_time_
        result["expected_remaining_months"] = (float(surv_median) - result["tenure"]).clip(lower=1, upper=120) if np.isfinite(surv_median) else 12.0
    result["expected_remaining_months"] = pd.to_numeric(result["expected_remaining_months"], errors="coerce").fillna(12).clip(1, 120)
    result["predicted_clv"] = result["MonthlyCharges"] * (1 - (1 + 0.01) ** (-result["expected_remaining_months"])) / 0.01
    numeric = result[["MonthlyCharges", "tenure", "churn_probability", "predicted_clv"]].fillna(0)
    if len(result) >= 4:
        labels = KMeans(n_clusters=min(4, max(2, len(result) // 2)), random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(numeric))
        result["segment"] = pd.Series(labels, index=result.index).map(lambda x: f"Segment {x + 1}")
    else:
        result["segment"] = "Segment 1"
    return result


def uplift_simulation(cohort: pd.DataFrame, discount: float, duration: int, effectiveness: float) -> dict[str, float]:
    n = len(cohort)
    saves = n * float(effectiveness)
    cost = n * float(discount) * int(duration)
    revenue = float(cohort["predicted_clv"].sum() / n * saves) if n else 0.0
    net = revenue - cost
    return {"expected_saves": saves, "cost": cost, "retained_revenue": revenue, "net_value": net, "roi": net / cost if cost else 0.0}


def write_postgres(df: pd.DataFrame, connection_url: str) -> None:
    from sqlalchemy import create_engine
    engine = create_engine(connection_url, pool_pre_ping=True)
    try:
        df.to_sql("customer_source", engine, if_exists="replace", index=False, chunksize=1000, method="multi")
    finally:
        engine.dispose()
