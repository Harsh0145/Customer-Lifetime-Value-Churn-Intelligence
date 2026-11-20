"""Randomized-campaign uplift analysis using CausalML's T-learner."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_MIRROR_URL = (
    "https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/"
    "main/data/raw/hillstrom.csv"
)
HILLSTROM_CREDIT = "Kevin Hillstrom / MineThatData, MineThatData E-Mail Analytics Challenge (2008)"
HILLSTROM_ARMS = {
    "Mens E-Mail": "Mens E-Mail",
    "Womens E-Mail": "Womens E-Mail",
}


def load_hillstrom(url: str = HILLSTROM_URL) -> pd.DataFrame:
    """Load the public 64k-row randomized email experiment."""
    last_error: Exception | None = None
    for source_url in dict.fromkeys([url, HILLSTROM_MIRROR_URL]):
        try:
            response = requests.get(source_url, timeout=45)
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content))
            required = {"segment", "visit", "conversion", "spend"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"Hillstrom file is missing columns: {', '.join(sorted(missing))}")
            return frame
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load Hillstrom from its source or mirror: {last_error}")


def _prepare_experiment(
    frame: pd.DataFrame, treatment_arm: str, outcome: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Keep one randomized email arm and the no-email control."""
    required = {"segment", outcome}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Campaign data is missing columns: {', '.join(sorted(missing))}")
    if treatment_arm not in HILLSTROM_ARMS:
        raise ValueError("Choose either the Mens E-Mail or Womens E-Mail experiment arm.")
    experiment = frame[frame["segment"].isin(["No E-Mail", treatment_arm])].copy()
    if outcome not in {"visit", "conversion"}:
        raise ValueError("The T-learner outcome must be the binary visit or conversion field.")
    experiment[outcome] = pd.to_numeric(experiment[outcome], errors="coerce")
    experiment = experiment.dropna(subset=[outcome])
    if not experiment[outcome].isin([0, 1]).all():
        raise ValueError(f"{outcome} must contain only binary 0/1 outcomes.")
    treatment = (experiment["segment"] == treatment_arm).astype(int)
    excluded = {"segment", "visit", "conversion", "spend", "customer_id", "customerid", "id"}
    feature_columns = [column for column in experiment.columns if column.lower() not in excluded]
    if not feature_columns:
        raise ValueError("No pre-campaign customer features were found.")
    features = experiment[feature_columns].copy()
    # Do not allow outcome leakage via any other post-campaign fields.
    return features, treatment, experiment[outcome].astype(int)


def _build_t_learner(features: pd.DataFrame):
    """Create a CausalML binary T-learner with a compact sklearn base model."""
    from causalml.inference.meta import BaseTClassifier

    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    estimator = Pipeline(
        [
            ("features", transformer),
            (
                "outcome_model",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    max_leaf_nodes=15,
                    min_samples_leaf=80,
                    l2_regularization=3.0,
                    random_state=42,
                ),
            ),
        ]
    )
    return BaseTClassifier(learner=estimator, control_name=0)


def fit_uplift(
    frame: pd.DataFrame,
    treatment_arm: str = "Mens E-Mail",
    outcome: str = "conversion",
    target_fraction: float = 0.2,
) -> dict:
    """Fit/evaluate a CausalML T-learner using an untouched randomized holdout.

    Held-out subgroup effects are evaluated by the randomized treatment-control
    difference within the subgroup ranked using training-only predictions.
    """
    features, treatment, y = _prepare_experiment(frame, treatment_arm, outcome)
    train_idx, test_idx = train_test_split(
        np.arange(len(features)),
        test_size=0.25,
        random_state=42,
        stratify=(treatment.astype(str) + "_" + y.astype(str)),
    )
    model = _build_t_learner(features)
    model.fit(features.iloc[train_idx], treatment.iloc[train_idx], y.iloc[train_idx])
    train_effect = np.asarray(model.predict(features.iloc[train_idx])).reshape(-1)
    holdout_effect = np.asarray(model.predict(features.iloc[test_idx])).reshape(-1)

    train_scores = pd.DataFrame(
        {"row_id": features.index.to_numpy()[train_idx], "estimated_uplift": train_effect}
    )
    holdout_scores = pd.DataFrame(
        {"row_id": features.index.to_numpy()[test_idx], "estimated_uplift": holdout_effect}
    )
    holdout_scores["treatment"] = treatment.iloc[test_idx].to_numpy()
    holdout_scores["outcome"] = y.iloc[test_idx].to_numpy()
    top_n = max(1, int(np.ceil(len(holdout_scores) * target_fraction)))
    ranked_holdout = holdout_scores.nlargest(top_n, "estimated_uplift").copy()

    treated = ranked_holdout.loc[ranked_holdout.treatment == 1, "outcome"]
    control = ranked_holdout.loc[ranked_holdout.treatment == 0, "outcome"]
    if treated.empty or control.empty:
        raise ValueError("The randomized holdout needs both treatment and control customers in the selected group.")
    uplift = float(treated.mean() - control.mean())
    standard_error = float(
        np.sqrt(
            treated.mean() * (1 - treated.mean()) / len(treated)
            + control.mean() * (1 - control.mean()) / len(control)
        )
    )

    all_treated = holdout_scores.loc[holdout_scores.treatment == 1, "outcome"]
    all_control = holdout_scores.loc[holdout_scores.treatment == 0, "outcome"]
    overall_uplift = float(all_treated.mean() - all_control.mean())
    overall_se = float(
        np.sqrt(
            all_treated.mean() * (1 - all_treated.mean()) / len(all_treated)
            + all_control.mean() * (1 - all_control.mean()) / len(all_control)
        )
    )

    # A randomized holdout uplift curve: compare observed treatment-control
    # response within successive score-ranked groups.
    ordered = holdout_scores.sort_values("estimated_uplift", ascending=False).reset_index(drop=True)
    ordered["rank_decile"] = pd.qcut(
        np.arange(len(ordered)), q=min(10, len(ordered)), labels=False, duplicates="drop"
    )
    groups = []
    for rank, group in ordered.groupby("rank_decile", sort=True):
        t = group.loc[group.treatment == 1, "outcome"]
        c = group.loc[group.treatment == 0, "outcome"]
        if len(t) and len(c):
            groups.append(
                {
                    "ranked_group": f"Top {(int(rank) + 1) * 10}%",
                    "customers": len(group),
                    "treatment_rate": float(t.mean()),
                    "control_rate": float(c.mean()),
                    "observed_uplift": float(t.mean() - c.mean()),
                }
            )
    groups_frame = pd.DataFrame(groups)

    recommendations = features.iloc[train_idx].copy()
    recommendations.insert(0, "source_row", features.index.to_numpy()[train_idx])
    recommendations["estimated_uplift"] = train_effect
    recommendations = recommendations.sort_values("estimated_uplift", ascending=False)

    return {
        "treatment_arm": treatment_arm,
        "outcome": outcome,
        "n_customers": len(features),
        "treatment_n": int(treatment.sum()),
        "control_n": int((treatment == 0).sum()),
        "train_n": len(train_idx),
        "holdout_n": len(test_idx),
        "target_fraction": target_fraction,
        "targeted_n": top_n,
        "targeted_uplift": uplift,
        "targeted_ci_low": uplift - 1.96 * standard_error,
        "targeted_ci_high": uplift + 1.96 * standard_error,
        "overall_uplift": overall_uplift,
        "overall_ci_low": overall_uplift - 1.96 * overall_se,
        "overall_ci_high": overall_uplift + 1.96 * overall_se,
        "holdout_curve": groups_frame,
        "recommended_customers": recommendations,
        "method": "CausalML T-learner; randomized test-set subgroup difference",
    }
