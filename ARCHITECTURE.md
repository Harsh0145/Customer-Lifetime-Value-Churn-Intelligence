# Architecture and data documentation

See the Mermaid architecture diagram in [README.md](README.md#architecture).

## Runtime flow

1. Read the IBM CSV from its public GitHub mirror or parse an uploaded CSV.
2. Validate core columns, convert numeric billing fields, impute blank `TotalCharges`, and derive simple service/account features.
3. Fit the churn classifier and a 25% stratified holdout metric, then fit survival and clustering models.
4. Display risk, explanations, survival, CLV proxy, segments, and scenario results in Streamlit.
5. On explicit user action, load the separate Hillstrom randomized email experiment, compare one treatment arm with the no-email control, fit a CausalML T-learner, and evaluate uplift in a randomized holdout.
6. On explicit user action, optionally replace the PostgreSQL `customer_source` table with Telco source rows.

## Campaign uplift data

`src/uplift.py` loads the 64,000-row Hillstrom MineThatData email experiment. Each analysis compares one of the two email arms with the no-email control. Pre-campaign customer fields feed the learner; the binary visit or conversion outcome is held out from the features. Reported subgroup lift is calculated from randomized test customers and is separate from the telecom churn model.

## PostgreSQL table

`customer_source` is created/replaced from the uploaded source dataframe by Pandas `to_sql`. The exact source columns depend on the uploaded CSV. Numeric-looking fields are inferred by Pandas/SQLAlchemy; the sample contains `customerID` as a string primary business identifier, but the demo table does not impose a primary key. The app does not store predictions or credentials in PostgreSQL.

Use a least-privilege database account with permission to create/drop the demo table in a dedicated schema. A production implementation should use migrations, stable types, a primary key, transactional upserts, access controls, and an audited retention policy.
