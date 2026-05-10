# PACE

**P**roposal **A**ssembly for **C**ounterfactual **E**xplanations — a counterfactual explainer for tabular classifiers.

PACE generates counterfactual explanations — the smallest change to a data point that flips the model's prediction.  It works with any binary classifier that exposes a `predict_proba` method, handles mixed numeric/categorical features natively, and supports actionability constraints.

## Install

```bash
pip install pacex
```

## Quick start

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pace import PACE, FeatureInfo
from pace.preprocessing import make_preprocessor, feature_info_from_preprocessor

# 1. Prepare data
num_cols = ["age", "hours_per_week", "capital_gain"]
cat_cols = ["workclass", "education", "occupation"]

pre = make_preprocessor(num_cols, cat_cols)
X_tr_sc = pre.fit_transform(X_train_df)   # numpy array, model space
X_te_sc = pre.transform(X_test_df)
feature_info = feature_info_from_preprocessor(pre, num_cols, cat_cols)

# 2. Train your model (any sklearn-compatible classifier)
model = RandomForestClassifier().fit(X_tr_sc, y_train)

# 3. Fit PACE once per (dataset, model) pair
explainer = PACE(
    X_train=X_tr_sc,
    predict_proba=model.predict_proba,
    feature_info=feature_info,
    pre=pre,
    num_cols=num_cols,
    cat_cols=cat_cols,
)

# 4. Explain a single instance
x_f = X_te_sc[0]
x_cf, report = explainer.explain(x_f, y_desired=1)

print(report)
# {'status': 'ok', 'l0': 2, 'l2': 0.83, 'p_cf': 0.71, ...}

# 5. Decode back to human-readable values
print(explainer.decode(x_cf))
# {'age': 34.0, 'hours_per_week': 45.0, 'workclass': 'Private', ...}
```

## Constraints and immutability

```python
x_cf, report = explainer.explain(
    x_f,
    y_desired=1,
    n_candidates=1200,          # more candidates → better quality
    max_changed_features=3,     # change at most 3 features
    immutable=["age", "sex"],   # these must not change
    constraints={
        "hours_per_week": (None, 60),   # hours ≤ 60
        "capital_gain":   (0, None),    # capital_gain ≥ 0
    },
)
```

Constraints are specified in the **original feature space** (before scaling).

## Purely numeric data

If your data has no categorical columns, skip `feature_info`:

```python
explainer = PACE(X_train=X_tr_sc, predict_proba=model.predict_proba)
x_cf, report = explainer.explain(x_f, y_desired=1)
```


## License

Apache 2.0
