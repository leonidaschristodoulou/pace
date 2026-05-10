"""
PACE — Proposal Assembly for Counterfactual Explanations.

Quick start
-----------
>>> from pace import PACE, FeatureInfo
>>> from pace.preprocessing import make_preprocessor, feature_info_from_preprocessor
>>>
>>> pre = make_preprocessor(num_cols, cat_cols)
>>> X_tr_sc = pre.fit_transform(X_train_df)
>>> feature_info = feature_info_from_preprocessor(pre, num_cols, cat_cols)
>>>
>>> explainer = PACE(
...     X_train=X_tr_sc,
...     predict_proba=model.predict_proba,
...     feature_info=feature_info,
...     pre=pre, num_cols=num_cols, cat_cols=cat_cols,
... )
>>> x_cf, report = explainer.explain(x_f, y_desired=1)
>>> explainer.decode(x_cf)
"""

from .core import FeatureInfo, StageAArtifacts
from .explainer import PACE

__version__ = "0.1.0"
__all__ = ["PACE", "FeatureInfo", "StageAArtifacts"]
