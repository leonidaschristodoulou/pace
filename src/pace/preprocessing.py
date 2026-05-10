from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .core import FeatureInfo


def make_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    scale_num: bool = True,
    ohe_sparse: bool = False,
) -> ColumnTransformer:
    """
    Build a sklearn ``ColumnTransformer`` that:
    - imputes + (optionally) standardises numeric columns
    - imputes + one-hot encodes categorical columns

    Call ``pre.fit_transform(X_df)`` to get the model-space array, then pass
    ``pre`` to ``feature_info_from_preprocessor`` to build the matching
    ``FeatureInfo``.

    Parameters
    ----------
    num_cols : list of str
        Names of continuous columns in the input DataFrame.
    cat_cols : list of str
        Names of categorical columns in the input DataFrame.
    scale_num : bool
        Whether to apply ``StandardScaler`` to numeric columns (default True).
    ohe_sparse : bool
        Whether to return a sparse matrix from ``OneHotEncoder`` (default False).
    """
    transformers = []

    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_num:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=ohe_sparse)),
            ]),
            cat_cols,
        ))

    if not transformers:
        raise ValueError("num_cols and cat_cols are both empty — nothing to transform.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def feature_info_from_preprocessor(
    pre: ColumnTransformer,
    num_cols: List[str],
    cat_cols: List[str],
    *,
    immutable_num_names: Optional[List[str]] = None,
    immutable_cat_names: Optional[List[str]] = None,
) -> FeatureInfo:
    """
    Build a :class:`~pace.FeatureInfo` from a fitted ``ColumnTransformer``
    produced by :func:`make_preprocessor`.

    Parameters
    ----------
    pre : fitted ColumnTransformer
    num_cols : list of str
        Same list passed to ``make_preprocessor``.
    cat_cols : list of str
        Same list passed to ``make_preprocessor``.
    immutable_num_names : list of str, optional
        Numeric feature names that must not be changed.
    immutable_cat_names : list of str, optional
        Categorical feature names that must not be changed.
    """
    num_dim = len(num_cols) if "num" in pre.named_transformers_ else 0
    num_idx = np.arange(num_dim, dtype=int)

    cat_groups = []
    if "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        offset = num_dim
        for levels in ohe.categories_:
            g = np.arange(offset, offset + len(levels), dtype=int)
            cat_groups.append(g)
            offset += len(levels)

    immutable_num = _resolve_immutable_num(num_cols, num_idx, immutable_num_names or [])
    immutable_cat = _resolve_immutable_cat(cat_cols, immutable_cat_names or [])

    return FeatureInfo(
        num_idx=num_idx,
        cat_groups=cat_groups,
        immutable_num=immutable_num,
        immutable_cat=immutable_cat,
    )


def decode(
    x: np.ndarray,
    pre: ColumnTransformer,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> dict:
    """
    Decode a model-space vector back to human-readable feature values.

    Inverts the scaling for numeric columns and the one-hot encoding for
    categorical columns.  Imputation is not reversed (the original missing
    values are unknown).

    Parameters
    ----------
    x : 1-D or 2-D array (single instance)
    pre : fitted ColumnTransformer from ``make_preprocessor``
    num_cols : restrict output to these numeric columns (None = all)
    cat_cols : restrict output to these categorical columns (None = all)

    Returns
    -------
    dict mapping feature name → decoded value
    """
    if hasattr(x, "shape"):
        x = x.reshape(1, -1) if x.ndim == 1 else x
    else:
        x = np.asarray(x).reshape(1, -1)

    fitted_cols = {
        name: list(cols)
        for name, _, cols in pre.transformers_
        if name != "remainder"
    }

    out = {}
    offset = 0

    if "num" in pre.named_transformers_:
        num_cols_used = fitted_cols.get("num", [])
        num_dim = len(num_cols_used)
        X_num = np.asarray(x[:, offset: offset + num_dim])

        num_pipe = pre.named_transformers_["num"]
        scaler = num_pipe.named_steps.get("scaler")
        num_orig = scaler.inverse_transform(X_num)[0] if scaler is not None else X_num[0]

        keep = [c for c in num_cols_used if num_cols is None or c in set(num_cols)]
        if keep:
            idx_map = {c: i for i, c in enumerate(num_cols_used)}
            out.update({c: float(num_orig[idx_map[c]]) for c in keep})

        offset += num_dim

    if "cat" in pre.named_transformers_:
        cat_cols_used = fitted_cols.get("cat", [])
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_orig = ohe.inverse_transform(x[:, offset:])[0]

        keep = [c for c in cat_cols_used if cat_cols is None or c in set(cat_cols)]
        if keep:
            idx_map = {c: i for i, c in enumerate(cat_cols_used)}
            out.update({c: cat_orig[idx_map[c]] for c in keep})

    return out


# ── internal helpers ──────────────────────────────────────────────────────────

def _resolve_immutable_num(
    num_cols: List[str],
    num_idx: np.ndarray,
    immutable_names: List[str],
) -> frozenset:
    out = set()
    for name in immutable_names:
        if name in num_cols:
            out.add(int(num_idx[num_cols.index(name)]))
    return frozenset(out)


def _resolve_immutable_cat(
    cat_cols: List[str],
    immutable_names: List[str],
) -> frozenset:
    return frozenset(i for i, c in enumerate(cat_cols) if c in immutable_names)
