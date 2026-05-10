from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.compose import ColumnTransformer

from .core import (
    FeatureInfo,
    StageAArtifacts,
    stageA_build,
    stageA_target_anchors_from_p,
    stageB_generate,
    stageC_select_best,
)
from .preprocessing import (
    _resolve_immutable_cat,
    _resolve_immutable_num,
    decode as _decode,
)


class PACE:
    """
    PACE counterfactual explainer.

    Workflow
    --------
    1. **Fit once** per (dataset, model) pair::

        explainer = PACE(
            X_train=X_tr_sc,
            predict_proba=model.predict_proba,
            feature_info=feature_info,  # from feature_info_from_preprocessor()
            pre=pre,                     # optional — needed for constraints + decode
            num_cols=num_cols,           # optional
            cat_cols=cat_cols,           # optional
        )

    2. **Explain** individual instances::

        x_cf, report = explainer.explain(x_f, y_desired=1)

    3. **Decode** back to original feature space (requires ``pre``)::

        explainer.decode(x_cf)

    Parameters
    ----------
    X_train : array of shape (n, d)
        Training data in the preprocessed (model) space.
    predict_proba : callable
        A function ``f(X) -> probabilities``.  Accepts both sklearn-style
        ``(n, 2)`` output and 1-D ``(n,)`` output (class-1 probability).
    feature_info : FeatureInfo, optional
        Categorical structure of the feature space.  Pass ``None`` for
        purely numeric data.
    pre : fitted ColumnTransformer, optional
        The preprocessor used to produce ``X_train``.  Required for named
        feature constraints and :meth:`decode`.
    num_cols : list of str, optional
        Numeric column names, in the same order passed to the preprocessor.
    cat_cols : list of str, optional
        Categorical column names, in the same order passed to the preprocessor.
    boundary_k : int
        Number of near-boundary training instances to keep in the pool (Stage A).
    anchor_k : int
        Number of near-boundary instances on each target class side (Stage A).
    perm_repeats : int
        Permutation repeats for feature importance estimation (Stage A).
    random_state : int
        Master random seed.
    """

    def __init__(
        self,
        *,
        X_train: np.ndarray,
        predict_proba: Callable,
        feature_info: Optional[FeatureInfo] = None,
        pre: Optional[ColumnTransformer] = None,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        boundary_k: int = 200,
        anchor_k: int = 200,
        perm_repeats: int = 8,
        random_state: int = 0,
    ) -> None:
        self._X_train = np.asarray(X_train)
        self._predict_proba = _wrap_predict_proba(predict_proba)
        self._feature_info = feature_info
        self._pre = pre
        self._num_cols = list(num_cols) if num_cols is not None else None
        self._cat_cols = list(cat_cols) if cat_cols is not None else None
        self._random_state = random_state

        self._stageA: StageAArtifacts = stageA_build(
            X_train=self._X_train,
            predict_proba_fn=self._predict_proba,
            boundary_k=boundary_k,
            perm_repeats=perm_repeats,
            feature_info=feature_info,
            random_state=random_state,
        )

        p_train = self._predict_proba(self._X_train).astype(np.float64)
        self._anchors: Dict[int, np.ndarray] = {}
        for y in (0, 1):
            anchors, _ = stageA_target_anchors_from_p(
                p_train=p_train,
                y_desired=y,
                anchor_k=anchor_k,
                random_state=random_state + 1,
            )
            self._anchors[y] = anchors

    # ── public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        x_f: np.ndarray,
        y_desired: int,
        *,
        n_candidates: int = 800,
        max_changed_features: int = 3,
        immutable: Optional[List[Union[str, int]]] = None,
        constraints: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        random_state: int = 0,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Generate a counterfactual explanation for a single instance.

        Parameters
        ----------
        x_f : 1-D array
            Factual instance in the preprocessed (model) space.
        y_desired : int
            Target class (0 or 1).
        n_candidates : int
            How many candidate counterfactuals to generate.  Higher values
            improve quality at the cost of runtime.
        max_changed_features : int
            Maximum number of features (or categorical groups) that may differ
            from ``x_f``.
        immutable : list of str or int, optional
            Features that must not change.  Strings are matched against
            ``num_cols`` / ``cat_cols`` (requires those to be set).  Integers
            are treated as column indices in the model space.
        constraints : dict, optional
            Range constraints in the **original feature space**.  Keys are
            feature names from ``num_cols``.  Values are ``(lo, hi)`` tuples
            where either bound can be ``None`` (unbounded).  Example::

                constraints={"age": (18, None), "hours_per_week": (None, 60)}

            Requires ``pre`` and ``num_cols`` to be set on the explainer.
        random_state : int
            Seed for candidate generation.

        Returns
        -------
        x_cf : ndarray or None
            Best counterfactual in model space, or ``None`` if none found.
        report : dict
            Diagnostics: ``status``, ``l0``, ``l2``, ``p_cf``, ``n_flip``, etc.
        """
        feature_info = self._apply_immutables(immutable)

        C, meta = stageB_generate(
            x_f=x_f,
            predict_proba_fn=self._predict_proba,
            X_train=self._X_train,
            boundary_idx=self._stageA.boundary_idx,
            anchors_idx=self._anchors.get(y_desired, np.array([], dtype=int)),
            feature_weights=self._stageA.feature_weights,
            unit_weights=self._stageA.unit_weights,
            feature_info=feature_info,
            n_candidates=n_candidates,
            max_changed_features=max_changed_features,
            random_state=random_state,
        )

        if constraints:
            mask = self._constraint_mask(C, constraints)
            C = C[mask]
            meta["sources"] = meta["sources"][mask]

        x_cf, report = stageC_select_best(
            x_f=x_f,
            C=C,
            predict_proba_fn=self._predict_proba,
            y_desired=y_desired,
            feature_info=feature_info,
            sources=meta["sources"],
        )

        return x_cf, report

    def decode(
        self,
        x_cf: np.ndarray,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
    ) -> dict:
        """
        Decode a model-space counterfactual back to human-readable values.

        Requires ``pre`` to have been provided at construction time.

        Parameters
        ----------
        x_cf : 1-D array (model space)
        num_cols : restrict output to these numeric columns (None = all)
        cat_cols : restrict output to these categorical columns (None = all)

        Returns
        -------
        dict mapping feature name → decoded value
        """
        if self._pre is None:
            raise ValueError(
                "decode() requires a fitted preprocessor. "
                "Pass pre=... when constructing PACE."
            )
        return _decode(x_cf, self._pre, num_cols, cat_cols)

    # ── internals ─────────────────────────────────────────────────────────────

    def _apply_immutables(
        self, immutable: Optional[List[Union[str, int]]]
    ) -> Optional[FeatureInfo]:
        """Return a FeatureInfo with immutables merged in (or the stored one)."""
        if not immutable:
            return self._feature_info

        add_num: set = set()
        add_cat: set = set()

        for item in immutable:
            if isinstance(item, str):
                if self._num_cols is None and self._cat_cols is None:
                    raise ValueError(
                        f"Cannot resolve immutable feature name '{item}': "
                        "num_cols and cat_cols were not provided to PACE()."
                    )
                if self._num_cols and item in self._num_cols:
                    fi = self._feature_info
                    if fi is not None and fi.num_idx is not None:
                        add_num.add(int(fi.num_idx[self._num_cols.index(item)]))
                    else:
                        add_num.add(self._num_cols.index(item))
                elif self._cat_cols and item in self._cat_cols:
                    add_cat.add(self._cat_cols.index(item))
                else:
                    raise ValueError(
                        f"Immutable feature '{item}' not found in num_cols or cat_cols."
                    )
            elif isinstance(item, (int, np.integer)):
                add_num.add(int(item))
            else:
                raise TypeError(f"immutable entries must be str or int, got {type(item)}")

        if self._feature_info is not None:
            fi = self._feature_info
            return FeatureInfo(
                cat_groups=fi.cat_groups,
                num_idx=fi.num_idx,
                immutable_num=fi.immutable_num | frozenset(add_num),
                immutable_cat=fi.immutable_cat | frozenset(add_cat),
            )
        else:
            d = self._X_train.shape[1]
            return FeatureInfo(
                cat_groups=[],
                num_idx=np.arange(d, dtype=int),
                immutable_num=frozenset(add_num),
                immutable_cat=frozenset(),
            )

    def _constraint_mask(
        self,
        C: np.ndarray,
        constraints: Dict[str, Tuple[Optional[float], Optional[float]]],
    ) -> np.ndarray:
        """Boolean mask of candidates satisfying all range constraints."""
        mask = np.ones(len(C), dtype=bool)

        if self._pre is None or self._num_cols is None:
            raise ValueError(
                "Range constraints require pre and num_cols to be set on PACE()."
            )

        num_pipe = self._pre.named_transformers_.get("num")
        if num_pipe is None:
            return mask

        n_num = len(self._num_cols)
        C_num_sc = C[:, :n_num]

        scaler = num_pipe.named_steps.get("scaler")
        C_num_orig = (
            scaler.inverse_transform(C_num_sc) if scaler is not None else C_num_sc
        )

        for feat, bounds in constraints.items():
            if feat not in self._num_cols:
                continue
            col_i = self._num_cols.index(feat)
            lo, hi = bounds
            vals = C_num_orig[:, col_i]
            if lo is not None:
                mask &= vals >= lo
            if hi is not None:
                mask &= vals <= hi

        return mask


# ── helpers ───────────────────────────────────────────────────────────────────

def _wrap_predict_proba(fn: Callable) -> Callable:
    """Accept sklearn-style (n, 2) or 1-D (n,) predict_proba output."""
    def wrapper(X: np.ndarray) -> np.ndarray:
        out = np.asarray(fn(X))
        return out[:, 1] if out.ndim == 2 else out
    return wrapper
