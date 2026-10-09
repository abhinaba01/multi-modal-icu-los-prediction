"""The four regression models from the paper, with DEFAULT hyperparameters.

Paper section 2.5: linear regression, SVM (RBF kernel), random forest, XGBoost.
No extra models, no tuning. Only non-defaults are RF ``ntree=1000`` and XGBoost
``nrounds=3000`` (from the reference R code).
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Ordered to match the paper's Table 4 / 6 rows.
REGRESSOR_NAMES = ["Linear regression", "SVM", "Random forest", "XGBoost"]


def regressor_factories(config: dict[str, Any], n_jobs: int = -1) -> dict[str, Callable[[], Any]]:
    """Return zero-arg factories that build a fresh regressor each call.

    Args:
        config: Parsed configuration.
        n_jobs: Threads for the tree models (``1`` when an outer loop is
            parallelized; ``-1`` uses all cores). Does not change RF results;
            XGB with ``n_jobs=1`` is the deterministic reference.

    Returns:
        Ordered mapping ``model_name -> factory()``.
    """
    seed = int(config["project"]["random_state"])
    rf_trees = int(config["models"]["rf_n_estimators"])
    xgb_rounds = int(config["models"]["xgb_n_estimators"])

    return {
        "Linear regression": lambda: LinearRegression(),
        # Reproduce e1071::svm's defaults faithfully:
        #   gamma = 1/p            -> gamma="auto" (not sklearn's "scale")
        #   scale = TRUE on x AND y -> StandardScaler on features (inner pipeline)
        #                              and on the target (TransformedTargetRegressor).
        # e1071 z-scores the target for regression by default; sklearn SVR does not,
        # which otherwise makes epsilon=0.1 mean 0.1 days instead of 0.1 SD of y.
        # cache_size only speeds up training (kernel-value cache); solution unchanged.
        "SVM": lambda: TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", gamma="auto", cache_size=1000)),
            transformer=StandardScaler()),
        # Match R randomForest's REGRESSION defaults (the reference used defaults):
        #   mtry = p/3   -> max_features=1/3   (sklearn default is all features)
        #   nodesize = 5 -> min_samples_leaf=5 (sklearn default is 1)
        # This is both more faithful to the GitLab code and much faster than
        # sklearn's full-feature, full-depth regression trees.
        "Random forest": lambda: RandomForestRegressor(
            n_estimators=rf_trees, max_features=1.0 / 3.0, min_samples_leaf=5,
            random_state=seed, n_jobs=n_jobs),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=xgb_rounds,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        ),
    }
