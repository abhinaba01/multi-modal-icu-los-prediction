"""The four classification models from the paper, with DEFAULT hyperparameters.

Paper section 2.5 lists exactly four models: logistic regression, SVM (RBF
kernel), random forest, and XGBoost. No extra models, no hyperparameter tuning.
The only non-default values are the two counts the reference R code set
explicitly: random forest ``ntree=1000`` and XGBoost ``nrounds=3000``. The
classification "logistic regression" is sklearn ``LogisticRegression`` (see
README for why, vs the R ``lm()`` shortcut).
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Ordered to match the paper's Table 3 rows.
CLASSIFIER_NAMES = ["Logistic regression", "SVM", "Random forest", "XGBoost"]


def classifier_factories(config: dict[str, Any], n_jobs: int = -1) -> dict[str, Callable[[], Any]]:
    """Return zero-arg factories that build a fresh classifier each call.

    Fresh instances are needed because each repeated split refits from scratch.

    Args:
        config: Parsed configuration (``models``/``project`` blocks).
        n_jobs: Threads for the tree models. Pass ``1`` when an outer loop is
            parallelized (avoids oversubscription); ``-1`` uses all cores. Only
            affects RF/XGB parallelism, not their results (RF is invariant; XGB
            with ``n_jobs=1`` is the deterministic reference).

    Returns:
        Ordered mapping ``model_name -> factory()``.
    """
    seed = int(config["project"]["random_state"])
    rf_trees = int(config["models"]["rf_n_estimators"])
    xgb_rounds = int(config["models"]["xgb_n_estimators"])

    return {
        # penalty=None = unregularized logistic regression (glm-style), matching
        # what "logistic regression" means statistically. sklearn's default L2
        # (C=1) is a sklearn-specific regularization with no R counterpart.
        "Logistic regression": lambda: LogisticRegression(penalty=None, max_iter=1000, random_state=seed),
        # probability=False on purpose: the RBF SVM's hard predictions (accuracy,
        # F1, balanced acc, MCC) come from predict(), and AUC is scored from
        # decision_function (rank-identical to Platt probabilities). This matches
        # the reference R code, whose SVM ran without probability estimation (the
        # `probabiliy=TRUE` there was a typo, so probability was never enabled and
        # never fed the metrics), and it makes the SVM ~5-7x faster to fit.
        # gamma="auto" (= 1/n_features) matches e1071::svm's default gamma=1/p
        # (NOT sklearn's default gamma="scale"). The StandardScaler reproduces
        # e1071's scale=TRUE default (it z-scores features internally before
        # fitting); sklearn's SVC does no internal scaling.
        # cache_size only speeds up training (kernel-value cache); the solution is
        # unchanged -- a faithfulness-neutral optimization for the SVM bottleneck.
        "SVM": lambda: make_pipeline(
            StandardScaler(), SVC(kernel="rbf", gamma="auto", cache_size=1000, random_state=seed)),
        "Random forest": lambda: RandomForestClassifier(n_estimators=rf_trees, random_state=seed, n_jobs=n_jobs),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=xgb_rounds,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        ),
    }
