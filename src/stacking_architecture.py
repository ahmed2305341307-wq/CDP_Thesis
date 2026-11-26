"""Single and two-level stacking with stratified out-of-fold probabilities."""
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def build_single_stacking(rf, svm, xgb, random_state=42, n_splits=5):
    return StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        stack_method="predict_proba", passthrough=False, n_jobs=1,
    )


def build_hierarchical_stacking(rf, svm, xgb, random_state=42, n_splits=5):
    level1 = build_single_stacking(rf, svm, xgb, random_state, n_splits)
    return StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("xgb", xgb), ("level1", level1)],
        final_estimator=GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.05, max_depth=2,
            min_samples_leaf=5, subsample=0.8, random_state=random_state,
        ),
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        stack_method="predict_proba", passthrough=False, n_jobs=1,
    )


def build_soft_voting(rf, svm, xgb):
    return VotingClassifier(estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)], voting="soft", n_jobs=1)
