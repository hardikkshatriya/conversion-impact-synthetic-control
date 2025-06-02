from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

def select_top_features(X, y, top_n=50):
    """
    Selects the top_n most important features using XGBoost feature importance.
    """
    model = XGBClassifier(
        learning_rate=0.01, max_depth=3, subsample=0.9,
        n_estimators=100, use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X, y)
    sfm = SelectFromModel(model, threshold=-float("inf"), max_features=top_n, prefit=True)
    return list(X.columns[sfm.get_support()])
