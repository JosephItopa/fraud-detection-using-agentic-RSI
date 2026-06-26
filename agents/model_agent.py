from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ModelAgent:

    def build(
        self,
        name
    ):

        models = {

            "rf":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42
            ),

            "xgb":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42
            ),

            "lgbm":
            LGBMClassifier(
                random_state=42
            )
        }

        return models[name]