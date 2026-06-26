import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

class TuningAgent:

    def optimize(
        self,
        X,
        y
    ):

        def objective(trial):

            model = XGBClassifier(

                max_depth=
                trial.suggest_int(
                    "max_depth",
                    3,
                    12
                ),

                learning_rate=
                trial.suggest_float(
                    "learning_rate",
                    0.001,
                    0.3
                ),

                n_estimators=
                trial.suggest_int(
                    "n_estimators",
                    100,
                    1000
                ),

                eval_metric="logloss"
            )

            score = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="roc_auc"
            ).mean()

            return score

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            objective,
            n_trials=20
        )

        return study.best_params