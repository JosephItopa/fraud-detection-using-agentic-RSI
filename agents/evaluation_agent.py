from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class EvaluationAgent:

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        pred = model.predict(X_test)

        prob = model.predict_proba(
            X_test
        )[:, 1]

        metrics = {

            "roc_auc":
            roc_auc_score(
                y_test,
                prob
            ),

            "pr_auc":
            average_precision_score(
                y_test,
                prob
            ),

            "precision":
            precision_score(
                y_test,
                pred
            ),

            "recall":
            recall_score(
                y_test,
                pred
            ),

            "f1":
            f1_score(
                y_test,
                pred
            ),

            "cm":
            confusion_matrix(
                y_test,
                pred
            )
        }

        return metrics