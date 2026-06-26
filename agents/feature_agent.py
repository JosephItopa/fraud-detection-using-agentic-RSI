from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)

class FeatureAgent:

    def run(
        self,
        X_train,
        y_train,
        X_test,
        k=20
    ):

        selector = SelectKBest(
            mutual_info_classif,
            k=k
        )

        X_train = selector.fit_transform(
            X_train,
            y_train
        )

        X_test = selector.transform(
            X_test
        )

        return X_train, X_test