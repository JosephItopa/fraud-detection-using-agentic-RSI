from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PreprocessingAgent:

    def run(self, df):

        X = df.drop(
            columns=["Class"]
        )

        y = df["Class"]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=42
            )
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )