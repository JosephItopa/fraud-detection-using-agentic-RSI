class ProfilerAgent:

    def run(self, df):

        report = {

            "shape": df.shape,
            "missing": df.isnull().sum(),
            "duplicates": df.duplicated().sum(),
            "fraud_rate":
                df["Class"].mean()
        }

        return report