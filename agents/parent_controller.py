class ParentController:

    def __init__(self):

        self.best_score = 0
        self.best_pipeline = None

    def reward(
        self,
        current_auc,
        previous_auc,
        cost
    ):

        delta = current_auc - previous_auc

        reward = (
            100 * delta
            - 0.1 * cost
        )

        return reward

    def accept(
        self,
        score
    ):

        if score > self.best_score:

            self.best_score = score
            return True

        return False