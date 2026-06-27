# agents/parent_controller.py

import random
import numpy as np


class ParentController:
    """
    Meta-controller responsible for recursively
    improving the ML pipeline.
    """

    ACTION_SPACE = [

        "use_rf",
        "use_xgb",
        "use_lgbm",

        "apply_smote",
        "remove_smote",

        "increase_features",
        "decrease_features",

        "increase_estimators",
        "decrease_estimators"
    ]

    def __init__(self):

        # Exploration probability
        self.epsilon = 0.20

        # Q-table
        self.q_table = {}

        # Best pipeline discovered so far
        self.best_config = None
        self.best_score = 0
        self.no_improvement_count = 0

    # STATE ENCODING

    def encode_state(self, state):
        """
        Convert dictionary state into a tuple so it
        can be stored inside a Q-table.
        """

        return (

            round(state["roc_auc"], 2),
            round(state["pr_auc"], 2),
            round(state["recall"], 2),

            state["model"],
            state["sampler"],
            state["feature_count"]
        )

    
    # ACTION SELECTION

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """

        encoded_state = self.encode_state(state)

        if encoded_state not in self.q_table:

            self.q_table[encoded_state] = {

                action: 0.0
                for action in self.ACTION_SPACE
            }

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(
                self.ACTION_SPACE
            )

        # Exploitation
        action_values = self.q_table[
            encoded_state
        ]

        return max(
            action_values,
            key=action_values.get
        )

    
    # PIPELINE SELF-MODIFICATION

    def apply_action(
        self,
        action,
        config
    ):
        """
        Modify the current pipeline configuration.
        """

        config = config.copy()

        if action == "use_rf":
            config["model"] = "rf"

        elif action == "use_xgb":
            config["model"] = "xgb"

        elif action == "use_lgbm":
            config["model"] = "lgbm"

        elif action == "apply_smote":
            config["sampler"] = "smote"

        elif action == "remove_smote":
            config["sampler"] = None

        elif action == "increase_features":
            config["feature_count"] += 5

        elif action == "decrease_features":
            config["feature_count"] = max(
                5,
                config["feature_count"] - 5
            )

        elif action == "increase_estimators":
            config["n_estimators"] += 100

        elif action == "decrease_estimators":
            config["n_estimators"] = max(
                100,
                config["n_estimators"] - 100
            )

        return config

    
    # REWARD FUNCTION

    def calculate_reward(
        self,
        old_state,
        new_state
    ):
        """
        Multi-objective reward.

        Reward good performance
        Penalize excessive computation.
        """

        delta_auc = (
            new_state["roc_auc"]
            -
            old_state["roc_auc"]
        )

        delta_pr = (
            new_state["pr_auc"]
            -
            old_state["pr_auc"]
        )

        delta_recall = (
            new_state["recall"]
            -
            old_state["recall"]
        )

        cost_penalty = (
            0.05
            *
            new_state["training_time"]
        )

        reward = (

            100 * delta_auc +
            50 * delta_pr +
            50 * delta_recall -
            cost_penalty
        )

        return reward

    # Q-LEARNING UPDATE

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
        alpha=0.10,
        gamma=0.95
    ):

        state = self.encode_state(state)
        next_state = self.encode_state(
            next_state
        )

        if next_state not in self.q_table:

            self.q_table[next_state] = {

                action: 0.0
                for action
                in self.ACTION_SPACE
            }

        current_q = self.q_table[
            state
        ][action]

        max_next_q = max(
            self.q_table[next_state].values()
        )

        new_q = current_q + alpha * (

            reward +
            gamma * max_next_q -
            current_q
        )

        self.q_table[
            state
        ][action] = new_q


def objective_achieved(
    self,
    state,
    objectives
):

    return (

        state["roc_auc"]
        >= objectives["target_roc_auc"]

        and

        state["pr_auc"]
        >= objectives["target_pr_auc"]

        and

        state["recall"]
        >= objectives["target_recall"]
    )

def update_progress(
    self,
    score,
    objectives
):

    improvement = (
        score -
        self.best_score
    )

    if improvement > objectives[
        "min_improvement"
    ]:

        self.best_score = score
        self.no_improvement_count = 0

    else:
        self.no_improvement_count += 1


def needs_human_direction(
    self,
    objectives
):

    return (
        self.no_improvement_count
        >=
        objectives["patience"]
    )

