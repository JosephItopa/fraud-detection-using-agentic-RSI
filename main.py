# main.py

import time
import pandas as pd

from environment.pipeline_environment import (
    PipelineEnvironment
)

from agents.parent_controller import (
    ParentController
)

from memory.replay_buffer import (
    ReplayBuffer
)

from pipelines.training import (
    train_and_evaluate
)


# LOAD DATA

df = pd.read_csv(
    "data/creditcard.csv"
)


# INITIALIZE RSI COMPONENTS

env = PipelineEnvironment()
controller = ParentController()
memory = ReplayBuffer()


# INITIAL PIPELINE

pipeline_config = {

    "model": "rf",

    "sampler": None,

    "feature_count": 20,

    "n_estimators": 300
}


# INITIAL STATE

metrics = {

    "roc_auc": 0,
    "pr_auc": 0,
    "f1": 0,
    "recall": 0
}

resources = {

    "training_time": 0
}

state = env.build_state(
    metrics,
    pipeline_config,
    resources
)


# RSI TRAINING LOOP

NUM_EPISODES = 50

for episode in range(NUM_EPISODES):

    print(
        f"\nEpisode {episode}"
    )

    # Observe state

    action = controller.choose_action(
        state
    )

    print(
        f"Action: {action}"
    )

    # Modify pipeline

    new_config = (
        controller.apply_action(
            action,
            pipeline_config
        )
    )


    # Train and evaluate

    start = time.time()

    metrics = train_and_evaluate(
        df,
        new_config
    )

    training_time = (
        time.time() - start
    )

    resources = {

        "training_time":
            training_time
    }


    # Build next state

    next_state = env.build_state(

        metrics,
        new_config,
        resources
    )

    # Compute reward

    reward = (
        controller.calculate_reward(
            state,
            next_state
        )
    )

    print(
        f"Reward: {reward:.3f}"
    )

    # Store experience

    memory.push(

        state,
        action,
        reward,
        next_state
    )

    # Update policy

    controller.update_q_table(

        state,
        action,
        reward,
        next_state
    )

    # Keep best pipeline

    if (
        next_state["roc_auc"]
        >
        controller.best_score
    ):

        controller.best_score = (
            next_state["roc_auc"]
        )

        controller.best_config = (
            new_config.copy()
        )

    # Recursive improvement

    state = next_state
    pipeline_config = new_config

# RESULTS

print("\nBest Pipeline")

print(
    controller.best_config
)

print(
    f"Best ROC AUC: "
    f"{controller.best_score:.4f}"
)