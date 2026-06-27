# environment/pipeline_environment.py

"""
The environment converts the current pipeline configuration and
evaluation metrics into a state representation that the RSI
controller can reason about.

State = representation of the current ML pipeline.
"""

class PipelineEnvironment:

    def __init__(self):
        pass

    def build_state(
        self,
        metrics,
        pipeline_config,
        resource_metrics
    ):
        """
        Construct the state dictionary.

        Parameters
        ----------
        metrics : dict
            Evaluation metrics.

        pipeline_config : dict
            Current pipeline settings.

        resource_metrics : dict
            Computational metrics.

        Returns
        -------
        dict
            State representation.
        """

        state = {

            # Model performance
            "roc_auc":
                metrics.get("roc_auc", 0),

            "pr_auc":
                metrics.get("pr_auc", 0),

            "f1":
                metrics.get("f1", 0),

            "recall":
                metrics.get("recall", 0),

            # Pipeline configuration
            "model":
                pipeline_config["model"],

            "sampler":
                pipeline_config["sampler"],

            "feature_count":
                pipeline_config["feature_count"],

            # Resource information
            "training_time":
                resource_metrics["training_time"]
        }

        return state