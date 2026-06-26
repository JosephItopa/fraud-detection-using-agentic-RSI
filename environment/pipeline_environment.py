class PipelineEnvironment:

    def build_state(
        self,
        metrics,
        pipeline_config,
        resource_metrics
    ):

        state = {

            "roc_auc":
                metrics["roc_auc"],

            "pr_auc":
                metrics["pr_auc"],

            "f1":
                metrics["f1"],

            "recall":
                metrics["recall"],

            "model":
                pipeline_config["model"],

            "sampler":
                pipeline_config["sampler"],

            "feature_count":
                pipeline_config["feature_count"],

            "training_time":
                resource_metrics["time"]
        }

        return state