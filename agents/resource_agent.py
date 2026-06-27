import time

class ResourceAgent:

    def measure(
        self,
        start
    ):

        return time.time() - start
    
    def resource_limit_reached(
        self,
        state,
        objectives
    ):

        return (
            state["training_time"]
            >
            objectives["max_training_time"]
        )