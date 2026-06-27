"""
Global objectives for the RSI system.
These objectives never change unless
a human changes them.
"""

OBJECTIVES = {

    # Performance objectives
    "target_roc_auc": 0.985,
    "target_pr_auc": 0.900,
    "target_recall": 0.950,

    # Improvement thresholds
    "min_improvement": 0.001,

    # Resource constraints
    "max_training_time": 600,
    "max_iterations": 100,

    # Stagnation criteria
    "patience": 10
}