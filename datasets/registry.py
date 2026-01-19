"""
Dataset registry for Classical ML Lab.
This file defines all supported datasets and their metadata.
"""

DATASET_REGISTRY = {
    # --------------------
    # REGRESSION DATASETS
    # --------------------
    "california_housing": {
        "task": "regression",
        "source": "sklearn",
        "description": "California housing prices dataset",
        "features": "numerical",
        "n_samples": 20640,
        "target": "median_house_value"
    },

    "diabetes": {
        "task": "regression",
        "source": "sklearn",
        "description": "Diabetes progression dataset",
        "features": "numerical",
        "n_samples": 442,
        "target": "disease_progression"
    },

    # -----------------------
    # CLASSIFICATION DATASETS
    # -----------------------
    "iris": {
        "task": "classification",
        "source": "sklearn",
        "description": "Iris flower classification dataset",
        "features": "numerical",
        "n_samples": 150,
        "n_classes": 3,
        "target": "species"
    },

    "breast_cancer": {
        "task": "classification",
        "source": "sklearn",
        "description": "Breast cancer diagnostic dataset",
        "features": "numerical",
        "n_samples": 569,
        "n_classes": 2,
        "target": "diagnosis"
    }
}
