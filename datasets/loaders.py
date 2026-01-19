from sklearn import datasets
from .registry import DATASET_REGISTRY


def load_dataset(name: str):
    """
    Load a dataset by name.

    Returns
    -------
    X : ndarray
    y : ndarray
    metadata : dict
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found in registry")

    info = DATASET_REGISTRY[name]

    if info["source"] == "sklearn":
        return _load_from_sklearn(name, info)

    raise NotImplementedError(f"Source {info['source']} not supported yet")


def _load_from_sklearn(name, info):
    if name == "california_housing":
        data = datasets.fetch_california_housing(as_frame=False)
        X, y = data.data, data.target

    elif name == "diabetes":
        data = datasets.load_diabetes()
        X, y = data.data, data.target

    elif name == "iris":
        data = datasets.load_iris()
        X, y = data.data, data.target

    elif name == "breast_cancer":
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target

    else:
        raise ValueError(f"Unhandled sklearn dataset: {name}")

    return X, y, info
