import numpy as np
from core.base.base_regressor import BaseRegressor


class LinearRegression(BaseRegressor):
    """
    Linear Regression implemented from scratch.

    Supports:
    - Normal Equation
    - Gradient-based optimization via pluggable optimizers
    """

    def __init__(
        self,
        fit_intercept=True,
        optimizer=None,
        n_iters=1000
    ):
        self.fit_intercept = fit_intercept
        self.optimizer = optimizer
        self.n_iters = n_iters

        self.params = {}
        self.loss_history = []

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def _initialize_params(self, n_features):
        self.params["w"] = np.zeros(n_features)

    def predict(self, X):
        X = self._add_intercept(X)
        return X @ self.params["w"]

    def _compute_loss(self, y_true, y_pred):
        """
        Mean Squared Error loss (scaled by 1/2 for convenience)
        """
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def _compute_gradients(self, X, y):
        y_pred = X @ self.params["w"]
        error = y_pred - y
        grad_w = X.T @ error / X.shape[0]
        return {"w": grad_w}

    def fit(self, X, y):
        X = self._add_intercept(X)
        self._initialize_params(X.shape[1])

        # -------------------------
        # Case 1: Normal Equation
        # -------------------------
        if self.optimizer is None:
            XtX = X.T @ X
            Xty = X.T @ y
            self.params["w"] = np.linalg.pinv(XtX) @ Xty
            return self

        # -------------------------
        # Case 2: Optimizer-based
        # -------------------------
        for _ in range(self.n_iters):
            grads = self._compute_gradients(X, y)
            self.optimizer.step(self.params, grads)

            loss = self._compute_loss(y, X @ self.params["w"])
            self.loss_history.append(loss)

        return self

    def get_params(self):
        return {
            "fit_intercept": self.fit_intercept,
            "optimizer": None if self.optimizer is None else self.optimizer.get_params(),
            "n_iters": self.n_iters
        }
