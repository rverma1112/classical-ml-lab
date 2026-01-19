import numpy as np
from core.base.base_classifier import BaseClassifier


class LogisticRegression(BaseClassifier):
    """
    Logistic Regression implemented from scratch.

    Supports:
    - Binary classification
    - Pluggable optimizers
    """

    def __init__(
        self,
        fit_intercept=True,
        optimizer=None,
        n_iters=1000,
        threshold=0.5
    ):
        self.fit_intercept = fit_intercept
        self.optimizer = optimizer
        self.n_iters = n_iters
        self.threshold = threshold

        self.params = {}
        self.loss_history = []

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def _initialize_params(self, n_features):
        self.params["w"] = np.zeros(n_features)

    def _sigmoid(self, z):
        # numerical stability
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return self._sigmoid(X @ self.params["w"])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def _compute_loss(self, y_true, y_prob):
        """
        Binary cross-entropy loss
        """
        eps = 1e-10
        return -np.mean(
            y_true * np.log(y_prob + eps)
            + (1 - y_true) * np.log(1 - y_prob + eps)
        )

    def _compute_gradients(self, X, y):
        y_prob = self._sigmoid(X @ self.params["w"])
        error = y_prob - y
        grad_w = X.T @ error / X.shape[0]
        return {"w": grad_w}

    def fit(self, X, y):
        if self.optimizer is None:
            raise ValueError("Logistic Regression requires an optimizer")

        X = self._add_intercept(X)
        self._initialize_params(X.shape[1])

        for _ in range(self.n_iters):
            grads = self._compute_gradients(X, y)
            self.optimizer.step(self.params, grads)

            y_prob = self._sigmoid(X @ self.params["w"])
            loss = self._compute_loss(y, y_prob)
            self.loss_history.append(loss)

        return self

    def get_params(self):
        return {
            "fit_intercept": self.fit_intercept,
            "threshold": self.threshold,
            "optimizer": self.optimizer.get_params(),
            "n_iters": self.n_iters
        }
