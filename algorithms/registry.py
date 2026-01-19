REGRESSION_ALGORITHMS = {
    "Linear Regression": {"category": "linear", "impl": "from_scratch"},
    "Ridge Regression": {"category": "linear", "impl": "from_scratch"},
    "Lasso Regression": {"category": "linear", "impl": "from_scratch"},
    "Elastic Net": {"category": "linear", "impl": "sklearn"},
    "SVR (RBF)": {"category": "kernel", "impl": "sklearn"},
    "Gaussian Process": {"category": "kernel", "impl": "sklearn"},
    "Decision Tree": {"category": "tree", "impl": "from_scratch"},
    "Random Forest": {"category": "ensemble", "impl": "sklearn"},
    "Gradient Boosting": {"category": "ensemble", "impl": "sklearn"},
    "XGBoost": {"category": "ensemble", "impl": "external"},
}

CLASSIFICATION_ALGORITHMS = {
    "Logistic Regression": {"category": "linear", "impl": "from_scratch"},
    "Naive Bayes": {"category": "probabilistic", "impl": "from_scratch"},
    "KNN": {"category": "distance", "impl": "from_scratch"},
    "Decision Tree": {"category": "tree", "impl": "from_scratch"},
    "SVM (RBF)": {"category": "kernel", "impl": "sklearn"},
    "Random Forest": {"category": "ensemble", "impl": "sklearn"},
    "Gradient Boosting": {"category": "ensemble", "impl": "sklearn"},
    "XGBoost": {"category": "ensemble", "impl": "external"},
}
