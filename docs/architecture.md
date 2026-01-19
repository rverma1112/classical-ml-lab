# Classical ML Lab — Architecture & Rules

## Purpose
A comprehensive, interview-grade classical machine learning lab with:
- From-scratch implementations
- Sklearn wrappers
- Experiment tracking
- Streamlit demo UI

## Non-Negotiable Rules
1. No ML logic inside Streamlit UI
2. All models must implement a common interface
3. Deterministic behavior (random_state everywhere)
4. From-scratch and sklearn code must be separated
5. Experiments are immutable once saved

## Layer Responsibilities
- datasets/: data loading and validation only
- core/: base classes, preprocessing, utilities
- algorithms/: model implementations
- metrics/: evaluation metrics
- evaluation/: bias-variance, robustness
- visualization/: plots only
- benchmarking/: performance analysis
- streamlit_app/: UI only

## Out of Scope
- Deep learning
- End-to-end MLOps
- AutoML
