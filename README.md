# ML Practical Session 3: Hypothesis Spaces, Feature Representations and Model Generalization

## Overview

This notebook predicts students' exam scores from their study hours (10 observations) and compares how different regression models fit and generalize. All models are evaluated on the same dataset with the same strategy: Leave-One-Out Cross-Validation (LOOCV).

## Contents

| Part | Topic |
|------|-------|
| A | Dataset exploration; fitting Linear, Polynomial (deg 2, 3), Decision Tree (depth 1, 3) and KNN (k = 3); training RMSE and R² |
| B | LOOCV splits; polynomial degrees 1, 2, 3, 5, 9; training vs. LOOCV RMSE |
| C | Feature representations; Ridge and Lasso with a `Pipeline` (polynomial features, scaling, regression) |
| D | Best configuration per model family; summary table and comparative plot |
| Conclusion | Effect of complexity, feature representation and regularization on generalization |

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib, scikit-learn
- Jupyter (Notebook, JupyterLab or VS Code)

## Methodology notes

- **Metric:** LOOCV RMSE is computed from `cross_val_score(..., cv=LeaveOneOut(), scoring="neg_mean_squared_error")`, taking the square root of the mean fold MSE.
- **Data leakage:** feature scaling (`StandardScaler`) is inside a `Pipeline`, so it is re-fitted on the training part of each fold only.
- **Regularization:** alpha is chosen from {0.01, 0.1, 1, 10} using LOOCV on the same ten points, so the reported error is slightly optimistic.
- **Degree 9:** without scaling, the degree-9 fit is numerically ill-conditioned (x⁹ reaches 10⁹), so its training RMSE does not reach 0. A check cell with standardized features is included.

## Main results (from the included run)

| Model | Best configuration | Training RMSE | LOOCV RMSE |
|-------|--------------------|---------------|------------|
| Linear Regression | default | 1.17 | 1.55 |
| Polynomial Regression | degree 2 | 1.04 | 1.57 |
| Lasso Regression | degree 5, alpha = 0.1 | 1.07 | 1.62 |
| Ridge Regression | degree 5, alpha = 0.1 | 1.09 | 1.81 |
| KNN | k = 3 | 2.66 | 6.00 |
| Decision Tree | depth 3 | 1.12 | 6.37 |

Takeaway: the data is nearly linear, so simple or regularized models generalize best. Higher complexity lowers training error but raises LOOCV error, and regularization recovers much of the loss for the degree-5 polynomial.
