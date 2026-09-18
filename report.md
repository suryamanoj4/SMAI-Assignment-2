# SMAI Assignment 2 — Question 1 Report

## 1. Problem and Dataset

This assignment predicts hourly bike-rental demand using linear regression on the UCI Bike Sharing hourly dataset.

- Dataset: `hour.csv`
- Number of observations: **17,379**
- Original columns: 17
- Target variable: `cnt` (total hourly bike rentals)
- Removed columns: `cnt`, `casual`, `registered`, `instant`, and `dteday` from the input features
- Train/validation/test split: **80% / 10% / 10%**
  - Training: 13,903 rows
  - Validation: 1,738 rows
  - Test: 1,738 rows
- The split uses a username-derived deterministic random seed.

The target is `cnt`. The columns `casual` and `registered` were excluded because they directly contribute to the target and would cause target leakage.

---

## 2. Feature Engineering and Preprocessing

### 2.1 Categorical variables

The following variables were treated as categorical and one-hot encoded:

- `season`
- `mnth`
- `hr`
- `holiday`
- `weekday`
- `weathersit`

`drop_first=True` was used to avoid redundant dummy columns.

The continuous variables were:

- `temp`
- `atemp`
- `hum`
- `windspeed`

These continuous features were standardized using **training-set statistics only**. The same training means and standard deviations were then applied to the validation and test sets to avoid data leakage.

### 2.2 Handling `workingday`

`workingday` was excluded as a standalone feature because it is linearly dependent on the weekday/holiday representation. However, it was retained in selected interaction terms.

The following five interaction features were added:

1. `hr8_weather1`
2. `hr17_weather1`
3. `hr8_workingday`
4. `hr17_workingday`
5. `hr12_weather2`

The final engineered design matrix contained:

- **57 features before adding the intercept**
- **58 columns after adding the intercept**

---

# Q1.1 — Hour Representation Ablation

## Approach

Two models were compared using the same train/validation split and the same additional features:

1. Treating `hr` as a raw integer feature.
2. Treating `hr` as a categorical variable using one-hot encoding.

The continuous variables were standardized using training-set statistics. A closed-form least-squares solution was used for both models.

## Results

| Hour representation | Number of features | Validation RMSE |
|---|---:|---:|
| Raw integer `hr` | 35 | 119.7124 |
| One-hot encoded `hr` | 57 | 93.0953 |

The validation RMSE improvement from one-hot encoding was:

\[
119.7124 - 93.0953 = 26.6171
\]

## Answer

One-hot encoding performed substantially better than treating hour as a raw integer. A raw integer representation imposes a simple linear relationship between hour and demand, while one-hot encoding allows each hour to have a separate effect. This is more suitable because bike demand varies nonlinearly across the day, with patterns such as morning and evening commuting peaks.

---

# Q1.2 — Feature Inspection and Visualization

The notebook includes the following exploratory visualizations:

- Distribution of hourly bike-rental demand (`cnt`)
- Temperature versus bike-rental demand
- Average bike-rental demand by hour
- Correlation heatmap of the original numerical and categorical-coded variables

The hourly-demand visualization indicates that demand changes substantially across the day, supporting the decision to represent `hr` as a categorical variable rather than as a single linear numerical feature.

The correlation heatmap is useful for identifying relationships among variables, but correlations between integer-coded categorical variables should be interpreted carefully.

---

# Q1.3 — Closed-Form Linear Regression and Conditioning

## Closed-form solution

The least-squares coefficient vector is computed using:

\[
\hat{\beta} = (X^TX)^{-1}X^Ty
\]

The implementation uses the design matrix with an explicit intercept column.

The final design matrix has 58 columns, including the intercept.

## Condition numbers

| Matrix | Condition number |
|---|---:|
| \(X\) | 100.4325 |
| \(X^TX\) | 10086.6807 |

The condition number of \(X^TX\) is approximately the square of the condition number of \(X\), which is expected for a full-rank matrix.

The feature design was modified to remove the standalone `workingday` column while retaining the interaction terms. This reduced the severe conditioning problem observed in the earlier feature representation.

## Closed-form results

| Split | RMSE |
|---|---:|
| Training | 93.0686 |
| Validation | 93.0953 |

The training and validation errors are close, suggesting that the model does not exhibit a large train-validation error gap for this feature representation.

> Numerical note: `np.linalg.lstsq` or `np.linalg.solve` is generally preferable to explicitly computing a matrix inverse with `np.linalg.inv`, because it is more numerically stable.

---

# Q1.3 — Gradient Descent and Learning-Rate Ablation

## Approach

Batch gradient descent was implemented from scratch. The weights were initialized to zero, and the mean squared-error objective was minimized iteratively.

The tested learning rates were:

\[
10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0
\]

Both standardized and unstandardized versions of the features were examined.

## Stability results

| Learning rate | Unstandardized | Standardized |
|---:|:---:|:---:|
| \(10^{-5}\) | Stable | Stable |
| \(10^{-4}\) | Stable | Stable |
| \(10^{-3}\) | Stable | Stable |
| \(10^{-2}\) | Stable | Stable |
| \(10^{-1}\) | Stable | Stable |
| \(1.0\) | Unstable | Unstable |

The largest learning rate classified as stable for both representations was:

\[
\alpha = 0.1
\]

When run with 50,000 iterations and a tolerance of \(10^{-8}\), both runs reached the maximum iteration limit rather than satisfying the convergence tolerance.

## Gradient-descent results

Using the notebook's final gradient-descent run with learning rate \(0.01\) and a maximum of 50,000 iterations:

| Split | RMSE |
|---|---:|
| Training | 93.5273 |
| Validation | 93.5902 |

The maximum absolute difference between the closed-form and gradient-descent coefficient vectors was approximately **59.2596**. This difference should be interpreted together with the prediction errors, because individual coefficients can differ when features are correlated even if the predictions are relatively similar.

---

# Q1.4 — Polynomial Regression and Bias–Variance Trade-off

## Dataset

A separate dataset, `bike_temp_comfort_zone.csv`, was used for polynomial regression.

- Training samples: 400
- Validation samples: 200
- Test samples: 200
- Input feature: `temp`
- Target: `demand`

Polynomial features from degree 1 through degree 12 were created. The polynomial powers were standardized using training-set statistics, while the intercept column was not standardized.

## Results

| Degree | Training RMSE | Validation RMSE |
|---:|---:|---:|
| 1 | 117.286843 | 121.769875 |
| 2 | 79.082943 | 74.987688 |
| 3 | 63.348213 | 60.557225 |
| 4 | 60.977979 | 57.773170 |
| 5 | 60.296579 | 56.719809 |
| 6 | 60.292042 | 56.713951 |
| 7 | 60.290184 | 56.690702 |
| 8 | 60.234848 | 57.172054 |
| 9 | 60.133104 | 57.526858 |
| 10 | 59.809638 | 58.160283 |
| 11 | 59.658924 | 59.117329 |
| 12 | 61.115685 | 58.822353 |

The lowest validation RMSE occurred at:

- **Best degree: 7**
- **Validation RMSE: 56.690702**
- **Test RMSE: 52.614201**

## Bias–variance interpretation

At low polynomial degrees, the model is too simple and has high bias, producing relatively large training and validation errors.

As the degree increases, the model becomes more flexible and the errors decrease. Around degree 7, validation performance is best. Beyond this point, training error continues to remain low or decrease, while validation error increases or fluctuates. This is consistent with increasing variance and overfitting at higher polynomial degrees.

The selected degree is therefore **7**, based on the lowest validation RMSE.

---

# Q1.5 — Ridge Regression

## Approach

Ridge regression was implemented from scratch using:

\[
\hat{\beta}_{\lambda}
=
(X^TX+\lambda I)^{-1}X^Ty
\]

The intercept was not penalized. The penalty matrix therefore has a zero in its first diagonal position.

For \(\lambda=0\), the implementation uses `np.linalg.lstsq`.

The tested values were:

\[
0,\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 10^{-1},\
1,\ 10,\ 100,\ 1000,\ 10000,\ 100000
\]

Two experiments were performed:

1. The complete training set.
2. A randomly selected 100-row training subset.

For the 100-row subset, one zero-variance feature was removed before fitting.

## Full training-set results

| Lambda | Training RMSE | Validation RMSE |
|---:|---:|---:|
| 0 | 93.068610 | 93.095315 |
| 0.0001 | 93.068610 | 93.095315 |
| 0.001 | 93.068610 | 93.095316 |
| 0.01 | 93.068610 | 93.095317 |
| 0.1 | 93.068668 | 93.095382 |
| 1 | 93.073930 | 93.100465 |
| 10 | 93.374510 | 93.396832 |
| 100 | 97.751905 | 97.846430 |
| 1000 | 120.145128 | 121.209243 |
| 10000 | 150.362031 | 151.743827 |
| 100000 | 171.723089 | 172.021255 |

### Full-data conclusion

The best validation result was obtained at:

\[
\lambda = 0
\]

with validation RMSE approximately **93.095315**.

For this feature representation and the full training dataset, positive regularization did not improve validation performance. This is a valid outcome: Ridge regression is an option for reducing variance, but it is not guaranteed to outperform ordinary least squares on every dataset.

The very small lambda values have almost no observable effect. Larger values progressively shrink the coefficients and increase both training and validation error.

## 100-row subset results

| Lambda | Training RMSE | Validation RMSE |
|---:|---:|---:|
| 0 | 61.127806 | 144.335725 |
| 0.0001 | 61.127811 | 144.309609 |
| 0.001 | 61.128292 | 144.077371 |
| 0.01 | 61.168424 | 141.999263 |
| 0.1 | 62.514011 | 131.900465 |
| 1 | 72.605212 | 126.096535 |
| 10 | 96.643177 | 141.814387 |
| 100 | 123.424911 | 159.164502 |
| 1000 | 144.580618 | 174.935869 |
| 10000 | 151.210815 | 180.115107 |
| 100000 | 152.034870 | 180.764766 |

### Subset conclusion

For the 100-row subset, the best validation result was obtained at:

\[
\lambda = 1
\]

The validation RMSE improved from approximately **144.335725** at \(\lambda=0\) to **126.096535** at \(\lambda=1\).

This demonstrates the benefit of regularization when the training sample is very small relative to the number of features. The unregularized model fits the 100-row subset closely but generalizes poorly. Ridge reduces coefficient magnitudes and improves validation performance.

The improvement was:

\[
144.335725 - 126.096535 \approx 18.239190
\]

---

# Q1.6 — Test-Set Evaluation

## Model selection

The regularization strength was selected using the validation RMSE from the **full training-set sweep**, as required.

The selected value was:

\[
\lambda = 0
\]

Therefore, the final test evaluation uses the full-data least-squares model.

Negative predictions were not clipped because the assignment explicitly requires evaluation without clipping. Although the target is a non-negative rental count, an unconstrained linear regression model can produce negative predictions.

The notebook reported **185 negative test predictions**.

## Final test metrics

| Metric | Value |
|---|---:|
| MAE | 69.135636 |
| RMSE | 92.152157 |
| \(R^2\) | 0.752184 |
| Pearson correlation | 0.867293 |

## Interpretation

- **MAE:** The average absolute prediction error is approximately 69.14 rental counts.
- **RMSE:** The RMSE is approximately 92.15. Because RMSE squares errors before averaging, it is more sensitive to large prediction errors than MAE.
- **\(R^2\):** The model explains approximately 75.2% of the variance in the test target under the evaluated setup.
- **Pearson correlation:** The correlation of approximately 0.867 indicates a strong positive linear association between predicted and observed demand.
- **Negative predictions:** The model produced 185 negative predictions. These were retained in the metric calculations to comply with the assignment requirement.

---

# Overall Conclusions

1. One-hot encoding the hour of day substantially improved validation performance compared with using raw integer hour values.
2. Removing the standalone `workingday` feature reduced the conditioning problem while retaining useful working-day information through interaction terms.
3. Closed-form linear regression achieved a validation RMSE of approximately 93.10.
4. Gradient descent produced a similar but slightly higher validation RMSE in the final run.
5. Polynomial regression on the temperature-demand dataset achieved its lowest validation RMSE at degree 7.
6. Ridge regression did not improve the full-data validation result, where \(\lambda=0\) was selected.
7. Ridge regression substantially improved validation performance on the 100-row subset, with \(\lambda=1\) reducing validation RMSE from approximately 144.34 to 126.10.
8. The final test evaluation, using the lambda selected on the full training sweep, achieved an RMSE of approximately 92.15 and an \(R^2\) of approximately 0.7522.
