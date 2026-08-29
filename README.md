This project's complete implementation is contained in startup-survival-time-prediction1.ipynb — a Kaggle notebook covering the full workflow end-to-end:

Data Loading & EDA — missing value inspection, skewness analysis, target distribution check
Preprocessing — a ColumnTransformer pipeline combining:
One-hot encoding (drop='first') for categorical features
Log transformation (log1p) + scaling for skewed financial columns
Standard scaling for remaining numerical columns
Model Training — Linear Regression, Ridge (L2), Lasso (L1), and ElasticNet, trained on identical preprocessed data
Evaluation — RMSE, MAE, R², and Adjusted R² for all four models
Residual Analysis — linearity, homoscedasticity, normality, and multicollinearity (VIF) checks
Overfitting Check — train vs. test performance gap
Coefficient Interpretation — ranked feature effects on predicted survival time
Prediction — applying the fitted pipeline to new, user-supplied startup data

