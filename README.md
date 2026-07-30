# Used-car-selling-price-prediction
Predicting the selling price of used cars from their attributes using linear regression and polynomial regression, implemented from scratch in NumPy (no scikit-learn). The dataset contains a car's name, manufacturing year, selling price, kilometres driven, fuel type, seller type, transmission, and owner history.

Dataset
File	Rows	Description
Train.csv	8,128	Training data
Test.csv	4,340	Test data

Target: selling_price Features used: year, km_driven (numeric) + fuel, seller_type, transmission, owner (one-hot encoded). The free-text name column is dropped.

Methods
1. Linear Regression

Two approaches are implemented from scratch and fitted on standardized features to obtain standardized coefficients β:

Multi-Feature (Normal) Equation — closed-form solution β = (XᵀX)⁻¹Xᵀy
Batch Gradient Descent — iterative optimization (lr = 0.1, 8,000 epochs)

Both are compared using RMSE, MSE, and MAE.

2. Polynomial Regression

The numeric features are expanded to polynomial degrees 1 through 10, a model is fitted at each degree, and the test RMSE is recorded to identify the optimal degree.

Results

Linear regression — both methods converge to the same solution (coefficient vectors differ by ~2e-6):

Metric	Normal Equation	Gradient Descent
RMSE	503,445.30	503,445.31
MSE	2.53e11	2.53e11
MAE	312,861.54	312,861.54

The Normal Equation is preferred — it is exact, needs no tuning, and is instant at this data size. Gradient descent only matches it after many tuned epochs.

Polynomial regression — RMSE is lowest at degree 3 (RMSE ≈ 492,038), then rises for higher degrees as they overfit the data.

<img width="1040" height="650" alt="image" src="https://github.com/user-attachments/assets/8ec9f6cb-41a1-4000-8fcc-9384d209cfad" />
