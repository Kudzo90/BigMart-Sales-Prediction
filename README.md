# BigMart Sales Prediction

## Overview
This project predicts sales for BigMart outlets using historical data and product attributes. It demonstrates data preprocessing, feature engineering, and modeling with both deep learning and gradient boosting techniques.

## Dataset
- **Source**: BigMart Sales dataset (Kaggle)
- **Train Size**: 8,523 rows, 12 features
- **Test Size**: 5,681 rows, 11 features

Features include:
- Item attributes (weight, type, MRP)
- Outlet attributes (size, location, type)
- Target: `Item_Outlet_Sales`

## Approach
1. **Data Preprocessing**
   - Handle missing values (median for numeric, mode for categorical)
   - Standardize categorical variables
   - Feature engineering: `Outlet_Age`, combined item types
   - One-hot encoding & scaling

2. **Models Implemented**
   - **Improved Neural Network**: Deep architecture with L2 regularization, Batch Normalization, Dropout.
   - **XGBoost**: Gradient boosting tuned with RandomizedSearchCV.
   - **Ensemble**: Average predictions from NN and XGBoost.

3. **Evaluation Metrics**
   - MSE, RMSE, R²

## Performance Summary
| Model        | MSE       | RMSE    | R²    |
|-------------|-----------|---------|-------|
| Improved NN | 1,207,398 | 1098.82 | 0.556 |
| XGBoost     | 1,117,793 | 1057.26 | 0.589 |
| Ensemble    | 1,145,670 | 1070.36 | 0.578 |

## Interpretation
- **XGBoost** achieved the lowest RMSE and highest R², making it the most accurate model.
- **Ensemble** improved slightly over NN but did not outperform XGBoost.
- **Improved NN** performed well but was less effective for this tabular dataset.

## Decision
For deployment and portfolio showcase:
- **XGBoost is the recommended model** due to superior accuracy and stability.
- Neural Network and Ensemble approaches are included for diversity and demonstration of deep learning skills.

## How to Run
```bash
# Clone repository
git clone https://github.com/your-username/bigmart-sales-prediction.git
cd bigmart-sales-prediction

# Install dependencies
pip install -r requirements.txt

# Run the script
python compare_models.py
