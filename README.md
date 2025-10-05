# Problem Statement
The diamond industry is highly dynamic, where prices and demand are influenced by multiple attributes such as size, shape, color, clarity, cut, and fluorescence. Each diamond enters the inventory at a certain date and goes through multiple price revisions before it is sold or delisted.

1. Forecast future prices (discounts) of diamonds within each category.

2. Analyze demand trends across different diamond categories to anticipate which types of stones will gain or lose popularity.


## Project Structure
  1. datapreprocessing.py
  2. feature_analysis.ipynb
  3. train.py
  4. lstm.py


---

## Diamond Data Preprocessing

Processed raw diamond datasets from Excel sheets to prepare a cleaned, feature-engineered CSV file for downstream analysis and modeling.

### Preprocessing Steps

#### 1. Load & Merge Sheets  
Reads the Excel file and merges the **IN-OUT** and **PRICE** sheets on `ereport_no`.

#### 2. Rename Columns  
Standardizes column names for consistency:
- `cut_group` → `cut`
- `size` → `size_range`
- `disc` → `discount`

#### 3. Basic Cleaning  
- Create `is_sold` flag (**1 if sold, else 0**).  
- Fill missing `out_date` with `"Unsold"`.

#### 4. Mapping Quality Attributes  
Convert categorical attributes to numeric scores:
- **Color**
- **Clarity**
- **Cut**
- **Fluorescence**

#### 5. Feature Engineering  
- Convert `size_range` to `size_mid` (midpoint of range).  
- Remove outliers in `discount` using the IQR method.  
- Create `category_count` for group frequency.

#### 6. Quality Scoring System  
Compute a composite `quality_score` based on:
- Cut (**40%**)
- Color (**30%**)
- Clarity (**20%**)
- Fluorescence (**10%**)

#### 7. Lag Features  
- `discount_lag1` and `discount_lag2` for previous discounts.  
- `discount_diff` (difference with previous discount).  
- Missing values replaced with `0`.

#### 8. Tenure Features  
- Calculate `tenure_days` = difference between `out_date` and `in_date`.  
- Categorize tenure into:
  - **Short-term** (≤ 15 days)  
  - **Medium-term** (16–30 days)  
  - **Long-term** (> 30 days)

---

## Output

- **preprocessed_diamond.csv** — cleaned and feature-engineered dataset ready for analysis and modeling.
## Modelling & Results

Several regression models were trained and evaluated to predict diamond discounts, including:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Support Vector Regression (Linear & RBF)**
- **K-Nearest Neighbors Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **Voting Regressor**
- **Stacking Regressor**
- **MLP Regressor**

### Best Performing Model

After testing all models, **XGBoost Regressor** gave the best performance in terms of R² score and RMSE.

**Model:**

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(random_state=42, objective="reg:squarederror")
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

Performance:

R² Score: 0.9673431709083928

RMSE:  1.492540964554467
