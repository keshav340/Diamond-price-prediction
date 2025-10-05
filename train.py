import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Load & Prepare Data
data = pd.read_csv("preprocessed_diamond.csv")

X = data.drop(columns=[
    'diamond_id', 'pdate', 'rap_rate', 'discount', 'rate',
    'in_date', 'size_range', 'out_date', 'tenure_days'
])
y = data['discount']

le = LabelEncoder()
X['shape'] = le.fit_transform(X['shape'])
X['tenure_category'] = le.fit_transform(X['tenure_category'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Helper Functions
def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model": model,
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "y_pred": y_pred
    }

def grid_search(model, param_grid, X_train, y_train, scoring="r2"):
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# Models & Hyperparameters
models_params = {
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (Lasso(max_iter=10000), {"alpha": [0.001, 0.01, 0.1, 1, 10]}),
    "DecisionTree": (DecisionTreeRegressor(random_state=42), {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }),
    "RandomForest": (RandomForestRegressor(random_state=42), {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"]
    }),
    "SVR_Linear": (SVR(kernel="linear"), {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5, 1]
    }),
    "SVR_RBF": (SVR(kernel="rbf"), {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5, 1],
        "gamma": ["scale", "auto"]
    }),
    "KNN": (KNeighborsRegressor(), {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }),
    "MLP": (MLPRegressor(max_iter=2000, random_state=42), {
        "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "lbfgs"],
        "learning_rate_init": [0.001, 0.01, 0.1]
    })
}

results = {}

#  Run Grid Search & Evaluation
for name, (model, params) in models_params.items():
    print(f"Running {name}...")
    best_model, best_params, best_score = grid_search(model, params, X_train, y_train)
    eval_results = evaluate(best_model, X_train, y_train, X_test, y_test)
    results[name] = {
        "model": best_model,
        "best_params": best_params,
        "best_cv_score": best_score,
        "test_r2": eval_results["r2"],
        "test_rmse": eval_results["rmse"]
    }

# Gradient Boosting & XGBoost (no grid for speed)
gbr = GradientBoostingRegressor(random_state=42)
results["GradientBoosting"] = evaluate(gbr, X_train, y_train, X_test, y_test)

xgb = XGBRegressor(random_state=42, objective="reg:squarederror")
results["XGBoost"] = evaluate(xgb, X_train, y_train, X_test, y_test)

#  Ensemble Models
voting_reg = VotingRegressor([
    ('rf', results["RandomForest"]["model"]),
    ('dt', results["DecisionTree"]["model"]),
    ('gbr', results["GradientBoosting"]["model"]),
    ('xgb', results["XGBoost"]["model"])
])
results["VotingRegressor"] = evaluate(voting_reg, X_train, y_train, X_test, y_test)

stacking_reg = StackingRegressor(
    estimators=[
        ('rf', results["RandomForest"]["model"]),
        ('dt', results["DecisionTree"]["model"]),
        ('gbr', results["GradientBoosting"]["model"]),
        ('xgb', results["XGBoost"]["model"])
    ],
    final_estimator=RidgeCV()
)
results["StackingRegressor"] = evaluate(stacking_reg, X_train, y_train, X_test, y_test)

#  Summary of Results
print("\nModel Performance Summary:")
for model_name, res in results.items():
    print(f"{model_name}: R² = {res['test_r2']:.4f}, RMSE = {res['test_rmse']:.4f}")
