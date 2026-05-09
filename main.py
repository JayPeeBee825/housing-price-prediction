"""
CS439 Final Project: Housing Price Prediction

Goal: Predict housing sale prices and identify which features are most important.
Uses the Ames Housing dataset with multiple regression models for comparison.
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler 

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict:
    """
    Evaluates regression model performance using RMSE and MAE.
    Args:
        y_true (pd.Series): Actual target variable values.
        y_pred (np.ndarray): Model predictions of target variable values.
        model_name (str): Name of the model.
    Returns:
        dict: Dictionary with 'rmse' and 'mae' keys.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name}")
    print(f"\tRMSE: {rmse:.4f}")
    print(f"\tMAE: {mae:.4f}")
    print()
    return {"rmse": rmse, "mae": mae}

def preprocess_data(df: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fills in missing values using the median of each column, calculated from the training set.

    Args:
        df (pd.DataFrame): Feature dataframe that may contain missing values.
        medians (pd.Series | None): Train-fitted medians that get passed in when processing the test data.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The filled dataframe and the medians used to fill it.
    """
    if medians is None:
        medians = df.median(numeric_only=True)
    return df.fillna(medians), medians

def plot_saleprice_distribution(df: pd.DataFrame) -> None:
    """
    Plots the raw `SalePrice` distribution before any transformation.

    Args:
        df (pd.DataFrame): Raw dataframe containing a `SalePrice` column.
    """
    sns.histplot(df["SalePrice"], kde = True)
    plt.title("SalePrice Distribution (Before Log Transform)")
    plt.show()

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Log-transforms the target, one-hot encodes features, splits into train/test sets, 
    and fills in missing values using the train-set medians.

    Args:
        df (pd.DataFrame): Raw dataframe containing a `SalePrice` column.
        test_size (float): Fraction of data reserved for testing. Defaults to 0.2.
        random_state (int): Seed for reproducibility of accuracy metrics. Defaults to 42.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    # Log-transform target before splitting
    y = np.log1p(df["SalePrice"])
    x_raw = df.drop("SalePrice", axis=1)

    # One-hot encode
    x_encoded = pd.get_dummies(x_raw, drop_first=True).astype(np.float64)

    # Use 80/20 split for data (80% train : 20% test)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_encoded, y, test_size = 0.2, random_state = 42
    )

    # Fill in missing values using train-set medians only
    x_train, train_medians = preprocess_data(x_train_raw)
    x_test, _ = preprocess_data(x_test_raw, train_medians)

    return x_train, x_test, y_train, y_test

def train_models(x_train, x_test, x_train_scaled, x_test_scaled, y_train) -> dict:
    """
    Trains all four regression models and returns their predictions.

    Args:
        x_train (pd.DataFrame): Unscaled training features.
        x_test (pd.DataFrame): Unscaled test features.
        x_train_scaled (np.ndarray): Scaled training features.
        x_test_scaled (np.ndarray): Scaled test features.
        y_train (pd.Series): Training target values.

    Returns:
        dict: model name -> (fitted model, predictions)  pairs.
    """
    models = {
        "Linear Regression": (LinearRegression(), x_train_scaled, x_test_scaled),
        "Ridge": (Ridge(alpha = 1.0), x_train_scaled, x_test_scaled),
        "Lasso": (Lasso(alpha = 0.001), x_train_scaled, x_test_scaled),
        "Random Forest": (RandomForestRegressor(n_estimators = 100, random_state = 42), x_train, x_test),
    }
    results = {}
    for name, (model, x_train, x_test) in models.items():
        model.fit(x_train, y_train)
        results[name] = (model, model.predict(x_test))
    return results

def plot_feature_importance(rf_model: RandomForestRegressor, feature_names: pd.Index) -> None:
    """
    Prints and plots the top 10 most important features from a Random Forest model.

    Args:
        rf_model (RandomForestRegressor): Fitted Random Forest model.
        feature_names (pd.Index): Column names of the training features
    """
    # This helps answer the research question: Which features are most important for predicting hosing prices?
    feat_importance = pd.Series(rf_model.feature_importances_, index = feature_names)
    top_10 = feat_importance.sort_values(ascending = False).head(10)

    print("Top 10 Most Important Features: \n")
    print(top_10)
    print()

    # Visualize top features 
    top_10.plot(kind = "barh", figsize = (8, 6))
    plt.title("Top 10 Important Features (Random Forest)")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def main():

    # Step 1: Load Data
    print("\nStep 1: Loading Data")
    print("=" * 50) 
    df = pd.read_csv("train.csv")
    print(f"Dataset shape: {df.shape}\n")

    # Step 2: Exploratory Data Analysis
    print("Step 2: Exploratory Data Analysis")
    print("=" * 50)
    plot_saleprice_distribution(df)

    # Step 3: Encode, Split, and Preprocess
    print("Step 3: Encode, Split, and Preprocess")
    print("=" * 50)
    x_train, x_test, y_train, y_test = prepare_data(df)
    print(f"Train shape (raw): {x_train.shape}")
    print(f"Test shape (raw): {x_test.shape}\n")

    # Step 4: Feature Scaling
    print("Step 4: Feature Scaling")
    print("=" * 50)
    scaler =  StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print("Scaling complete.\n")

    # Step 5: Train Models 
    print("Step 5: Train Models")
    print("=" * 50)
    results = train_models(x_train, x_test, x_train_scaled, x_test_scaled, y_train)
    print("All models trained.\n")

    # Step 6: Model Evaluation 
    print("Step 6: Model Evaluation")
    print("=" * 50)
    for name, (_, preds) in results.items():
        evaluate_model(y_test, preds, name)

    # Step 7: Feature Importance
    print("Step 7: Feature Importance Analysis")
    print("=" * 50)
    rf_model, _ = results["Random Forest"]
    plot_feature_importance(rf_model, x_train.columns)
    
if __name__ == "__main__":
    main()
