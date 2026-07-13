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

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the housing dataset from a CSV file

    Args:
        file_path (str): File path of the housing data

    Returns:
        pd.DataFrame: DataFrame of housing data
    """
    return pd.read_csv(file_path)

def plot_saleprice_distribution(df: pd.DataFrame) -> None:
    """
    Plots the raw `SalePrice` distribution before any transformation.

    Args:
        df (pd.DataFrame): Raw dataframe containing a `SalePrice` column.
    """
    sns.histplot(df["SalePrice"], kde = True)
    plt.title("SalePrice Distribution (Before Log Transform)")
    plt.tight_layout()
    plt.savefig("figures/saleprice_distribution.png", dpi = 300)
    plt.show()

def separate_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates the target variable from the housing features and log-transforms the target.

    Args:
        df (pd.DataFrame): DataFrame of housing data

    Returns:
        tuple[pd.DataFrame, pd.Series]: DataFrame of housing features and 1D array (Series) of respective log-transformed sale prices (target variable, y)
    """
    y = np.log1p(df["SalePrice"])
    X = df.drop("SalePrice", axis=1)
    return X, y

def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes categorical housing features and converts the dataframe to float64.

    Args:
        X (pd.DataFrame): DataFrame of housing features before encoding

    Returns:
        pd.DataFrame: DataFrame of housing features after encoding
    """
    X_encoded = pd.get_dummies(X, drop_first = True).astype(np.float64)
    return X_encoded

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into training and testing sets.

    Args:
        X (pd.DataFrame): Encoded housing feature dataframe
        y (pd.Series): Log-transformed target variable
        test_size (float, optional): Fraction of the dataset used for testing. Defaults to 0.2.
        random_state (int, optional): Random seed used to ensure reproducible splits. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: (X_train, X_test, y_train, y_test)
    """
    
    return train_test_split(X, y, test_size = test_size, random_state = random_state)                                        
     

def fill_missing_values(df: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
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

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Log-transforms the target, one-hot encodes features, splits into train/test sets, 
    and fills in missing values using the train-set medians.

    Args:
        df (pd.DataFrame): Raw dataframe containing a `SalePrice` column.
        test_size (float): Fraction of data reserved for testing. Defaults to 0.2.
        random_state (int): Seed for reproducibility of accuracy metrics. Defaults to 42.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    X, y = separate_target(df)
    X = encode_features(X)

    X_train, X_test, y_train, y_test = split_data(
        X, 
        y, 
        test_size = test_size, 
        random_state = random_state
    )

    X_train, train_medians = fill_missing_values(X_train)
    X_test, _ = fill_missing_values(X_test, train_medians)

    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardizes the training and testing feature sets using the training data

    Args:
        X_train (pd.DataFrame): Training data matrix of housing features
        X_test (pd.DataFrame): Testing data matrix of housing features

    Returns:
        tuple[np.ndarray, np.ndarray, StandardScaler]: (X_train_scaled, X_test_scaled, scaler)
    """

    scaler =  StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame, X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Trains all four regression models and returns their predictions.

    Args:
        X_train (pd.DataFrame): Unscaled training features.
        X_test (pd.DataFrame): Unscaled test features.
        X_train_scaled (np.ndarray): Scaled training features.
        X_test_scaled (np.ndarray): Scaled test features.
        y_train (pd.Series): Training target values.

    Returns:
        dict: model name -> (fitted model, test predictions).
    """
    models = {
        "Linear Regression":    (LinearRegression(), X_train_scaled, X_test_scaled),
        "Ridge":                (Ridge(alpha = 1.0), X_train_scaled, X_test_scaled),
        "Lasso":                (Lasso(alpha = 0.001), X_train_scaled, X_test_scaled),
        "Random Forest":        (RandomForestRegressor(n_estimators = 100, random_state = 42), X_train, X_test),
    }
    results = {}
    for name, (model, training_features, testing_features) in models.items():
        model.fit(training_features, y_train)
        predictions = model.predict(testing_features)
        results[name] = (model, predictions)
    return results

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
    plt.savefig("figures/random_forest_feature_importance.png", dpi = 300)
    plt.show()
    
def main():

    # Step 1: Load Data
    print("\nStep 1: Loading Data")
    print("=" * 50) 
    df = load_data("data/data.csv")
    print(f"Dataset shape: {df.shape}\n")

    # Step 2: Exploratory Data Analysis
    print("Step 2: Exploratory Data Analysis")
    print("=" * 50)
    plot_saleprice_distribution(df)

    # Step 3: Encode, Split, and Preprocess
    print("Step 3: Encode, Split, and Preprocess")
    print("=" * 50)
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}\n")

    # Step 4: Feature Scaling
    print("Step 4: Feature Scaling")
    print("=" * 50)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, 
        X_test
    )
    print("Scaling complete.\n")

    # Step 5: Train Models 
    print("Step 5: Train Models")
    print("=" * 50)
    results = train_models(
        X_train, 
        X_test, 
        X_train_scaled, 
        X_test_scaled, 
        y_train
    )
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
    plot_feature_importance(rf_model, X_train.columns)
    
if __name__ == "__main__":
    main()
