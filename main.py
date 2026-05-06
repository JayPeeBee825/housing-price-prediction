# CS439 Final Project 
# Goal: Predict housing prices and figure out which features matter most 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler 

# Step 1: Load Data
# using the Ames Housing dataset from Kaggle 
# this dataset has a lot of features like lot sizes, rooms, neighborhood, etc. 
df = pd.read_csv("train.csv")
print("Dataset shape:", df.shape)

# Step 2: Basic EDA 
# looking at distribution of sale price 
# we expect it to be skewed, which is common in housing data 
sns.histplot(df["SalePrice"], kde = True)
plt.title("SalePrice Distribution (Before Log Transform)")
plt.show()

# Step 3: Preprocessing 
# since SalePrice is right skewed, we apply log transform
# this helps linear models perform better 
df["SalePrice"] = np.log1p(df["SalePrice"])
# separate numerical and categorical columns 
num_cols = df.select_dtypes(include = ["int64", "float64"]).columns 
cat_cols = df.select_dtypes(include = ["object"]).columns
# handling missing values - numerical (median), categorical (most frequent value)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# convert categorical variables into numbers using dummy encoding 
# this increases number of features but allows models to use them
df = pd.get_dummies(df, drop_first = True)       

# Step 4: Train/Test 
# x = features, y = target (SalePrice)
x = df.drop("SalePrice", axis = 1)
y = df["SalePrice"]
# using 80/20 split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

# Step 5: Scaling (for linear model)
# scaling helps models like Linear Regression, Ridge, Lasso 
scaler =  StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Train Models 
# baseline model: Linear Regression 
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
lr_preds = lr.predict(x_test_scaled)
# ridge (handles multicollinearity)
ridge = Ridge(alpha = 1.0)
ridge.fit(x_train_scaled, y_train)
ridge_preds = ridge.predict(x_test_scaled)
# lasso (does feature selection)
lasso = Lasso(alpha = 0.001)
lasso.fit(x_train_scaled, y_train)
lasso_preds = lasso.predict(x_test_scaled)
# random forest (captures non linear relationships)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x_train, y_train) 
rf_preds = rf.predict(x_test)

# Step 7: Evaluation 
# using RMSE (main metric from proposal) and MAE (for interpretability)
def evaluate(y_true, y_pred, name):
    rsme = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(name)
    print("RMSE:", round(rsme, 4))
    print("MAE:", round(mae, 4))
    print()
    
print("\nModel Results:\n")

evaluate(y_test, lr_preds, "Linear Regression")
evaluate(y_test, ridge_preds, "Ridge")
evaluate(y_test, lasso_preds, "Lasso")
evaluate(y_test, rf_preds, "Random Forest")

# Step 8: Feature Importance 
# this helps answer our research question: which features are most important for predicting hosing prices 
importances = rf.feature_importances_
features = x.columns 
feat_series = pd.Series(importances, index = features)
top_feats = feat_series.sort_values(ascending = False).head(10)
print("Top 10 Most Important Features: \n")
print(top_feats)
# visualize top features 
top_feats.plot(kind = "barh")
plt.title("Top 10 Important Features (Random Forest)")
plt.gca().invert_yaxis()
plt.show()
