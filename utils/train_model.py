"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')

train = train[(train['Commodities'] == 'APPLE GOLDEN DELICIOUS')]

#y_train = train['avg_price_per_kg']
#X_train = train[['Total_Qty_Sold','Stock_On_Hand']]
X = train[['Weight_Kg','Low_Price','High_Price','Sales_Total','Total_Qty_Sold','Total_Kg_Sold','Sales_Total','avg_price_per_kg']].drop('avg_price_per_kg', axis=1)
y = train['avg_price_per_kg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=4)

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

rf = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=90, max_features=3, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=3,
                      min_samples_split=4, min_weight_fraction_leaf=0.0,
                      n_estimators=300, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
rf.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/apples_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))

# Pickle model for use within our API
save_path = '../assets/trained-models/apples_random_forests.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(rf, open(save_path,'wb'))
