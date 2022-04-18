import numpy as np 
import pandas as pd

data = pd.read_csv("Real estate.csv")

#Performing data wrangling
data = data.drop(columns="No")

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

score = regressor.score(x_test, y_test)

from sklearn.svm import SVR
svr_regressor = SVR(kernel="rbf")
svr_regressor.fit(x_test, y_test)

y_svr_pred = svr_regressor.predict(x_train) 
svr_score = svr_regressor.score(x_test, y_test)

from sklearn.tree import DecisionTreeRegressor
dtree_regressor = DecisionTreeRegressor()
dtree_regressor.fit(x_train, y_train)
]
y_dtree_pred = dtree_regressor.predict(x_test)
d_score = dtree_regressor.score(x_test, y_test)

from sklearn.ensemble import RandomForestRegressor
ranfor_regressor = RandomForestRegressor(n_estimators=100)
ranfor_regressor.fit(x_train, y_train)

y_ranfor_score = ranfor_regressor.score(x_test, y_test)
y_ranfor_predict = ranfor_regressor.predict(x_test)