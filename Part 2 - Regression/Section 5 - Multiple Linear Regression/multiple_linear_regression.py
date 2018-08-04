import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
estimator.fit(X_train,y_train)

y_pred = estimator.predict(X_test)

X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

X_opt = X[:,[0,1,2,3,4,5]]

import statsmodels.formula.api as sch
regressor_OLS = sch.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]

import statsmodels.formula.api as sch
regressor_OLS = sch.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]

import statsmodels.formula.api as sch
regressor_OLS = sch.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]

import statsmodels.formula.api as sch
regressor_OLS = sch.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

from sklearn.metrics import mean_squared_error

trainerrorvalues = []
testerrorvalues = []


for samples in range(1,41):
    X_temp = X_train[0:samples]
    y_temp = y_train[0:samples]
    estimator.fit(X_temp,y_temp)
    ypredtrain = estimator.predict(X_temp)
    ypredtest = estimator.predict(X_test)
    trainerror = mean_squared_error(y_train[0:samples],ypredtrain)
    testerror = mean_squared_error(y_test,ypredtest)
    trainerrorvalues.append(trainerror)
    testerrorvalues.append(testerror)
    
    
plt.plot(range(1,41),trainerrorvalues,color='red')
plt.plot(range(1,41),testerrorvalues,color='blue')
plt.show()















