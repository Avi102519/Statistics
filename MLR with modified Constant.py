import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv(r'/Users/aviswe/Desktop/830/Datasets/Investment.csv')

x = ds.iloc[:,:-1]
y = ds.iloc[:,4]

x= pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

m_coef = regressor.coef_
print(m_coef)

c_intercept = regressor.intercept_  #Intercept if for Y axis so only one Intercept
print(c_intercept)

#x = np.append(arr=np.ones((50,1)).astype(int), values=x,axis=1)
#create a column of 42467
column= np.full((50,1),42467)
#append the column to x
x = np.append(arr=column,values=x,axis=1)

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
#Ordinary Least Squares
regressor_OLS = sm.OLS(endog =y,exog = x_opt).fit()
regressor_OLS.summary()

# Feature elimination which has highest P value here 4 has 0.99
# P-value shall bre less than0.05
import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]
#Ordinary Least Squares
regressor_OLS = sm.OLS(endog =y,exog = x_opt).fit()
regressor_OLS.summary()



import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]
#Ordinary Least Squares
regressor_OLS = sm.OLS(endog =y,exog = x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,3]]
#Ordinary Least Squares
regressor_OLS = sm.OLS(endog =y,exog = x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1]]
#Ordinary Least Squares
regressor_OLS = sm.OLS(endog =y,exog = x_opt).fit()
regressor_OLS.summary()










