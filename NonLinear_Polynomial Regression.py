import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ds = pd.read_csv(r'/Users/aviswe/Desktop/830/Datasets/emp_sal.csv')

x=ds.iloc[:,1:2].values
y=ds.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


#Linear Regression Visualization
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('Linear Regression Graph')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(x),color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression Graph)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

from sklearn.svm import SVR
svr_reg = SVR()
svr_reg.fit(x,y)


svr_pred = svr_reg.predict([[6.5]])
print(svr_pred)






