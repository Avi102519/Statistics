import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ds = pd.read_csv(r'/Users/aviswe/Desktop/830/Datasets/Salary_Data.csv')
ds
x = ds.iloc[:,[0]] #[:,:-1]
y = ds.iloc[:,1] #[:,-1]

ds.isnull().sum()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=True)


#Reshape the array
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp = m_slope*12 +c_intercept
print(pred_12yr_emp_exp)



pred_20yr_emp_exp = m_slope * 20 + c_intercept
print(pred_20yr_emp_exp)
