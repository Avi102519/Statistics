
import numpy as np 	#Array		
import matplotlib.pyplot as plt		
import pandas as pd

dataset= pd.read_csv(r"/Users/aviswe/Desktop/FSDS/Morning Batch/26th March/26th- Machine learning/5. Data preprocessing/Data.csv")

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer = imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
LabelEncoder_x = LabelEncoder

LabelEncoder_x.fit_transform(x[:0])
x[:,0]= LabelEncoder_x.fit_transform(x[:0])


LabelEncoder_x = LabelEncoder()
x=LabelEncoder_x.fit_transform(x)

LabelEncoder_y = LabelEncoder()
y=LabelEncoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)





#split the dataset with 75-25 | 85-15 | 70-30
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.25, train_size= 0.75, random_state=0) 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.15, train_size= 0.85, random_state=0)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.30, train_size= 0.70, random_state=0)

#The random_state parameter allows you to control this randomness by setting a fixed seed value.When you set random_state to a particular value, the randomness becomes reproducible. This means that if you run the same code with the same random_state value multiple times, you'll get the same results each time
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.25, train_size= 0.75)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.20, train_size= 0.80)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.30, train_size= 0.70)

#CREATE A DUMMY VARIABLE

from sklearn.preprocessing import LabelEncoder
#IMPUTE CATEGORICAL VALUE FOR INDEPENDENT 
labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:,0]) 
x[:,0] = labelencoder_x.fit_transform(x[:,0]) 

#IMPUTE CATEGORICAL VALUE FOR DEPENDENT 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.15, train_size= 0.85, random_state=0)
#DataProcessing_ex1.py
#Displaying DataProcessing_ex1.py