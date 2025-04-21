#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import dataset
data = pd.read_csv(r'/Users/aviswe/Desktop/830/Datasets/Data_purchase.csv')

#split data into iv and dv
x = data.iloc[:,:-1].values
y = data.iloc[:,3].values

#Missing value treatment
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent") #Hyper parameter tuning with median

#Parameter Tuning by default it will be mean strategy
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#impute dv(n-0,y-1)
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()

labelencoder_x.fit_transform(x[:,0])
x[:,0] = labelencoder_x.fit_transform(x[:,0])


labelencoder_y=LabelEncoder()

y = labelencoder_y.fit_transform(y)

#Split the data to Train and Test size
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state=0)





