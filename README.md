# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Aprajith R
RegisterNumber:  212222080006


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
```

## Output:
![Screenshot 2024-02-25 221823](https://github.com/Aprajith-R/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161153978/15f5dd05-f946-46d2-b6c3-903b0397365c)
![Screenshot 2024-02-25 221638](https://github.com/Aprajith-R/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/161153978/df6780ec-09f1-4b99-b0b9-db593943c908)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
