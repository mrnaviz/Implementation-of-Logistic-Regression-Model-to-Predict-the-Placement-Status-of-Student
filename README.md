# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries which are used for the program.

2. Load the dataset.

3. Check for null data values and duplicate data values in the dataframe.

4. Apply logistic regression and predict the y output.

5. Calculate the confusion,accuracy and classification of the dataset.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NAVEEN KUMAR B
RegisterNumber:212222230091

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85,90,80]])

*/
```

## Output:
![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/4011ee82-18e4-4469-9e7f-4239293c644e)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/e670abde-b3a5-4dc1-89f4-fa4a81ab8814)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/46537eb3-5b3f-4212-aa31-1a2185ec8ba0)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/25c64a29-cfb2-46f8-b31b-e396ef9130da)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/7d2b7c5e-6cc6-40f3-9bce-371dbdb441d1)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/cf65b418-093d-4632-ad7e-d4dd373da214)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/7963afb9-4106-4fb7-b3c0-5726bd571fb1)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/00407fed-4258-4e00-b8c7-9e0d6c5dff1f)

![image](https://github.com/mrnaviz/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123350791/60a3feb1-59ba-4d3e-af45-b7ba52a35265)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
