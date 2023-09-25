# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values



## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mahisha S
RegisterNumber:  212222040095

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### Placement Data:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/4ff5640b-a4ff-4341-a634-24c87fd674ea)

### Salary Data:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/408cd8e6-4604-40ad-aaa8-ae0f9c2b2d94)

### Checking the null() function:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/d05bf68b-10e4-410f-a95d-921108157898)

### Data Duplicate:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/7696dd8b-a60a-4d70-8c56-84f274a36615)

### Print Data:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/ad539a16-7586-4c6d-b425-4b1df2f12426)

### Data-status:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/f7d5d1ab-bba1-4df6-b93e-fab542012612)

### y_prediction array:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/99a421e5-f45f-49f9-b623-3ff3d6b25a7e)

### Accuracy value:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/e9f28b7c-ccda-4328-9db8-46fc1b532db4)

### Confusion array:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/d286f5b0-1807-4972-965e-6c8c7861b723)

### Classification report:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/b32c6aa4-3b7e-4f3b-a859-0cf14dd31dc3)

### Prediction of LR:
![image](https://github.com/mahishasivakumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559812/fc9e2ead-5107-492b-8453-899d3c9280a5)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
