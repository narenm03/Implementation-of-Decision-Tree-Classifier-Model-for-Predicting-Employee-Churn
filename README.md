# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1.START

STEP 2. Import the required libraries.

STEP 3. Upload the csv file and read the dataset.

STEP 4. Check for any null values using the isnull() function.

STEP 5. From sklearn.tree inport DecisionTreeRegressor.

STEP 6. Import metrics and calculate the Mean squared error.

STEP 7. Apply metrics to the dataset, and predict the output.

STEP 8.END

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NARENDHARAN.M
RegisterNumber:  212223230134
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Head
![Screenshot 2024-09-20 102257](https://github.com/user-attachments/assets/9714be12-3a22-4ce9-b6b0-ad370809b989)

### Info

![Screenshot 2024-09-20 101841](https://github.com/user-attachments/assets/31cdafb0-45bf-4885-aff0-a90e8ee16769)

### salary head

![Screenshot 2024-09-20 102331](https://github.com/user-attachments/assets/08abc5ea-b4a1-4e2b-81c9-2f82236bbc7b)

### x.head()
![Screenshot 2024-09-20 102331](https://github.com/user-attachments/assets/5f5666d9-58cb-4ab1-9398-d7e8d3c45dc4)
![Screenshot 2024-09-20 104948](https://github.com/user-attachments/assets/375ea967-58df-4a61-82ca-af1322e2fa6e)

### Accuracy
![Screenshot 2024-09-20 103126](https://github.com/user-attachments/assets/cffbfb8f-44e4-41b6-8cf7-5c43e86b88f8)

### Data Prediction
![Screenshot 2024-09-20 103137](https://github.com/user-attachments/assets/12b037fa-7b13-451e-9c05-d1c8166c16c3)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
