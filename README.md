# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
import pandas as pd

import numpy as np

df=pd.read_csv("bmi.csv")

df

<img width="496" height="640" alt="image" src="https://github.com/user-attachments/assets/364b4cbe-d7b1-4d97-adce-32f50d5e7abf" />

df.head()

<img width="435" height="290" alt="image" src="https://github.com/user-attachments/assets/7c86eed5-e544-4fb3-8a7d-e1692ae3ba42" />

df.dropna()

<img width="459" height="625" alt="image" src="https://github.com/user-attachments/assets/ed5ce24d-6b2d-47b7-b2f8-844c0a3eb93a" />

max_vals=np.max(np.abs(df[['Height','Weight']]))

max_vals

<img width="89" height="40" alt="image" src="https://github.com/user-attachments/assets/9b6c2110-99c8-4921-9c12-3c661ae8da3b" />


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head(10)

<img width="478" height="556" alt="image" src="https://github.com/user-attachments/assets/39cc60de-289d-4f30-8b0c-7d653ed46a61" />

df1=pd.read_csv("bmi.csv")

df2=pd.read_csv("bmi.csv")

df3=pd.read_csv("bmi.csv")

df4=pd.read_csv("bmi.csv")

df5=pd.read_csv("bmi.csv")

df1

<img width="488" height="641" alt="image" src="https://github.com/user-attachments/assets/dd2faf12-dd94-4673-8072-b260f907a5c1" />

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])

df.head(10)

<img width="474" height="527" alt="image" src="https://github.com/user-attachments/assets/d518c872-dc06-49fe-bb7a-a30fecb2c7f7" />

from sklearn.preprocessing import Normalizer

scaler=Normalizer()

df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])

df2

<img width="488" height="641" alt="image" src="https://github.com/user-attachments/assets/1e724b01-dabb-46c9-b3c3-3aabf788236e" />

from sklearn.preprocessing import MaxAbsScaler

max1=MaxAbsScaler()

df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])

df3


<img width="516" height="632" alt="image" src="https://github.com/user-attachments/assets/eab569b8-99db-4294-a687-27f8bf7bc2e0" />


from sklearn.preprocessing import RobustScaler

roub=RobustScaler()

df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])

df4

<img width="551" height="638" alt="image" src="https://github.com/user-attachments/assets/f799ac6e-d8cf-4b42-a2d3-e7eaadca06dc" />

from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif

from sklearn.feature_selection import chi2

data=pd.read_csv("income(1) (1).csv")

data

<img width="1004" height="293" alt="image" src="https://github.com/user-attachments/assets/57bd416d-40df-4bab-a40f-b5d61ba21cb6" />

import pandas as pd

import numpy as np

data1=pd.read_csv(r"C:\Users\acer\Downloads\titanic_dataset (1).csv")

data1


<img width="1381" height="486" alt="image" src="https://github.com/user-attachments/assets/0bb4e5bb-2bfd-4c9e-be47-c1276ae7cc44" />

data1=data1.dropna()

x=data1.drop(['Survived','Name','Ticket'],axis=1)

y=data1['Survived']

data1['Sex']=data1['Sex'].astype('category')

data1['Cabin']=data1['Cabin'].astype('category')

data1['Embarked']=data1['Embarked'].astype('category')

data1

<img width="1293" height="493" alt="image" src="https://github.com/user-attachments/assets/79cd7016-7179-4753-9731-0157c0cbf15c" />

data1=data1.dropna()

x=data1.drop(['Survived','Name','Ticket'],axis=1)

y=data1['Survived']

data1['Sex']=data1['Sex'].astype('category')

data1['Cabin']=data1['Cabin'].astype('category')

data1['Embarked']=data1['Embarked'].astype('category')

data1['Sex']=data1['Sex'].cat.codes

data1['Cabin']=data1['Cabin'].cat.codes

data1['Embarked']=data1['Embarked'].cat.codes

data1

<img width="1293" height="489" alt="image" src="https://github.com/user-attachments/assets/acbb0ec8-2783-4087-9fc9-907c012a8d29" />

from sklearn.feature_selection import SelectKBest,chi2

import pandas as pd

k=5

selector=SelectKBest(score_func=chi2,k=k)

x=pd.get_dummies(x)

x_new=selector.fit_transform(x,y)

x_encoded=pd.get_dummies

selector=SelectKBest(score_func=chi2,k=5)

x_new=selector.fit_transform(x_encoded,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

<img width="846" height="79" alt="image" src="https://github.com/user-attachments/assets/6227f1b2-1fab-4214-9329-348abacb2d03" />

from sklearn.feature_selection import SelectKBest,f_regression

import pandas as pd

selector=SelectKBest(score_func=f_regression,k=5)

x_new=selector.fit_transform(x_encoded,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

<img width="801" height="55" alt="image" src="https://github.com/user-attachments/assets/02212dab-a790-44f8-a849-c99c6496384c" />

from sklearn.feature_selection import SelectKBest,mutual_info_classif

import pandas as pd

selector=SelectKBest(score_func=mutual_info_classif,k=5)

x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

<img width="741" height="56" alt="image" src="https://github.com/user-attachments/assets/1f9b62ed-79e3-4f55-acb6-aa230a2653ec" />

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

sfm=SelectFromModel(model,threshold='mean')

x=pd.get_dummies(x)

sfm.fit(x,y)

selected_features=x.columns[sfm.get_support()]

print("Selected Features:")

print(selected_features)

<img width="845" height="125" alt="image" src="https://github.com/user-attachments/assets/e16ffbef-37c6-42ab-a14c-b89ee5c2a410" />

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x,y)

feature_selection=model.feature_importances_
threshold=0.1
selected_features=x.columns[feature_selection>threshold]

print("Selected Features:")

print(selected_features)


<img width="825" height="66" alt="image" src="https://github.com/user-attachments/assets/564a59d0-1f7d-4ea0-8de2-737ba0f41eba" />


model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x,y)

feature_importance=model.feature_importances_
threshold=0.15

selected_features=x.columns[feature_importance>threshold]

print("Selected Features:")

print(selected_features)

<img width="955" height="67" alt="image" src="https://github.com/user-attachments/assets/602bf561-fbbd-408d-9216-18fbdc964b33" />



# RESULT:
       # INCLUDE YOUR RESULT HERE
