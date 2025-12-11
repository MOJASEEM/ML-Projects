import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import SimpleImputer

df = pd.read_csv("D:/Data Science/customer_churn_prediction/ANN/telecom_customer_churn/Telco-Customer-Churn.csv")

print(df.sample(5))
df.drop('customerID',axis='columns',inplace=True)
print(df.dtypes)
print(df.TotalCharges.values)
df.TotalCharges = pd.to_numeric(df.TotalCharges,errors='coerce')
print(df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()])

print(df[df.TotalCharges!=' '].shape)
df1 = df[df.TotalCharges!=' ']
print(df1.shape)
print(df1.dtypes)
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.TotalCharges.values
print(df1[df1.Churn=='No'])
tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]

plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()
mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges      
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges      

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.show()
def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}') 
print_unique_col_values(df1)
df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)
print_unique_col_values(df1)
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)
for col in df1:
    print(f'{col}: {df1[col].unique()}') 
df1['gender'].replace({'Female':1,'Male':0},inplace=True)
print(df1.gender.unique())
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
print(df2.columns)
print(df2.sample(5))
print(df2.dtypes)
cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
print(df2.sample(5))
X = df2.drop('Churn',axis='columns')
y = df2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape)
print(len(X_train.columns))
def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(26, input_shape=(26,), activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100)
    print("\n# Evaluate on test data",model.evaluate(X_test, y_test))
    yp = model.predict(X_test)
    y_pred = []
    for element in yp:
        if element > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print(model.evaluate(X_test, y_test))
    
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    
    print("Classification Report: \n", classification_report(y_test, y_preds))

    print(y_pred[:10])
    return y_pred

y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)
print(classification_report(y_test,y_preds))
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_preds)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
imputer = SimpleImputer(strategy='mean')  # Or 'median' or 'most_frequent'
X_imputed = imputer.fit_transform(X)
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X_imputed, y)
y_sm.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)
print(classification_report(y_test,y_preds))
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_preds)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

