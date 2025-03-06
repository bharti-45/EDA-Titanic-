# EDA-Titanic-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("C:\\Users\\sharm\\Downloads\\Titanic-Dataset.csv")
df.info()
print(df.head())

#summary statistics
print(df.describe())
print(df.describe(include=["object"]))

#handle missing values
df.dropna(inplace=True)

#set style for visualizations
sns.set(style ="whitegrid")

#plot survival count
plt.figure(figsize=(6,4))
sns.countplot(data=df,x="survived",palette="coolwarm")
plt.title("Survival count")
plt.xticks(ticks=[0,1],labels=["Not Survived", "Survived"])
plt.ylabel("Count")
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived", palette="coolwarm")
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived", palette="coolwarm")
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Age Distribution by Survival
plt.figure(figsize=(8, 5))
sns.histplot(df[df["Survived"] == 1]["Age"], bins=30, kde=True, color="green", label="Survived")
sns.histplot(df[df["Survived"] == 0]["Age"], bins=30, kde=True, color="red", label="Not Survived")
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

# Data Preprocessing
imputer = SimpleImputer(strategy='median')  # Fill missing Age with median
df['Age'] = imputer.fit_transform(df[['Age']])
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)  # Drop non-useful columns
df.dropna(subset=['Embarked'], inplace=True)  # Drop rows with missing Embarked

# Encode categorical variables
label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])
df['Embarked'] = label_enc.fit_transform(df['Embarked'])

# Define features and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
