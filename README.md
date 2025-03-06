# EDA-Titanic-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os

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
