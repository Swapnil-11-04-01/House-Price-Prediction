# Importing modules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Loading Data
House_Pricing = pd.read_csv('House Pricing.csv')
House_Pricing.columns

Data = House_Pricing.dropna(axis = 0)

# Labels and features
Y = Data.Price

Data_Features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] # Features

X = Data[Data_Features]

# Model
Model = DecisionTreeClassifier(random_state=1)
Model.fit(X, Y)

# Training Model
print(House_Pricing.head())
