import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_excel('Crop_recommendation.xlsx' , engine='openpyxl')

x = data.iloc[:, :7]
y = data.iloc[:, 7]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100)
rand = rf.fit(x_train, y_train)

pickle.dump(rand, open('crop.pkl', 'wb'))