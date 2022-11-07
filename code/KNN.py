

# Project - KNN method
# SF2935 HT22
# Group 4

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt
import math
from sklearn import neighbors
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from scipy.spatial import distance_matrix
import scipy.stats
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import MinMaxScaler



# --------- Training ----------

# Read the training file
df = pd.read_csv('project_train.csv')

# Clean data from invalid values
df = df.drop(df[(df.energy >= 1) ].index)
df = df.drop(df[(df.energy <= 0) ].index)

df = df.drop(df[(df.loudness >= 0) ].index)
df = df.drop(df[(df.loudness <= -100) ].index)

# Scale the data using Min-Max Scaling
scaler = MinMaxScaler()
dfscale = scaler.fit_transform(df)
df = pd.DataFrame(dfscale, columns=df.columns)

# Take out the y column and remove it from the training-set
y = df.Label

# Tweak the data based on correlation matrix
x = df.drop('key', axis=1)
x = x.drop('liveness', axis=1)
corrmat = x.corr(method='pearson')

plt.matshow(corrmat)
plt.xticks(range(len(corrmat)), corrmat.columns, rotation ='vertical')
plt.yticks(range(len(corrmat)), corrmat.columns, rotation ='horizontal')

x = df.drop('Label', axis=1)


# Split data into train and test
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0)
# above not used when evaluating
x_train = x
y_train = y

# Fit and transform y
ley = preprocessing.LabelEncoder()
y_train = ley.fit_transform(y_train)
# y_test = ley.fit_transform(y_test)

# Fit and transform all columns
le = preprocessing.LabelEncoder()
x_train = x_train.apply(le.fit_transform)
# x_test = x_test.apply(le.fit_transform)



# --- KNN-TRAINING USING K=29 ---
k = 29
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train,y_train)


# # Test the data on the test-group
# test_preds_grid = knn.predict(x_test)
# test_mse = mean_squared_error(y_test, test_preds_grid)
# test_rmse = math.sqrt(test_mse)

# print("KNN (k=" + str(k) + "): " + str(test_rmse))
# print("KNN (k=" + str(k) + "): " + str(accuracy_score(y_test, test_preds_grid)))



# # --- GRIDSEARCH --- (NOT USED)
# parameters = {"n_neighbors": range(5, 35), "weights": ["distance"], "p": [1]}
# gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)
# gridsearch.fit(x_train, y_train)
# GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters)
# # print(gridsearch.best_params_)

# # Test the data on the test-group
# test_preds_grid = gridsearch.predict(x_test)
# test_mse = mean_squared_error(y_test, test_preds_grid)
# test_rmse = math.sqrt(test_mse)
# print("GridSearch (k=" + str(gridsearch.best_params_["n_neighbors"]) + "): " + str(test_rmse))
# print("GridSearch (k=" + str(gridsearch.best_params_["n_neighbors"]) + "): " + str(accuracy_score(y_test, test_preds_grid)))



# --------- Evaluation ----------

# # Read the evaluation file
df = pd.read_csv('project_test.csv')

# # Clean data from invalid values
df = df.drop(df[(df.energy >= 1)].index)
df = df.drop(df[(df.energy <= 0)].index)

df = df.drop(df[(df.loudness >= 0)].index)
df = df.drop(df[(df.loudness <= -100)].index)

# # Scale the data using Min-Max Scaling
scaler = MinMaxScaler()
dfscale = scaler.fit_transform(df)
df = pd.DataFrame(dfscale, columns=df.columns)

# # Fit and transform
df = df.apply(le.fit_transform)

KNNpred = knn.predict(df)

# Save the predicted values to txt-file
np.savetxt('testPred2.txt', KNNpred, fmt='%s')
