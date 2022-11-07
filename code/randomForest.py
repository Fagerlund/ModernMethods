
import pandas as pd
from statistics import mean
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# Importing data
data = pd.read_csv("project_train.csv")

y = data["Label"]
y.to_frame()
X = data.drop("Label",axis=1)

# Removing bad data
id1 = X[(X["energy"] > 1)].index
X = X.drop(id1,axis=0)
y = y.drop(id1)
id2 = X[(X["loudness"] < -60)].index
X = X.drop(id2,axis=0)
y = y.drop(id2)
id3 = X[(X["loudness"] > 0)].index
X_df = X.drop(id3,axis=0)
y_df = y.drop(id3)

X_df = X_df.reset_index(drop=True)
y_df = y_df.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.20, random_state=None)

# Finding optimal parameters
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [4,5,6,7,8,9,10,11,12,13,14,15,20,30],
    'max_features': [4,5,6,7,8,9,10,11,12,13],
}

search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)

search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# Make pipeline
pipe = make_pipeline(None, RandomForestClassifier(n_estimators=200,max_depth=10,criterion='entropy',max_features=7))

# Train the model with CV and print mean validation score
scores = cross_val_score(pipe, X_train,y_train,cv=10)
print(mean(scores))

# Train the model with all training data and print test score
pipe.fit(X_train,y_train)
print(pipe.score(X_test,y_test))

# Predict labels from test data
X_TEST = pd.read_csv("project_test.csv")
y_pred = pipe.predict(X_TEST)
print(y_pred)