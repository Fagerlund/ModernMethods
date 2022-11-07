
from keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#train_data.hist(bins=12, figsize=(15, 10))
#train_data["energy"]=(train_data['energy'] <= 0)|(train_data['energy']>=1)
#train_data["loudness"]=(train_data['loudness'] <= -100)|(train_data['loudness']>=0)
#The listed intervalls below are given in the assginment 


train_data=pd.read_csv("data1.csv")
train_data = train_data.drop(train_data[(train_data.energy >= 1) ].index)
train_data = train_data.drop(train_data[(train_data.energy <= 0) ].index)

train_data = train_data.drop(train_data[(train_data.loudness >= 0) ].index)
train_data = train_data.drop(train_data[(train_data.loudness <= -100) ].index)

#Divided into liked label and disliked labeled, plotted to give an overview.

data_1=train_data.loc[train_data['Label'] == 1]
data_0=train_data.loc[train_data['Label'] == 0]

data_0.hist(bins=30)#, figsize=(15, 10))
data_1.hist(bins=30)#, figsize=(15, 10))
train_data, test_data = train_test_split(train_data, test_size=0.2)
Y_Train=train_data["Label"]
Y_Test=test_data["Label"]
#X=train_data.drop(columns='key')
#normalization

#X_Train = pd.get_dummies(data=train_data, columns=['key'])

X_Train=train_data
X_Train = X_Train.drop(columns=['key'])
X_Train = X_Train.drop(columns=['mode'])
#X_Train = X_Train.drop(columns=['instrumentalness'])
X_Train = X_Train.drop(columns=['tempo'])

X_Train =(X_Train-X_Train.mean())/X_Train.std() #<- this one works well
X_Train=X_Train.drop(columns=['Label'])


#X_Test = pd.get_dummies(data=test_data, columns=['key'])

X_Test = test_data
X_Test =X_Test.drop(columns=['key'])
X_Test =X_Test.drop(columns=['mode'])
#X_Test = X_Test.drop(columns=['instrumentalness'])
X_Test = X_Test.drop(columns=['tempo'])

X_Test =(X_Test-X_Test.mean())/X_Test.std() #<- this one works well
X_Test=X_Test.drop(columns=['Label'])
#X=X.drop(columns=['Label'])
#X=X.drop(columns='Label')

#X=X.drop(columns=['key'])
# %% Done with data-Prepp

from keras.models import Sequential
from keras.layers import Dense

#model = Sequential()
#model.add(Dense(8,input_dim=len(X_Train.columns),activation="relu"))
#model.add(Dense(10,activation="relu"))
#model.add(Dense(6,activation="relu"))
#model.add(Dense(4,activation="relu"))
##model.add(Dense(5,activation="relu"))
##model.add(Dense(6,activation="relu"))
##model.add(Dense(5,activation="relu"))
#model.add(Dense(1,activation="sigmoid"))

#model.summary()
# %%
from tensorflow import keras
# learnn=[0.1,0.05,0.01,0.005,0.0001,0.00005,0.00001]
# for learn in learnn:
#     #opt = keras.optimizers.Adam(learning_rate=0.001)
#     model = Sequential()
#     model.add(Dense(8,input_dim=len(X_Train.columns),activation="relu"))
#     #model.add(Dense(10,activation="relu"))
#     model.add(Dense(6,activation="relu"))
#     model.add(Dense(4,activation="relu"))
#     #model.add(Dense(5,activation="relu"))
#     #model.add(Dense(6,activation="relu"))
#     #model.add(Dense(5,activation="relu"))
#     model.add(Dense(1,activation="sigmoid"))

#     opt = keras.optimizers.Adam(learning_rate=learn)
#     model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    
#     model.fit(x=X_Train,y=Y_Train,epochs=50)#,verbose=1)
    
#     print(model.evaluate(X_Test,(Y_Test)))
#%%
#Found the former optimal learning rate
model = Sequential()
model.add(Dense(9,input_dim=len(X_Train.columns),activation="relu"))
model.add(Dense(9,activation="relu"))
#model.add(Dense(6,activation="relu"))
#model.add(Dense(10,activation="relu"))
#model.add(Dense(5,activation="relu"))
model.add(Dense(9,activation="relu"))
#model.add(Dense(5,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

model.fit(x=X_Train,y=Y_Train,epochs=40)#,verbose=1)

print(model.evaluate(X_Test,(Y_Test)))


#%%


org_data=pd.read_csv("data1.csv")
org_data = org_data.drop(org_data[(org_data.energy >= 1) ].index)
org_data = org_data.drop(org_data[(org_data.energy <= 0) ].index)

org_data = org_data.drop(org_data[(org_data.loudness >= 0) ].index)
org_data = org_data.drop(org_data[(org_data.loudness <= -100) ].index)


Y_org=org_data["Label"]
org_data = org_data.drop(columns=['key'])
org_data = org_data.drop(columns=['mode'])
#X_Train = X_Train.drop(columns=['instrumentalness'])
org_data = org_data.drop(columns=['tempo'])

org_data =(org_data-org_data.mean())/org_data.std() #<- this one works well
org_data=org_data.drop(columns=['Label'])



model = Sequential()
model.add(Dense(9,input_dim=len(org_data.columns),activation="relu"))
model.add(Dense(9,activation="relu"))
#model.add(Dense(6,activation="relu"))
#model.add(Dense(10,activation="relu"))
#model.add(Dense(5,activation="relu"))
model.add(Dense(9,activation="relu"))
#model.add(Dense(5,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

model.fit(x=org_data,y=Y_org,epochs=40)#,verbose=1)




#%%

Predict=pd.read_csv("project_test.csv")


Predict = Predict.drop(columns=['key'])
Predict = Predict.drop(columns=['mode'])
#X_Train = X_Train.drop(columns=['instrumentalness'])
Predict = Predict.drop(columns=['tempo'])

Predict =(Predict-Predict.mean())/Predict.std() #<- this one works well

Predicted_Vals=model.predict(Predict)
Predicted_Vals2=[]
for val in Predicted_Vals:
    if val<0.5:
        Predicted_Vals2.append(0)
        
    else:
        Predicted_Vals2.append(1)


