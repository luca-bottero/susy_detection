#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import xgboost as xgb
import time
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score


#%%
datas = pd.read_csv('./Data/SUSY.csv')
datas.columns = ["label","lepton 1 pT"," lepton 1 eta"," lepton 1 phi"," lepton 2 pT"," lepton 2 eta"," lepton 2 phi"," missing energy magnitude"," missing energy phi"," MET_rel"," axial MET"," M_R"," M_TR_2"," R"," MT2"," S_R"," M_Delta_R"," dPhi_r_b"," cos(theta_r1)"]

#%%
#Exploratory analisys
datas.head()

datasHist = datas.hist(bins = 50, figsize = (20,20))                    #Datas distribution histograms
plt.savefig("./Results/Datas_histograms.png", facecolor = 'white')

plt.figure(figsize = (20,20))
datasCorr = plt.imshow(datas.corr(), cmap='viridis', interpolation='none')  #Datas correlation matrix (default: Fischer)
plt.colorbar(datasCorr)
plt.savefig("./Results/Datas_corr_matrix.png", facecolor = "white")

#%%
#Data preparation
X_train, X_test, y_train, y_test = train_test_split(datas.drop(["label"], axis=1), datas["label"], test_size = 0.9)

# %%
#GRADIENT BOOSTED DECISION TREES scikitlearn
#VERY SLOW! (about 4 min for 10 trees and max_depth = 1)

#GBDT = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=2, verbose = 1).fit(X_train, y_train)

#%%
start = time.time()

dtrain = xgb.DMatrix(X_train, label = y_train)
XGBParam = {"eta": 1, 
            "objective": "binary:hinge", 
            "max_depth" : 9, 
            "verbosity":2, 
            "tree_method":"gpu_hist",
            "num_parallel_tree": 1}
num_round = 10
XGB = xgb.train(XGBParam, dtrain, num_round)

end = time.time()

XGBScore = np.sqrt(mean_squared_error(XGB.predict(xgb.DMatrix(X_test)),y_test))
XGBAccuracy = accuracy_score(XGB.predict(xgb.DMatrix(X_test)).round(),y_test)
XGBAUC = roc_auc_score(XGB.predict(xgb.DMatrix(X_test)).round(),y_test)
print("Elapsed time: " + str(end - start))
#print("Root Mean Square error: " + str(XGBScore))
print("Fraction of correctly labeled data: " + str(XGBAccuracy))
print("Area under ROC curve: " + str(XGBAUC))


# %%
#NEURAL NETWORK MODEL

InputNum = len(datas.columns) - 1

inputs = keras.Input(shape=(InputNum,), name="Input")
x = layers.Dense(20)(inputs)
for i in range(6):
    x = layers.Dense(20)(x)
outputs = layers.Dense(1, name="Output")(x)

model = keras.Model(inputs=[inputs], outputs=[outputs])

#keras.utils.plot_model(model, "NN_model.png", show_shapes=True)

simple_sgd = keras.optimizers.Adadelta(lr=0.01)  
model.compile(loss='binary_crossentropy',
  optimizer=simple_sgd, metrics=['accuracy'])

max_epochs = 25
h = model.fit(X_train, y_train, batch_size=32,
  epochs=max_epochs, verbose = 2)
#%%
print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

plt.plot(h.history['loss'])
plt.yscale('log')
plt.title('model loss')
plt.ylabel('log(loss)')
plt.xlabel('epoch')

# %%
predictions = np.round(model.predict(X_test))S

print(predictions)
print(y_test)
print(predictions != y_test)

incorrects = X_test[predictions != y_test]
incorrects

# %%


# %%
