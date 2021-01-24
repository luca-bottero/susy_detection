#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import xgboost as xgb
import time
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

#%%
datas = pd.read_csv('./Data/SUSY.csv')
datas.columns = ["label","lepton 1 pT"," lepton 1 eta"," lepton 1 phi"," lepton 2 pT"," lepton 2 eta"," lepton 2 phi"," missing energy magnitude"," missing energy phi"," MET_rel"," axial MET"," M_R"," M_TR_2"," R"," MT2"," S_R"," M_Delta_R"," dPhi_r_b"," cos(theta_r1)"]

#%%
#EXPLORATORY ANALYSIS
#Data preparation
X_train, X_test, y_train, y_test = train_test_split(datas.drop(["label"], axis=1), datas["label"], test_size = 0.9)

datas.head()

if os.path.exists("./Results/Datas_scatter.png") == False:          #this plot is rather slow, so it is better not to redo every time
    datasScatter = sns.pairplot(datas.sample(50000), hue = 'label', plot_kws={'alpha': 0.1}, corner = True)       #use only a small fraction (about 0.1%) of datas due to size issue
    datasScatter.savefig("./Results/Datas_scatter.png", facecolor = 'white')

datasHist = datas.hist(bins = 50, figsize = (20,20))                    #Datas distribution histograms
plt.savefig("./Results/Datas_histograms.png", facecolor = 'white')

plt.figure(figsize = (20,20))
#datasCorr = plt.imshow(datas.corr(), cmap='viridis', interpolation='none')  #Datas correlation matrix (default: Fischer)
sns.heatmap(datas.corr(), annot=True, square=True, mask=np.triu(np.ones_like(datas.corr(), dtype=bool)))
plt.savefig("./Results/Datas_corr_matrix.png", facecolor = "white")

PolyFeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit_transform(X_train,y_train)
print(PolyFeatures)

# %%
#GRADIENT BOOSTED DECISION TREES scikitlearn
#VERY SLOW! (about 4 min for 10 trees and max_depth = 1)

#GBDT = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=2, verbose = 1).fit(X_train, y_train)

#%%
#XGBOOST MODEL
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

XGBScore = mean_squared_error(XGB.predict(xgb.DMatrix(X_test)),y_test)          #Mean squared error
XGBAccuracy = accuracy_score(XGB.predict(xgb.DMatrix(X_test)).round(),y_test)   #Accuracy
XGBAUC = roc_auc_score(XGB.predict(xgb.DMatrix(X_test)).round(),y_test)         #AUC score
print("Elapsed time: " + str(end - start))
#print("Root Mean Square error: " + str(XGBScore))
print("Fraction of correctly labelled data: " + str(XGBAccuracy))
print("Area under ROC curve: " + str(XGBAUC))

# %%
#NEURAL NETWORK MODEL

InputNum = len(datas.columns) - 1

inputs = keras.Input(shape=(InputNum,), name="Input")

x = layers.Dense(32)(inputs)
for i in range(2):
    #x = layers.Dense(36 + i*(5-i)*3)(x)
    x = layers.Dense(32,activation='sigmoid')(x)

outputs = layers.Dense(1,activation='sigmoid', name="Output")(x)

model = keras.Model(inputs=[inputs], outputs=[outputs], name = "NN_model")

model.summary()
model.save("./Model/NN_model_config")

#simple_sgd = keras.optimizers.Adadelta(lr = 0.01)  
simple_sgd = keras.optimizers.SGD(lr = 0.015, momentum = 0.015)

metrics = ['binary_accuracy','AUC', 'Precision']

model.compile(loss='binary_crossentropy', optimizer=simple_sgd, metrics=metrics)
 
max_epochs = 15
h = model.fit(X_train, y_train, batch_size=32, epochs=max_epochs, verbose = 2)

#%%
#MODEL EVALUATION
keras.utils.plot_model(model, "./Model/NN_model.png", show_shapes=True)

print("Evaluate on test data")
#results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

#plt.plot(h.history['loss'], label = 'loss')
plt.plot(h.history['auc'], label = "auc")
plt.plot(h.history['binary_accuracy'], label = "accuracy")
plt.legend( shadow=True, fontsize='x-large')
#plt.yscale('log')
plt.title('model loss')
plt.ylabel('metrics')
plt.xlabel('epoch')
# %%
#RESULTS EXPLORATORY ANALYSIS
predictions = np.round(model.predict(X_test))

result = pd.DataFrame(y_test)
result["prediction"] = predictions
result["correct"] = np.where(result["label"] == result["prediction"], True, False)
result.head(20)

CorrectIdxs = np.where(result["correct"] == True)
WrongIdxs = np.where(result["correct"] == False)

wrongHist = datas.iloc[WrongIdxs[0]].hist(bins = 50, figsize = (20,20))   #Wrongly labelled distribution histograms
plt.savefig("./Results/Wrong_histograms.png", facecolor = 'white')

plt.figure(figsize = (20,20))
wrongCorr = plt.imshow(datas.iloc[WrongIdxs[0]].corr(), cmap='viridis', interpolation='none')  #Datas correlation matrix (default: Fischer)
plt.colorbar(wrongCorr)
plt.savefig("./Results/Wrong_corr_matrix.png", facecolor = "white")


#%%

# %%
