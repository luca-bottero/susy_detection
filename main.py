#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

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

# %%
#GRADIENT BOOSTED DECISION TREES scikitlearn
#VERY SLOW! (about 4 min for 10 trees and max_depth = 1)
X_train, X_test, y_train, y_test = train_test_split(datas.drop(["label"], axis=1), datas["label"], test_size = 0.1)

GBDT = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=2, verbose = 1).fit(X_train, y_train)
#%%
GBDTParam = GBDT.get_params()
#%%
GBDTParam









# %%
