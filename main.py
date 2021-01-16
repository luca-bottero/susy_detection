#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# %%
