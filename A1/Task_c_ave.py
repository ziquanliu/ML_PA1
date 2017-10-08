import regclass as rgr
import numpy as np
import scipy.io as scio
import random as rd
import matplotlib.pyplot as plt

data=scio.loadmat('poly_data.mat')
polyx=data['polyx']#1*100
polyy=data['polyy']#100*1
sampx=data['sampx']#1*50
sampy=data['sampy']#50*1
thtrue=data['thtrue']#6*1


dim=5
variance=5 #variance of data
hyperparam={'rgl':1.8,'lasso':1,'Bayes':1}#hyperparameter

num_points = 44
E_Robust = np.zeros((num_points))
E_LSE = np.zeros((num_points))
E_Rgl = np.zeros((num_points))
E_Bayes = np.zeros((num_points))
E_Las = np.zeros((num_points))


for j in range(100):
    for i in range(50-num_points, 50):
        Predict_E = rgr.train_on_sample(sampx, sampy, polyx, polyy, dim, variance, i, hyperparam)
        E_Robust[i +num_points- 50] += Predict_E['Rob']
        E_LSE[i +num_points- 50] += Predict_E['LSE']
        E_Rgl[i +num_points- 50] += Predict_E['Rgl']
        E_Bayes[i +num_points- 50] += Predict_E['Bayes']
        E_Las[i +num_points- 50] += Predict_E['Las']


E_Robust=E_Robust/100
E_LSE=E_LSE/100
E_Rgl=E_Rgl/100
E_Bayes=E_Bayes/100
E_Las=E_Las/100

np.save('Error of Robust', E_Robust)
np.save('Error of LSE', E_LSE)
np.save('Error of Rgl', E_Rgl)
np.save('Error of Bayes', E_Bayes)
np.save('Error of Lasso', E_Las)