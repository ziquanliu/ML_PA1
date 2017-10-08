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


num_sam=[6,10,20,30,40]
new_train=rgr.Samp_Data(sampx,sampy,num_sam[4])
print new_train
fea_train=rgr.PolyFea(new_train['x'].transpose(),dim)
fea_test=rgr.PolyFea(polyx,dim)
rgr.Task_b(fea_train,new_train['x'],new_train['y'],fea_test,polyx,polyy,variance,hyperparam,new_train['y'])





