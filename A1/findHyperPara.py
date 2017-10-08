import regclass as rgr
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

data=scio.loadmat('poly_data.mat')
polyx=data['polyx']#1*100
polyy=data['polyy']#100*1
sampx=data['sampx']#1*50
sampy=data['sampy']#50*1
thtrue=data['thtrue']#6*1


def PolyFeaTran(x, d_phi):
    phi_x = np.zeros(d_phi + 1)
    for i in range(d_phi + 1):
        phi_x[i] = pow(x, i)
    return phi_x


def PolyFea(samplex, d_phi):
    n_samp = samplex.shape[1]
    Mat_Fea = np.zeros((d_phi + 1, n_samp))
    # print "Feature matrix shape", Mat_Fea.shape
    for i in range(n_samp):
        # print "Feature vector shape", FeaTran(sampx[0,i],d_phi).shape
        Mat_Fea[:, i] = PolyFeaTran(samplex[0, i], d_phi)
    return Mat_Fea



dim_Fea=5 #Feature dimension
variance=5 #variance of data
Fea_Trn=PolyFea(sampx,dim_Fea) #Feature of Training data
Fea_Tst=PolyFea(polyx,dim_Fea) #Feature of Testing data


regress=rgr.Regressor(Fea_Trn,sampy,variance)

#-----------------------------------------regularized LS------------------------------------------------------------
rgl_lambda=np.array([1.5,1.8,2,2.2,3])
Test_Error_rgl=np.zeros(rgl_lambda.shape)


for i in range(rgl_lambda.shape[0]):
    theta_Rgl = regress.Regul_Reg(rgl_lambda[i])
    Test_Pred_rgl = rgr.nonBayes_Pred(theta_Rgl, Fea_Tst)
    Test_Error_rgl[i] = rgr.Pred_Err(polyy, Test_Pred_rgl)


#plt.plot(rgl_lambda.transpose(),Test_Error_rgl.transpose(),'.',color='b')
#plt.xlim()
#plt.ylim()
#plt.show()
#choose 1.8 as rgl_lambda


#---------------------------------------------LASSO------------------------------------------------------------------
lasso_t=np.array([0.8,1,1.1,1.2])
Test_Error_las=np.zeros(lasso_t.shape)


for i in range(lasso_t.shape[0]):
    theta_las = regress.LASSO_Reg(lasso_t[i])
    Test_Pred_las = rgr.nonBayes_Pred(theta_las, Fea_Tst)
    Test_Error_las[i] = rgr.Pred_Err(polyy, Test_Pred_las)


#plt.plot(lasso_t.transpose(),Test_Error_las.transpose(),'.',color='b')
#plt.xlim()
#plt.ylim()
#plt.show()
#choose 1 as lasso_t

#---------------------------------------------Bayes-----------------------------------------------------------------
Bayes_alpha=np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17])
Test_Error_Bayes=np.zeros(Bayes_alpha.shape)


for i in range(Bayes_alpha.shape[0]):
    theta_Bayes = regress.Bay_Reg(Bayes_alpha[i])
    Test_Pred_Bayes = rgr.Bayes_Pred(Fea_Tst, theta_Bayes)
    Test_Error_Bayes[i] = rgr.Pred_Err(polyy, Test_Pred_Bayes['miu_pred'])


plt.plot(Bayes_alpha.transpose(),Test_Error_Bayes.transpose(),'.',color='b')
plt.xlim()
plt.ylim()
plt.show()
#choose 10 as Bayes_alpha







