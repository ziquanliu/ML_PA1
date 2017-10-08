import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
import random as rd

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


class Regressor(object):
    def __init__(self,train_feature,sampley,sigma_sq):
        self.feat_trn = train_feature
        self.sampy = sampley
        self.var=sigma_sq

    def LSE_Reg(self):
        Temp_Mat = self.feat_trn.dot(self.feat_trn.transpose())
        Temp_Mat_I = np.linalg.inv(Temp_Mat)
        theta = Temp_Mat_I.dot(self.feat_trn).dot(self.sampy)
        return theta



    def Regul_Reg(self,rgl_lamda):
        d_f = self.feat_trn.shape[0]
        M_lamda = np.eye(d_f, d_f)
        M_lamda[0, 0] = 0
        Temp_mat = self.feat_trn.dot(self.feat_trn.transpose()) + rgl_lamda * M_lamda
        Temp_mat_I = np.linalg.inv(Temp_mat)
        theta = Temp_mat_I.dot(self.feat_trn).dot(self.sampy)
        return theta


    def LASSO_Reg(self,t):
        d_f = self.feat_trn.shape[0]
        c = -2 * self.feat_trn.dot(self.sampy)
        A = 2 * self.feat_trn.dot(self.feat_trn.transpose())
        M_temp1 = np.column_stack((A, -A))
        M_temp2 = np.column_stack((-A, A))
        Q = np.row_stack((M_temp1, M_temp2))  # quadratic matrix
        # print "shape of Q",Q.shape
        q = np.row_stack((c, -c))  # linear vector
        # print "shape of q",q.shape
        I_d = np.eye(d_f, d_f)
        Z_d = np.zeros((d_f, d_f))
        con_temp1 = np.column_stack((I_d, I_d))
        con_temp2 = np.column_stack((-I_d, Z_d))
        con_temp3 = np.column_stack((Z_d, -I_d))
        cons = np.row_stack((con_temp1, con_temp2, con_temp3))
        # print "shape of constraint", cons.shape
        h_temp1 = t * np.ones((d_f, 1))
        h_temp2 = np.zeros((2 * (d_f), 1))
        h = np.row_stack((h_temp1, h_temp2))
        # print "shape of h",h.shape
        Q_cvx = cvx.matrix(Q, tc='d')
        q_cvx = cvx.matrix(q, tc='d')
        G_cvx = cvx.matrix(cons, tc='d')
        h_cvx = cvx.matrix(h, tc='d')
        sol = cvx.solvers.qp(Q_cvx, q_cvx, G_cvx, h_cvx)
        theta_pm = sol['x']
        # print "theta_pm",theta_pm
        # print "theta_pm_t1", theta_pm[0:d_f+1]
        # print "theta_pm_t2", theta_pm[d_f+1:2*d_f+2]
        theta = theta_pm[0:d_f] - theta_pm[d_f:2 * d_f]
        return theta


    def LASSO_Reg_Alt(self,Las_lambda):
        d_f=self.feat_trn.shape[0]
        f_up=self.feat_trn.dot(self.sampy)
        f=Las_lambda*np.eye(d_f*2,1)-np.row_stack((f_up,-f_up))
        H_temp=self.feat_trn.dot(self.feat_trn.transpose())
        H_up=np.column_stack((H_temp,-H_temp))
        H_down=np.column_stack((-H_temp,H_temp))
        H=np.row_stack((H_up,H_down))
        G=-np.eye(2*d_f,2*d_f)
        cons=np.zeros((2*d_f,1))
        f_cvx=cvx.matrix(f,tc='d')
        H_cvx=cvx.matrix(H,tc='d')
        G_cvx=cvx.matrix(G,tc='d')
        cons_cvx=cvx.matrix(cons,tc='d')
        sol=cvx.solvers.qp(H_cvx,f_cvx,G_cvx,cons_cvx)
        x=sol['x']
        print 'optimizatin result +',x[0:d_f]
        print 'optimization result -',x[d_f:2*d_f]
        theta = x[0:d_f] - x[d_f:2 * d_f]
        return theta



    def Robust_Reg(self):
        d_f = self.feat_trn.shape[0]
        Mat_Fea = self.feat_trn.transpose()
        n = self.sampy.shape[0]
        G_top = np.column_stack((-Mat_Fea, -np.eye(n, n)))
        G_down = np.column_stack((Mat_Fea, -np.eye(n, n)))

        G = np.row_stack((G_top, G_down))
        y = np.zeros((n, 1))
        for i in range(n):
            y[i] = self.sampy[i]
        h = np.row_stack((-y, y))
        c = np.row_stack((np.zeros((d_f, 1)), np.ones((n, 1))))
        G_cvx = cvx.matrix(G, tc='d')
        # print "G shape", G.shape
        h_cvx = cvx.matrix(h, tc='d')
        # print "h shape", h.shape
        c_cvx = cvx.matrix(c, tc='d')
        # print "c shape", c.shape
        sol = cvx.solvers.lp(c_cvx, G_cvx, h_cvx)
        x = sol['x']
        theta = x[0:d_f]
        return theta

    def Bay_Reg(self,c_alpha):
        d_f = self.feat_trn.shape[0]
        # calculate posterior
        big_sgm_tmp = (self.feat_trn.dot(self.feat_trn.transpose()) / self.var) + (np.eye(d_f, d_f) / c_alpha)
        big_sgm = np.linalg.inv(big_sgm_tmp)  # estimated covariance
        miu = big_sgm.dot(self.feat_trn).dot(self.sampy) / self.var  # estimated mean
        Bay_post = {'mean': miu, 'cov': big_sgm}
        return Bay_post



def nonBayes_Pred(theta,Mat_Fea_Tst):
    n_samp=Mat_Fea_Tst.shape[1]
    y_E=np.zeros(n_samp)
    for i in range(n_samp):
        y_E[i]=Mat_Fea_Tst[:,i].transpose().dot(theta)
    return y_E


def Bayes_Pred(Mat_Feat,theta_post):
    n_test=Mat_Feat.shape[1]
    miu_pred=np.zeros((n_test,1))
    sgm_pred = np.zeros((n_test, 1))
    for i in range(n_test):
        miu_pred[i]=Mat_Feat[:,i].transpose().dot(theta_post['mean'])
        sgm_pred[i]=Mat_Feat[:,i].transpose().dot(theta_post['cov']).dot(Mat_Feat[:,i])
    Bay_pred={'miu_pred':miu_pred,'sgm_pred':sgm_pred}
    return Bay_pred


def nonBayes_Plt(smpx,smpy,smp_pre_y,polyx,polyy,poly_pre_y,name,save_name):
    f = plt.figure(name)
    plt.plot(polyx.transpose(),polyy,'-',color='k',label='True Poly')
    plt.plot(polyx.transpose(),poly_pre_y,'.',color='r',label='Estimated Poly')
    #plt.plot(smpx.transpose(),smpy,'.',color='green',label='True Sample')
    #plt.plot(smpx.transpose(),smp_pre_y,'.',color='b',label='Estimated Sample')
    plt.legend()
    plt.title(name)
    plt.savefig(save_name+'.eps',dpi=300)
    plt.show()



def Bayes_Ply(smpx,smpy,Train_Pred,polyx,polyy,Test_Pred,MSE):
    plt.plot(polyx.transpose(), polyy, '-', color='k', label='True Poly')
    plt.errorbar(polyx.transpose(), Test_Pred['miu_pred'], yerr=np.sqrt(Test_Pred['sgm_pred']), fmt='.',
                 ecolor='red', color='red', label='Estimated Poly')
    #plt.plot(smpx.transpose(), smpy, '.', color='green', label='True Sample')
    #plt.errorbar(smpx.transpose(), Train_Pred['miu_pred'], yerr=np.sqrt(Train_Pred['sgm_pred']), fmt='.',
    #             ecolor='red', color='red', label='Estimated Sample')
    plt.legend()
    plt.title('Bayes Regression: MSE='+str(MSE))
    plt.savefig('BayesRegression.eps', dpi=300)
    plt.show()


def Error_Plot(error,name,num):
    f = plt.figure(name)
    plt.plot(range(50-num,50),error,'.-',color='red')
    plt.title(name)
    plt.savefig(name+'.eps',dpi=300)


def Pred_All_Err(truey,Test_Pred,Bayes_Test_Pred):
    Pred_Error = {}
    n_train = truey.shape[0]

    for key in Test_Pred:
        error_sq = np.square(np.reshape(Test_Pred[key], (n_train, 1)) - truey)
        error = np.sum(error_sq) / n_train
        Pred_Error[key] = error

    Bayes_err_sq = np.square(Bayes_Test_Pred['miu_pred'] - truey)
    Pred_Error['Bayes'] = np.sum(Bayes_err_sq) / n_train
    return Pred_Error


def Pred_Err(truey,Test_Pred):
    n_train = truey.shape[0]
    error_sq = np.square(np.reshape(Test_Pred, (n_train, 1)) - truey)
    error = np.sum(error_sq) / n_train
    return error


#sample data from training set sampx and sampy
def Samp_Data(smpx,smpy,num):
    b=range(smpx.shape[1])
    A = rd.sample(b,num)#sample  num different points
    New_Samp={'x':np.zeros((num,1)),'y':np.zeros((num,1))}
    for i in range(num):
        New_Samp['x'][i,0]=smpx[0,A[i]]
        New_Samp['y'][i,0]=smpy[A[i],0]
    return New_Samp


#train on different size of data and compute error
def train_on_sample(smpx,smpy,polyx,polyy,dim,variance,num,hyperp):
    New_Samp = Samp_Data(smpx, smpy, num)
    Fea_Train = PolyFea(New_Samp['x'].transpose(), dim)
    Fea_Test = PolyFea(polyx, dim)
    regress = Regressor(Fea_Train, New_Samp['y'], variance)
    rgl_lambda = hyperp['rgl']  # hyperparameter of regulized LS
    lasso_t = hyperp['lasso']  # hyperparameter of LASSO
    Bay_alpha = hyperp['Bayes']  # hyperparameter of Bayes
    print Fea_Train.shape[1]
    if dim<Fea_Train.shape[1]:
        theta = {'LSE': regress.LSE_Reg(), 'Rgl': regress.Regul_Reg(rgl_lambda), 'Las': regress.LASSO_Reg(lasso_t),
             'Rob': regress.Robust_Reg()}
    else:
        theta = {'Rgl': regress.Regul_Reg(rgl_lambda), 'Las': regress.LASSO_Reg(lasso_t)}

    Bayes_theta = regress.Bay_Reg(Bay_alpha)
    Test_Pred = {}
    for key in theta:
        Test_Pred[key] = nonBayes_Pred(theta[key], Fea_Test)
    Bayes_Test_Pred = Bayes_Pred(Fea_Test, Bayes_theta)
    Pred_Error = Pred_All_Err(polyy, Test_Pred, Bayes_Test_Pred)
    return Pred_Error


def Add_Outliers(smpx,smpy,num):
    b = range(smpx.shape[1])
    A = rd.sample(b, num)  # sample  num different points
    New_Samy=smpy
    for i in range(num):
        New_Samy[A[i]]=10**3*np.random.random()
    return New_Samy



def Task_b(Fea_Trn,smpx,smpy,Fea_Tst,polyx,polyy,var,hyperp,org_samy):
    regress = Regressor(Fea_Trn, smpy, var)
    rgl_lambda = hyperp['rgl']  # hyperparameter of regulized LS
    lasso_t = hyperp['lasso']  # hyperparameter of LASSO
    Bay_alpha = hyperp['Bayes']  # hyperparameter of Bayes
    theta = {'LSE': regress.LSE_Reg(), 'Rgl': regress.Regul_Reg(rgl_lambda), 'Las': regress.LASSO_Reg(lasso_t),
             'Rob': regress.Robust_Reg()}
    Bayes_theta = regress.Bay_Reg(Bay_alpha)
    Test_Pred = {}
    Train_Pred = {}
    for key in theta:
        Test_Pred[key] = nonBayes_Pred(theta[key], Fea_Tst)
        Train_Pred[key] = nonBayes_Pred(theta[key], Fea_Trn)
    Bayes_Test_Pred = Bayes_Pred(Fea_Tst, Bayes_theta)
    Bayes_Train_Pred = Bayes_Pred(Fea_Trn, Bayes_theta)
    Pred_Err = Pred_All_Err(polyy, Test_Pred, Bayes_Test_Pred)
    for key in theta:
        nonBayes_Plt(smpx, org_samy, Train_Pred[key], polyx, polyy, Test_Pred[key], key+': MSE='+str(Pred_Err[key]),key)

    # Bayes
    Bayes_Ply(smpx, org_samy, Bayes_Train_Pred, polyx, polyy, Bayes_Test_Pred,Pred_Err['Bayes'])

    Value_to_Re={'Bayes_theta':Bayes_theta,'nonBayes_theta':theta,'Error':Pred_Err}
    return Value_to_Re

