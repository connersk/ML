import pdb
import random
import pylab as pl
from sklearn import linear_model
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from gradient_descent import polynomial_design_matrix as pdm
import numpy as np

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    X = X.reshape(len(X))
    return X, Y

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

def ridge(X,Y,lam):
    clf = linear_model.Ridge(alpha=lam, fit_intercept=False)
    clf.fit(X,Y)
    return clf

def compute_aic(err,n,phi,lam):
    eigs = np.linalg.eigvals(np.dot(phi.T,phi))
    df = np.sum([e/(e+lam) for e in eigs])
    return n*np.log(err)+2*df

def error(w,phi,Y):
    nparam = np.shape(phi)[1]
    w = w.reshape((nparam,1)) # make sure column vector
    return np.sum((Y - np.dot(phi, w).reshape((len(Y),1)))**2)

def dev():
    M = 3
    X, Y = regressAData()
    X = X.reshape(len(X))
    phiX = pdm(X,M)
    res = ridge(phiX,Y,1)
    plt.plot(X,Y,'o',label='A')
    X, Y = regressBData()
    plt.plot(X,Y,'o', label='B')
    Xv, Yv = validateData()
    plt.plot(Xv,Yv,'o', label='val')
    plt.legend(loc='best')
    plt.savefig('dev.png')
    print train_poly(X,Y,Xv,Yv,3,1)

def train_poly(X,Y,valX, valY,M,lam):
    phiX = pdm(X,M)
    phivalX = pdm(valX,M)
    res = ridge(phiX,Y,lam)
    e = error(res.coef_, phivalX, valY)
    aic = compute_aic(e,len(valY), phivalX, lam)
    return res, e, aic

def train_aic(X,Y,lam,M):
    Xv, Yv = validateData()
    aic = [[0]*len(M) for j in range(len(lam))]
    for ii, l in enumerate(lam):
        for jj, m in enumerate(M):
            res,e,a = train_poly(X,Y,Xv,Yv,m,l)
            #print 'lam={}'.format(l)
            #print 'M={}'.format(m)
            #print 'SSE = {:.3f}'.format(e)
            #print 'AIC = {:.3f}'.format(a)
            aic[ii][jj] = a
    return np.array(aic)

def plot_aic():
    XA, YA = regressAData()
    lam = [0,1e-5,1e-3,.1,.3,.5,1,5,10]
    M = [1,2,3,4,5,6,7,8]
    aic = train_aic(XA,YA,lam,M)
    print aic
    print aic.min()

    plt.figure(1)
    for i in range(len(M)):
        print 'M= {}: AIC_min = {}, lambda_AIC_min = {}'.format(M[i],aic[:,i].min(),lam[aic[:,i].argmin()])
        plt.plot(lam,aic[:,i], label='M={}'.format(M[i]))
    plt.legend(loc='best')
    plt.xlabel('$\lambda$')
    plt.ylabel('AIC')
    plt.title('Train A')
    plt.savefig('AIC_A.png')

    XB, YB = regressBData()
    aic = train_aic(XB,YB,lam,M)
    print aic
    print aic.min()
    plt.figure(2)
    for i in range(len(M)):
        print 'M= {}: AIC_min = {}, lambda_AIC_min = {}'.format(M[i],aic[:,i].min(),lam[aic[:,i].argmin()])
        plt.plot(lam,aic[:,i], label='M={}'.format(M[i]))
    plt.legend(loc='best')
    plt.xlabel('$\lambda$')
    plt.ylabel('AIC')
    plt.title('Train B')
    plt.savefig('AIC_B.png')

def test():
    plt.figure(3)
    plt.clf()

    XA, YA = regressAData()
    XB, YB = regressBData()
    Xv, Yv = validateData()
    plt.plot(XA,YA,'o', mfc='none', label='A')
    plt.plot(XB,YB,'o', mfc='none', label='B')
    plt.plot(Xv,Yv,'o', mfc='none', label='val')

    xx = np.linspace(-3,3)
    # train on A with M=2, lam=0
    M=2
    phiXA = pdm(XA,M)
    phiXB = pdm(XB,M)
    res = ridge(phiXA,YA,0)
    plt.plot(xx,res.predict(pdm(xx,M)), label='train A')
    e = error(res.coef_,phiXB,YB)
    print 'Train on A; Test on B; SSE = {:.3f}'.format(e)

    # train on B with M=1, lam=5
    M=1
    phiXA = pdm(XA,M)
    phiXB = pdm(XB,M)
    res = ridge(phiXB,YB,5)
    plt.plot(xx,res.predict(pdm(xx,M)), label='train B')
    e = error(res.coef_,phiXA,YA)
    print 'Train on B; Test on A; SSE = {:.3f}'.format(e)

    plt.legend(loc='best')
    plt.savefig('test.png')

if __name__ == '__main__':
    #plot_aic()
    test()
