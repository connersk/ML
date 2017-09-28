import lassoData
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import gradient_descent as gd

def design_matrix(x):
    ncols = 13
    ndata = len(x)
    x = x.reshape(ndata)
    phi = np.empty((ndata, ncols))
    phi[:,0] = x
    for i in range(1,13):
        phi[:,i] = np.sin(i*.4*np.pi*x)
    return phi

def lasso(X,Y, lam):
    if not lam:
        clf = linear_model.LinearRegression(fit_intercept=False)
    else:
        clf = linear_model.Lasso(alpha=lam, fit_intercept=False, max_iter=1e5)
    clf.fit(X,Y)
    return clf

def ridge(X,Y, lam):
    if not lam:
        clf = linear_model.LinearRegression(fit_intercept=False)
    else:
        clf = linear_model.Ridge(alpha=lam, fit_intercept=False)
    clf.fit(X,Y)
    return clf


def dev():
    X, Y = lassoData.lassoTrainData()
    phi = design_matrix(X)
    plt.plot(X,Y,'o',label='training')
    Xplot = design_matrix(np.linspace(-1,1,100))
    wtrue = np.loadtxt('lasso_true_w.txt')
    wtrue = wtrue.reshape((13,1))
    plt.plot(np.linspace(-1,1,100),np.dot(Xplot,wtrue), label='True model')
    for lam in (.001,.01,.5,1):
        res = lasso(phi,Y,lam)
        print res.coef_
        plt.plot(np.linspace(-1,1,100),res.predict(Xplot), label='LASSO; $\lambda = {}$'.format(lam))
    plt.xlim((-1,1))
    plt.legend(loc='best')
    plt.savefig('/tmp/test.png')

def train():
    lambdas = np.arange(.05,1.15,.2)
    lambdas = [0,.001,.01,.02,.03,.04]+list(lambdas)
    X, Y = lassoData.lassoTrainData()
    phi = design_matrix(X)
    plt.figure(1)
    plt.plot(X,Y,'o',label='Training')
    xx = np.linspace(-1,1,100)
    Xplot = design_matrix(xx)
    wtrue = np.loadtxt('lasso_true_w.txt')
    wtrue = wtrue.reshape((13,1))
    plt.plot(xx,np.dot(Xplot,wtrue), label='True model')
    results = []
    ii = 0
    for lam in lambdas:
        res = lasso(phi,Y,lam)
        results.append(res)
        print res.coef_
        if ii%2 == 0:
            plt.plot(xx,res.predict(Xplot), label='LASSO; $\lambda = {}$'.format(lam))
        ii += 1


    Xval, Yval = lassoData.lassoValData()
    plt.plot(Xval,Yval, 'o', label='Validation')
    phival = design_matrix(Xval)
    #err_train = []
    err_val = []
    reg_penalty = []
    aic = []
    for res in results:
        #e1 = np.sum((Y - np.dot(phi,res.coef_.reshape((13,1))).reshape((len(Y),1)))**2)
        e2 = np.sum((Yval - np.dot(phival, res.coef_.reshape((13,1))).reshape((len(Yval),1)))**2)
        #err_train.append(e1)
        err_val.append(e2)
        reg_penalty.append(np.sum(np.abs(res.coef_)))
        aic.append(compute_aic(res.coef_.reshape(13),e2,len(Yval)))

    #print err_train
    print 'lambdas'
    print lambdas
    print 'SSE'
    print err_val
    print 'regularization'
    print reg_penalty
    print 'AIC'
    print aic
    plt.xlim((-1,1))
    plt.legend(loc='best')
    plt.savefig('train_lasso.png')
    plt.figure(2)
    #plt.plot(lambdas, err_train, label='training error')
    plt.plot(np.log(lambdas), err_val, label='SSE')
    plt.plot(np.log(lambdas), reg_penalty, label='regularization penalty')
    plt.plot(np.log(lambdas), aic, label='AIC')
    plt.xlabel('$ln(\lambda)$')
    plt.legend(loc='best')
    plt.title('LASSO regularization')
    plt.tight_layout()
    plt.savefig('training_errors.png')

def compute_aic(w,err,n):
    k = len([x for x in w if x])
    return 2*k + n*np.log(err)


def train_ridge():
    lambdas = np.arange(.05,1.15,.2)
    #lambdas = [0,.001,.01]+list(lambdas)+[1,3,5]
    lambdas = [0,.001,.01,.02,.03,.04]+list(lambdas)
    X, Y = lassoData.lassoTrainData()
    phi = design_matrix(X)
    plt.figure(1)
    plt.clf()
    plt.plot(X,Y,'o',label='Training')
    xx = np.linspace(-1,1,100)
    Xplot = design_matrix(xx)
    wtrue = np.loadtxt('lasso_true_w.txt')
    wtrue = wtrue.reshape((13,1))
    plt.plot(xx,np.dot(Xplot,wtrue), label='True model')
    results = []
    ii = 0
    for lam in lambdas:
        res = ridge(phi,Y,lam)
        results.append(res)
        print res.coef_
        if ii%2 == 0:
            plt.plot(xx,res.predict(Xplot), label='ridge; $\lambda = {}$'.format(lam))
        ii += 1


    Xval, Yval = lassoData.lassoValData()
    plt.plot(Xval,Yval, 'o', label='Validation')
    phival = design_matrix(Xval)
    #err_train = []
    err_val = []
    reg_penalty = []
    aic = []
    for res in results:
        #e1 = np.sum((Y - np.dot(phi,res.coef_.reshape((13,1))).reshape((len(Y),1)))**2)
        e2 = np.sum((Yval - np.dot(phival, res.coef_.reshape((13,1))).reshape((len(Yval),1)))**2)
        #err_train.append(e1)
        err_val.append(e2)
        reg_penalty.append(np.sum(res.coef_**2))
        aic.append(compute_aic(res.coef_.reshape(13),e2,len(Yval)))

    #print err_train
    print 'lambdas'
    print lambdas
    print 'SSE'
    print err_val
    print 'regularization'
    print reg_penalty
    print 'AIC'
    print aic
    plt.xlim((-1,1))
    plt.legend(loc='best')
    plt.savefig('train_ridge.png')
    plt.figure(2)
    plt.clf()
    #plt.plot(lambdas, err_train, label='training error')
    plt.plot(np.log(lambdas), err_val, label='SSE')
    plt.plot(np.log(lambdas), reg_penalty, label='regularization penalty')
    plt.plot(np.log(lambdas), aic, label='AIC')
    plt.xlabel('$ln(\lambda)$')
    plt.title('Ridge regularization')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('ridge_training_errors.png')

def error(w,phi,Y):
    return np.sum((Y - np.dot(phi, w.reshape((13,1))).reshape((len(Y),1)))**2)

def test():
    Xtrain, Ytrain = lassoData.lassoTrainData()
    phi_train = design_matrix(Xtrain)
    print '{} data points for training'.format(len(Xtrain))
    Xval, Yval= lassoData.lassoValData()
    phi_val= design_matrix(Xval)
    print '{} data points for validation'.format(len(Xval))
    Xtest, Ytest = lassoData.lassoTestData()
    phi_test = design_matrix(Xtest)
    print '{} data points for testing'.format(len(Xtest))

    xx = np.linspace(-1,1,100)
    phi_plot = design_matrix(xx)


    plt.figure(3)
    plt.plot(Xtrain,Ytrain,'o',label='Training')
    plt.plot(Xval,Yval,'o',label='Validation')
    plt.plot(Xtest,Ytest,'o',label='Testing')

    wtrue = np.loadtxt('lasso_true_w.txt').reshape((13,1))
    plt.plot(xx,np.dot(phi_plot,wtrue), label='True model')

    lam = 0.01
    res_train = lasso(phi_train,Ytrain,lam)
    #res_val = lasso(phi_val,Yval,lam)
    #res_test = lasso(phi_test,Ytest,lam)

    plt.plot(xx,res_train.predict(phi_plot),label='LASSO')
    #plt.plot(xx,res_val.predict(phi_plot),label='LASSO validation')
    #plt.plot(xx,res_test.predict(phi_plot),label='LASSO testing')

    lam = 0.01
    res_ridge = ridge(phi_train, Ytrain, lam)
    plt.plot(xx,res_ridge.predict(phi_plot), label='Ridge')

    lam = 0
    res_noreg = lasso(phi_train, Ytrain, lam)
    plt.plot(xx,res_noreg.predict(phi_plot),label='$\lambda = 0$')

    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('all_fitting.png')

    print 'Training error with wtrue  : {:.3f}'.format(error(wtrue,phi_train,Ytrain))
    print 'Training error with w lasso: {:.3f}'.format(error(res_train.coef_,phi_train,Ytrain))
    print 'Training error with w ridge: {:.3f}'.format(error(res_ridge.coef_,phi_train,Ytrain))
    print 'Training error with w noreg: {:.3f}'.format(error(res_noreg.coef_,phi_train,Ytrain))
    print 'Validation error with wtrue  : {:.3f}'.format(error(wtrue,phi_val,Yval))
    print 'Validation error with w lasso: {:.3f}'.format(error(res_train.coef_,phi_val,Yval))
    print 'Validation error with w ridge: {:.3f}'.format(error(res_ridge.coef_,phi_val,Yval))
    print 'Validation error with w noreg: {:.3f}'.format(error(res_noreg.coef_,phi_val,Yval))
    print 'Testing error with wtrue  : {:.3f}'.format(error(wtrue,phi_test,Ytest))
    print 'Testing error with w lasso: {:.3f}'.format(error(res_train.coef_,phi_test,Ytest))
    print 'Testing error with w ridge: {:.3f}'.format(error(res_ridge.coef_,phi_test,Ytest))
    print 'Testing error with w noreg: {:.3f}'.format(error(res_noreg.coef_,phi_test,Ytest))
    plot_w(wtrue, 'w true', 'w_true.png')
    plot_w(res_train.coef_, 'LASSO w', 'w_lasso.png')
    plot_w(res_ridge.coef_, 'ridge w', 'w_ridge.png')
    plot_w(res_noreg.coef_, 'unregularized w', 'w_noreg.png')

def plot_w(weights, title, out_png):
    weights = weights.reshape(13)
    plt.figure(4,(3,3))
    plt.clf()
    plt.plot([-1,15],[0,0],color='k',lw=1)
    plt.bar(range(1,len(weights)+1), weights)
    plt.title(title)
    plt.xlabel('$i$')
    plt.ylabel('$w_i$')
    plt.tight_layout()
    plt.xlim((0,14))
    plt.ylim((-.5,6))
    plt.xticks(range(14))
    plt.savefig(out_png)


if __name__ == '__main__':
    #dev()
    train()
    train_ridge()
    test()

