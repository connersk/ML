from loadFittingDataP2 import getData
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import gradient_descent as gd

def stochastic_gradient_descent(func, deriv, X, Y, weights0, tau, k, tol, maxloops=100000, return_error=False):
    ndata = np.shape(X)[0]
    nparams = np.shape(weights0)[0]
    t = 0
    while 1:
        order = range(ndata)
        np.random.shuffle(order)
        weights0_copy = weights0.copy()
        for i in order:
            xx = X[i,:].reshape((1,nparams))
            yy = Y[i].reshape((1,1))
            step = (tau+t)**(-k)
            d = deriv(weights0,xx,yy)
            weights1 = weights0-step*d
            weights0 = weights1
        fx1 = func(weights1,X,Y)
        fx0 = func(weights0_copy,X,Y)
        if abs(fx1-fx0) < tol:
            break
        t += 1
        if t > maxloops:
           raise RuntimeError, "Maxloops exceeded. Error is {}".format(abs(fx1-fx0))
    return weights1, t

def fit_polynomial(X, Y, M, out_png=None):
    '''Problem 2.1'''
    ndata = len(X)
    nparams = M+1
    Y = np.reshape(Y, (ndata,1))

    phi = gd.polynomial_design_matrix(X, M)
    w0 = np.zeros((nparams,1))
    tau = 100
    k = .75
    tol = 1e-6
    weights, niter = stochastic_gradient_descent(gd.least_squares_objective,
                                             gd.least_squares_gradient,
                                             phi, Y,
                                             w0, tau, k, tol)
    print 'Niterations: {}'.format(niter)
    weights_opt = gd.analytic_least_squares(phi, Y)
    if out_png:
        plt.figure(1, figsize=(4,4))
        plt.clf()
        plt.plot(X,np.array(Y),'o',color='blue', label='data')

        xp = np.linspace(0, 1, 100)
        y_model = np.cos(np.pi*xp) + np.cos(2*np.pi*xp)
        plt.plot(xp, y_model, color='orange', label='true model')

        y_regress = np.dot(gd.polynomial_design_matrix(xp,M), weights.reshape((nparams,1)))
        plt.plot(xp, y_regress, color='red', label='fitted model')
        SSE = gd.least_squares_objective(weights, gd.polynomial_design_matrix(X, M), Y.reshape((ndata,1)))

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.title('M = {}, SSE = {:.2f}'.format(M, SSE))
        plt.tight_layout()
        plt.savefig(out_png)
    return weights, niter

def main():
    X, Y = getData(False)
    ndata = len(X)
    for M in (0,1,2,3,4,6,8,10):
        #print 'M=%i' % M
        weights, _ = fit_polynomial(X,Y,M,'sgd_plots/regress_m_%i.png' % M)
        #print weights
        print 'SSE = {}'.format(gd.least_squares_objective(weights, gd.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))
        print 'SSE derivative = {}'.format(gd.least_squares_gradient(weights, gd.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))

        print 'M=%i & ' % M, 'w = ', [round(w,3) for w in weights[:,0]], '\\\\'
if __name__ == '__main__':
    main()
