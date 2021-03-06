from loadFittingDataP2 import getData
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import gradient_descent

def fit_polynomial(X, Y, M, out_png=None):
    '''Problem 2.1'''
    ndata = len(X)
    nparams = M+1
    Y = np.reshape(Y, (ndata,1))

    phi = gradient_descent.polynomial_design_matrix(X, M)
    weights = gradient_descent.analytic_least_squares(phi, Y)
    if out_png:
        plt.figure(1, figsize=(4,4))
        plt.clf()
        plt.plot(X,np.array(Y),'o',color='blue', label='data')

        xp = np.linspace(0, 1, 100)
        y_model = np.cos(np.pi*xp) + np.cos(2*np.pi*xp)
        plt.plot(xp, y_model, color='orange', label='true model')

        y_regress = np.dot(gradient_descent.polynomial_design_matrix(xp,M), weights.reshape((nparams,1)))
        plt.plot(xp, y_regress, color='red', label='fitted model')
        SSE = gradient_descent.least_squares_objective(weights, gradient_descent.polynomial_design_matrix(X, M), Y.reshape((ndata,1)))

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.title('M = {}, SSE = {:.2f}'.format(M, SSE))
        plt.tight_layout()
        plt.savefig(out_png)
    return weights

def main():
    X, Y = getData(False)
    ndata = len(X)
    for M in (0,1,2,3,4,6,8,10):
        #print 'M=%i' % M
        weights = fit_polynomial(X,Y,M,'regress_m_%i.png' % M)
        #print weights
        #print 'SSE = {}'.format(gradient_descent.least_squares_objective(weights, gradient_descent.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))
        #print 'SSE derivative = {}'.format(gradient_descent.least_squares_gradient(weights, gradient_descent.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))

        print 'M=%i & ' % M, 'w = ', [round(w,3) for w in weights[:,0]], '\\\\'
if __name__ == '__main__':
    main()
