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
        plt.figure(1)
        plt.clf()
        plt.plot(X,np.array(Y),'o',color='blue')

        xp = np.linspace(0, 1, 100)
        y_model = np.cos(np.pi*xp) + 1.5*np.cos(2*np.pi*xp)
        plt.plot(xp, y_model, color='yellow')

        y_regress = np.dot(gradient_descent.polynomial_design_matrix(xp,M), weights.reshape((nparams,1)))
        plt.plot(xp, y_regress, color='red')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression (M={})'.format(M))

        plt.savefig(out_png)
    return weights

def main():
    X, Y = getData(False)
    ndata = len(X)
    for M in (0,1,3,10):
        print 'M=%i' % M
        weights = fit_polynomial(X,Y,M,'regress_m_%i.png' % M)
        print weights
        print 'SSE = {}'.format(gradient_descent.least_squares_objective(weights, gradient_descent.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))
        print 'SSE derivative = {}'.format(gradient_descent.least_squares_gradient(weights, gradient_descent.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))

if __name__ == '__main__':
    main()
