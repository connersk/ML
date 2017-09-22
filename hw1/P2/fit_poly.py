from loadFittingDataP2 import getData
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import pdb; pdb.set_trace()
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

        #def polynomial(xx):
        #    yy = 0
        #    for ii in range(M+1):
        #        w = weights.item((ii, 0))
        #        yy += w*xx**ii
        #    return yy

        y_regress = np.dot(gradient_descent.polynomial_design_matrix(xp,M), weights.reshape((nparams,1)))
        #poly = np.poly1d(z)
        #y_regress = map(poly, xp)
        plt.plot(xp, y_regress, color='red')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression (M={})'.format(M))

        plt.savefig(out_png)

def polynomial(x,weights):
    assert len(np.shape(weights)) == 1
    yy = [w*x**ii for ii, w in enumerate(weights)]
    return np.sum(yy)

def main():
    X, Y = getData(False)
    for M in (0,1,3,10):
        weights = fit_polynomial(X,Y,M,'regress_m_%i.png' % M)

if __name__ == '__main__':
    main()
