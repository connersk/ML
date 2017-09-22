from loadFittingDataP2 import getData
import numpy as np
import sys, os
import matplotlib.pyplot as plt

def fit_polynomial(X, Y, M, out_png=None):
    '''Problem 2.1'''
    # TODO: replace with our implementation
    #z = np.polyfit(X, Y, M)

    assert len(np.shape(X)) == 1
    A = np.empty((len(X),M+1))
    for i in range(M+1):
        A[:,i] = X**i
    A = np.matrix(A)
    Y = np.matrix(Y).T # Nx1 matrix
    weights = np.linalg.inv(A.T*A)*A.T*Y
    weights = np.reshape(np.array(weights),M+1) # back to np array

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

        y_regress = [polynomial(xx, weights) for xx in xp]
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

def compute_SSE(X, Y, M, weights):
    '''Problem 2.2'''
    SSE = 0
    deriv = 0
    for x,y in zip(X,Y):
        diff = y-polynomail(x,weights)
        SSE += diff**2
        deriv -= 2*x*diff
    return SSE, deriv

def central_difference(func, step, x):
    return (func(x+0.5*step) - func(x-0.5*step))/step

def main():
    X, Y = getData(False)
    for M in (0,1,3,10):
        weights = fit_polynomial(X,Y,M,'regress_m_%i.png' % M)

if __name__ == '__main__':
    main()
