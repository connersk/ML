from loadFittingDataP2 import getData
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import gradient_descent

def cosine_design_matrix(x, M):
    '''
    Create the design matrix for a cosine basis

    Inputs:
        x (np.array): np array of input data
        M (int): order of the cosine series

    Output:
        phi (np.array with dims (ndata,M+1)): design matrix for GLS
    '''
    assert len(np.shape(x)) == 1, "Data must be 1 dimensional"
    ndata = len(x)
    phi = np.empty((ndata,M))
    for i in range(1,M+1):
        phi[:,i-1] = np.cos(i*np.pi*x)
    return phi


def fit_cosines(X, Y, M, out_png=None):
    '''Problem 2.4'''
    ndata = len(X)
    nparams = M
    Y = np.reshape(Y, (ndata,1))

    phi = cosine_design_matrix(X, M)
    weights = gradient_descent.analytic_least_squares(phi, Y)
    if out_png:
        plt.figure(1)
        plt.clf()
        plt.plot(X,np.array(Y),'o',color='blue')

        xp = np.linspace(0, 1, 100)
        y_model = np.cos(np.pi*xp) + 1.5*np.cos(2*np.pi*xp)
        plt.plot(xp, y_model, color='yellow')

        y_regress = np.dot(cosine_design_matrix(xp,M), weights.reshape((nparams,1)))
        plt.plot(xp, y_regress, color='red')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('GLS with cosine basis (M={})'.format(M))

        plt.savefig(out_png)
    return weights

def main():
    X, Y = getData(False)
    for M in (1,2,3,8):
        print 'M=%i' % M
        weights = fit_cosines(X,Y,M,'regress_cos_m_%i.png' % M)
        print weights
        print 'SSE = {}'.format(gradient_descent.least_squares_objective(weights, cosine_design_matrix(X, M), Y))
        print 'SSE derivative = {}'.format(gradient_descent.least_squares_gradient(weights, cosine_design_matrix(X, M), Y))


if __name__ == '__main__':
    main()
