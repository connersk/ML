from loadFittingDataP2 import getData
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import gradient_descent as gd

def run_gradient_descent(func, deriv, x0, h, tol, return_error=False):
    iterations = 0
    while 1:
        dx0 = deriv(x0)
        x1 = x0 - h*dx0
        fx1 = func(x1)
        fx0 = func(x0)
        if np.all(abs(fx1-fx0) < tol):
            break
        x0 = x1
        iterations += 1
    return x1, iterations


def fit_polynomial(X, Y, M, out_png=None, h=1e-2, tol=1e-6):
    '''Problem 2.1'''
    ndata = len(X)
    nparams = M+1
    Y = np.reshape(Y, (ndata,1))

    phi = gd.polynomial_design_matrix(X, M)
    w0 = np.zeros((nparams,1))
    weights, niter = run_gradient_descent(lambda w: gd.least_squares_objective(w, phi, Y),
                                         lambda w: gd.least_squares_gradient(w, phi, Y),
                                         w0, h, tol)
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
        weights, _ = fit_polynomial(X,Y,M,'gd_plots/regress_m_%i.png' % M)
        #print weights
        print 'SSE = {}'.format(gd.least_squares_objective(weights, gd.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))
        print 'SSE derivative = {}'.format(gd.least_squares_gradient(weights, gd.polynomial_design_matrix(X, M), Y.reshape((ndata,1))))

        print 'M=%i & ' % M, 'w = ', [round(w,3) for w in weights[:,0]], '\\\\'
if __name__ == '__main__':
    main()
