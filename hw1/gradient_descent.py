import numpy as np

def run_gradient_descent(func, deriv, x0, h, tol, return_error=False):
    x = []
    d = []
    f = []
    err = []
    iterations = 0
    while 1:
        dx0 = deriv(x0)
        x.append(x0)
        d.append(dx0)
        x0 = x0.reshape((x0.shape[0],1))
        dx0 = dx0.reshape((dx0.shape[0],1))
        x1 = np.subtract(x0, h*dx0)
        fx1 = func(x1)
        fx0 = func(x0)
        f.append(fx0)
        err.append(abs(fx1-fx0))
        if np.all(abs(fx1-fx0) < tol):
            x.append(x1)
            f.append(fx1)
            break
        x0 = x1
        iterations += 1
    if return_error:
        x, d, f, iterations, err
    else:
        return x, d, f, iterations


def central_difference(func, step,x):
    central_difference = np.zeros(len(x))
    for i in range(len(x)):
        x_diff = np.zeros(len(x))
        x_diff[i] = 0.5*step
        central_difference[i] = (func(x+x_diff) - func(x-x_diff))/step
    return central_difference

def least_squares_objective(weights, X, Y):
    return np.sum((np.dot(X,weights) - Y)**2)

def least_squares_gradient(weights, X, Y):
    return (2*np.dot(X.T,np.dot(X,weights)-Y))

def analytic_least_squares(X,Y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), Y)

def stochastic_gradient_descent(func, deriv, X, Y, weights0, tau, k, tol, maxloops=100000, return_error=False, return_f=False):
    ndata = np.shape(X)[0]
    nparams = np.shape(weights0)[0]
    t = 0
    err = []
    f = []
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
            t += 1
            if t > maxloops:
                raise RuntimeError, "Maxloops exceeded. Last 10 values of error: {}".format(err[-10:])
        fx1 = func(weights1,X,Y)
        fx0 = func(weights0_copy,X,Y)
        f.append(fx0)
        err.append(abs(fx1-fx0))
        if abs(fx1-fx0) < tol:
            break
    if return_f:
        return weights1, t, err, f
    if return_error:
        return weights1, t, err
    else:
        return weights1, t

# def mini_batch_gradient_descent(func, deriv, X, Y, weights0, tau, k, tol, maxloops=100000):
#     ndata = np.shape(X)[0]
#     nparams = np.shape(weights0)[0]
#     t = 0
#     err = []
#     while 1:
#         order = range(ndata)
#         np.random.shuffle(order)
#         weights0_copy = weights0.copy()
#         for i in order:
#             xx = X[i,:].reshape((1,nparams))
#             yy = Y[i].reshape((1,1))
#             step = (tau+t)**(-k)
#             d = deriv(weights0,xx,yy)
#             weights1 = weights0-step*d
#             weights0 = weights1
#             t += 1
#             if t > maxloops:
#                 raise RuntimeError, "Maxloops exceeded. Last 10 values of error: {}".format(err[-10:])
#         fx1 = func(weights1,X,Y)
#         fx0 = func(weights0_copy,X,Y)
#         err.append(abs(fx1-fx0))
#         if abs(fx1-fx0) < tol:
#             break
#     return weights1, t




def polynomial_design_matrix(x, M):
    '''
    Create the design matrix for a polynomial basis

    Inputs:
        x (np.array): np array of input data
        M (int): order of the polynomial

    Output:
        phi (np.array with dims (ndata,M+1)): design matrix for GLS
    '''
    assert len(np.shape(x)) == 1, "Data must be 1 dimensional"
    ncols = M +1
    ndata = len(x)
    phi = np.empty((ndata,ncols))
    for i in range(ncols):
        phi[:,i] = x**i
    return phi


