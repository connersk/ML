import loadFittingDataP1
import loadParametersP1
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
#import pdb; pdb.set_trace()
import gradient_descent

def plot_gradient_descent(objective, gradient, obj_args,initial_guess,h,cc, out_png=None):
    '''Problem 1.1'''

    results = gradient_descent.run_gradient_descent(lambda x: objective(args[0],args[1],x),
    	lambda x: gradient(args[0],args[1],x), initial_guess, h, cc)
	
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


#Objective funtions and gradients

def gaussian_objective(u, cov, x):
    return -np.power(10,4) / (np.sqrt(np.power(2*np.pi,len(u))*np.linalg.det(cov))) *np.exp(-0.5 * np.dot(np.dot(np.subtract(x,u),np.linalg.inv(cov)), np.subtract(x,u)))
    
def gaussian_gradient(u, cov, x):
    return -1*gaussian_objective(u,cov,x)*np.linalg.inv(cov)*np.subtract(x,u)

def quadratic_objective(A,b,x):
    return 0.5 * np.subtract(np.dot(np.dot(x,A),x),np.dot(x,b))

def quadratic_gradient(A,b,x):
    return np.subtract(np.dot(A,x),b)



def main():
    X, Y = loadFittingDataP1.getData()
    Y = Y.reshape((100,1))
    w_opt = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), Y)
    gaussMean,gaussCov,quadBowlA,quadBowlb = loadParametersP1.getData()
    gaussMean = gaussMean.reshape((2,1))
    quadBowlb = quadBowlb.reshape((2,1))


if __name__ == '__main__':
    main()
