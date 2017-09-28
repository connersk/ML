import loadFittingDataP1
import loadParametersP1
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
#import pdb; pdb.set_trace()
import gradient_descent
from gradient_descent import least_squares_objective
from gradient_descent import least_squares_gradient
from gradient_descent import analytic_least_squares


def main():
    X, Y = loadFittingDataP1.getData()
    Y = Y.reshape((100,1))
    w_opt = analytic_least_squares(X,Y)


    #Problem 1.3.abc, running batch and stochastic gd on a variety of start points, saving num_iters, weights, calcing
    	#difference, plot both num iters and the difference
    #----------------------------------------------------------------------------------------
    start_points = [np.zeros((10,1)), np.zeros((10,1))+10, np.zeros((10,1))-10,
    				 np.zeros((10,1))+100, np.zeros((10,1))-100,
    				 np.zeros((10,1))+1000, np.zeros((10,1))-1000,
    				 20 * np.random.random_sample((10, 1)) - 10]
    batch_iterations = []
    batch_weights = []
    batch_diff = []
    batch_f = []
    stochastic_iterations = []
    stochastic_weights = []
    stochastic_diff = []
    stochastic_f = []

    
    fig, ax = plt.subplots()
    #labels = map(lambda x: str([round(i,2) for i in x]), points)


    start_points = [np.zeros((10,1))]

    for point in start_points:

    	#batch gd
    	w_batch, d_batch, f_batch, iters_batch = gradient_descent.run_gradient_descent(
    		func = lambda theta: least_squares_objective(theta, X, Y),
    		deriv = lambda theta: least_squares_gradient(theta, X, Y),
    		x0 = point,
    		h = 10.**(-6),
    		tol = 0.1
    		)
    	batch_weights.append(w_batch[-1])
    	batch_iterations.append(iters_batch*100) #since every batch iteration is 1 round of the whole dataset
    	batch_diff.append(np.linalg.norm(w_opt - w_batch[-1])/np.linalg.norm(w_opt))
        batch_f.append(f_batch)
        plt.plot(range(0,iters_batch+1), map(np.log, f_batch[:-1]),color="k")

    	#stochastic gd
    	w_sgd, iters_sgd, err_sgd, f_sgd  = gradient_descent.stochastic_gradient_descent(func=least_squares_objective, 
    		deriv = least_squares_gradient, X=X, Y=Y, weights0=point, tau=10.**8,k=.75,tol=.1, return_f=True)
    	stochastic_weights.append(w_sgd)
    	stochastic_iterations.append(iters_sgd)
    	stochastic_diff.append(np.linalg.norm(w_opt - w_sgd)/np.linalg.norm(w_opt))
        stochastic_f.append(stochastic_f)
        print iters_sgd/100
        print len(f_sgd)
        plt.plot(range(0,iters_sgd/100), map(np.log, f_sgd),color="b")


    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective function',color='k')
    plt.title("Least Squares Objective stuff")   
    #plt.legend(labels,shadow=True,fancybox=True)    
    plt.show()

    print "batch iterations"
    print batch_iterations
    print "batch diff"
    print batch_diff

    print "stochastic iterations"
    print stochastic_iterations
    print "stochastic_diff"
    print stochastic_diff

    #idk if this really needs a graph or not


    







if __name__ == '__main__':
    main()