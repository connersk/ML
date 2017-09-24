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

    x_res, d_res, f_res, iters = gradient_descent.run_gradient_descent(lambda x: objective(obj_args[0],obj_args[1],x),
    	lambda x: gradient(obj_args[0],obj_args[1],x), initial_guess, h, cc)

    f_out = np.concatenate(f_res).ravel().tolist()
    f_diff =  [f_out[i+1]-f_out[i] for i in range(len(f_out)-1)]

    print iters
    print len(f_diff)
    print len(f_out)


	
    if out_png:

        # plt.figure(1)
        # plt.clf()
        # plt.plot(range(0,iters+1),f_out[:-1],'o',color='blue')


        # Code for Plotting Objective function and difference in objective function
        fig, ax1 = plt.subplots()
        ax1.plot(range(0,iters+1),f_out[:-1], 'b-')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Objective function',color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(range(0,iters+1),f_diff, 'r.')
        ax2.set_ylabel('Difference in objective funciton between two steps', color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()


        # xp = np.linspace(0, 1, 100)
        # y_model = np.cos(np.pi*xp) + 1.5*np.cos(2*np.pi*xp)
        # plt.plot(xp, y_model, color='yellow')

        # #def polynomial(xx):
        # #    yy = 0
        # #    for ii in range(M+1):
        # #        w = weights.item((ii, 0))
        # #        yy += w*xx**ii
        # #    return yy

        # y_regress = np.dot(gradient_descent.polynomial_design_matrix(xp,M), weights.reshape((nparams,1)))
        # #poly = np.poly1d(z)
        # #y_regress = map(poly, xp)
        # plt.plot(xp, y_regress, color='red')

        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Linear Regression (M={})'.format(M))

        plt.savefig(out_png)


#Objective funtions and gradients

def gaussian_objective(u, cov, x):
	x = x.reshape((x.shape[0],1))
	result = ((-10**4)/(np.sqrt((2*np.pi)**2 * np.linalg.det(cov)))) * np.exp((-0.5)*np.dot(np.dot((x-u).T,np.linalg.inv(cov)), x-u))
	return result
    
def gaussian_gradient(u, cov, x):
	x = x.reshape((x.shape[0],1))
	gauss = np.asscalar(gaussian_objective(u,cov,x))
	return -1*np.dot(np.dot(gauss,np.linalg.inv(cov)),np.subtract(x,u))

def quadratic_objective(A,b,x):
	x = x.reshape((x.shape[0],1))
	return 0.5 * np.subtract(np.dot(x.T, np.dot(A,x)),np.dot(x.T,b))

def quadratic_gradient(A,b,x):
	x = x.reshape((x.shape[0],1))
	return np.subtract(np.dot(A,x),b)



def main():
    #X, Y = loadFittingDataP1.getData()
    #Y = Y.reshape((100,1))
    #w_opt = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), Y)
    gaussMean,gaussCov,quadBowlA,quadBowlb = loadParametersP1.getData()
    gaussMean = gaussMean.reshape((2,1))
    quadBowlb = quadBowlb.reshape((2,1))


    # plot_gradient_descent(gaussian_objective, gaussian_gradient, [gaussMean, gaussCov], 
    # 	np.array([5,5]), .01, .001, "test_plot.png")
    plot_gradient_descent(quadratic_objective, quadratic_gradient, [quadBowlA, quadBowlb], 
    	np.array([-10,-10]), .01, .001, "obj_and_diff.png")
    # print gaussian_objective(gaussMean, gaussCov,? np.array([5,5]))
    # print gaussian_gradient(gaussMean, gaussCov, np.array([5,5]))


if __name__ == '__main__':
    main()
