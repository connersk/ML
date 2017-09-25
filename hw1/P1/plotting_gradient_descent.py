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

def plot_gradient_descent(objective, gradient, obj_args,initial_guess,h,cc, out_png=None, return_vals=False,return_x=False):
    '''Problem 1.1'''

    x_res, d_res, f_res, iters = gradient_descent.run_gradient_descent(lambda x: objective(obj_args[0],obj_args[1],x),
    	lambda x: gradient(obj_args[0],obj_args[1],x), initial_guess, h, cc)

    f_out = np.concatenate(f_res).ravel().tolist()
    f_diff =  [f_out[i+1]-f_out[i] for i in range(len(f_out)-1)]

    print x_res[-1], f_res[-1]

    if return_vals:
    	return iters, f_out, f_diff

    if return_x:
    	return x_res

	
    if out_png:

    	#code for making a 3D plot of the objective function

    	# fig = plt.figure()
    	# ax = fig.add_subplot(111, projection='3d')
    	# X = np.arange(-10, 10, 0.25)
    	# Y = np.arange(-10, 10, 0.25)
    	# calcXY = np.vstack((X,Y))
    	# Z = []
    	# for i in range(0,len(X)):
    	# 	Z.append(objective(obj_args[0],obj_args[1], np.array([X[i],Y[i]])))
    	# Z = np.concatenate(Z).ravel().tolist()
    	# X, Y = np.meshgrid(X, Y)

    	# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
     #                   linewidth=0, antialiased=False)
    	# ax.zaxis.set_major_locator(LinearLocator(10))
    	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    	# fig.colorbar(surf, shrink=0.5, aspect=5)
    	


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

        plt.show()

        #plt.savefig(out_png)


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
	return np.subtract(0.5*np.dot(x.T, np.dot(A,x)),np.dot(x.T,b))

def quadratic_gradient(A,b,x):
	x = x.reshape((x.shape[0],1))
	return np.subtract(np.dot(A,x),b)

# For plotting effect of starting guess
def start_guess(objective, gradient, obj_params, points, cols,h,cc,func_type, out_png):
	#Note: points and cols must be of the same length

	# run gradient descent at all of the points
    iterations = []
    obj_vals = []
    obj_diff = []
    for point in points:
    	iters, f_out, f_diff = plot_gradient_descent(objective, gradient, obj_params, np.array(point), 
    		h, cc, return_vals=True)
    	iterations.append(iters)
    	obj_vals.append(f_out)
    	obj_diff.append(f_diff)

    #and then plot
    fig, ax = plt.subplots()
    labels = map(lambda x: str([round(i,2) for i in x]), points)
    for i in range(0, len(iterations)):
    	plt.plot(range(0,iterations[i] + 1), obj_vals[i][:-1], cols[i],label=labels[i])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective function',color='k')
    plt.title(func_type + " : Effect of starting guess")   
    plt.legend(labels,shadow=True,fancybox=True)	
    plt.savefig(out_png)


# For plotting effect of step size
def step_size_effect(objective, gradient, obj_params, steps, cols,start_guess,cc,func_type, out_png):
	#Note: points and cols must be of the same length

	# run gradient descent at all of the points
    iterations = []
    obj_vals = []
    obj_diff = []
    for step in steps:
    	iters, f_out, f_diff = plot_gradient_descent(objective, gradient, obj_params, np.array(start_guess), 
    		step, cc, return_vals=True)
    	iterations.append(iters)
    	obj_vals.append(f_out)
    	obj_diff.append(f_diff)

    #and then plot
    fig, ax = plt.subplots()
    #labels = map(lambda x: str([round(i,2) for i in x]), points)
    labels = map(str, steps)
    for i in range(0, len(iterations)):
    	plt.plot(range(0,iterations[i] + 1), obj_vals[i][:-1], cols[i],label=labels[i])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective function',color='k')
    plt.title(func_type + " : Effect of step size")   
    plt.legend(labels,shadow=True,fancybox=True)	
    plt.savefig(out_png)

# For plotting effect of covergence criteria
def covergence_criteria_effect(objective, gradient, obj_params, ccs, cols,start_guess,step,func_type, out_png):
	#Note: points and cols must be of the same length

	# run gradient descent at all of the points
    iterations = []
    obj_vals = []
    obj_diff = []
    for cc in ccs:
    	iters, f_out, f_diff = plot_gradient_descent(objective, gradient, obj_params, np.array(start_guess), 
    		step, cc, return_vals=True)
    	iterations.append(iters)
    	obj_vals.append(f_out)
    	obj_diff.append(f_diff)

    #and then plot
    fig, ax = plt.subplots()
    labels = map(str, ccs)
    for i in range(0, len(iterations)):
    	plt.plot(range(0,iterations[i] + 1), obj_vals[i][:-1], cols[i],label=labels[i])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective function',color='k')
    plt.title(func_type + " : Effect of covergence criteria")   
    plt.legend(labels,shadow=True,fancybox=True)	
    plt.savefig(out_png)

# For plotting norm of gradient
def plot_norm_of_gradient(objective, gradient, obj_args,start_guess,step,cc,func_type, out_png):

	x_res, d_res, f_res, iters = gradient_descent.run_gradient_descent(lambda x: objective(obj_args[0],obj_args[1],x),
    	lambda x: gradient(obj_args[0],obj_args[1],x), start_guess, step, cc)
	gradient_norm = map(lambda x: np.linalg.norm(gradient(obj_args[0],obj_args[1],x)), x_res)
	
	fig, ax = plt.subplots()
	plt.plot(range(0,iters + 1), gradient_norm[:-1])# cols[i],label=labels[i])
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Norm of Gradient',color='k')
	plt.title(func_type + " : Norm of Gradient")   
	#plt.legend(labels,shadow=True,fancybox=True)	
	plt.savefig(out_png)


# For plotting norm of difference between approx and exact gradients over the gradient descent process
def plot_finite_diff_vs_gradient(objective, gradient, obj_args, finite_steps,cols, start_guess,step,cc,func_type, out_png):

	x_res, d_res, f_res, iters = gradient_descent.run_gradient_descent(lambda x: objective(obj_args[0],obj_args[1],x),
    	lambda x: gradient(obj_args[0],obj_args[1],x), start_guess, step, cc)

	#the exact gradient
	gradient_vals = map(lambda x: gradient(obj_args[0],obj_args[1],x), x_res)

	# calcualting approx gradiet for different finite difference step sizes
	exact_approx_differences = []
	for finite_step in finite_steps:
		finite_diffs = []
		for i in range(len(x_res)):
			if i == 0:
				val = x_res[i]
			else:
				val = [np.asscalar(x_res[i][0]),np.asscalar(x_res[i][1])]
			approx_grad = gradient_descent.central_difference(lambda x: objective(obj_args[0],obj_args[1],x),
		 		finite_step, val).reshape((2,1))
		 	finite_diffs.append(approx_grad)
		#calculating norm of difference between approximate and exact gradient calculations
		exact_approx_differences.append(map(lambda x,y: np.linalg.norm(x-y), gradient_vals,finite_diffs))

	#plotting
	fig, ax = plt.subplots()
	labels = map(str, finite_steps)
	for i in range(0, len(exact_approx_differences)):
		plt.plot(range(0,iters + 1), exact_approx_differences[i][:-1], cols[i],label=labels[i])
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Norm of exact gradient - approximate gradient',color='k')
	plt.title("Quatratic: Gradient differences")   
	plt.legend(labels,shadow=True,fancybox=True)	
	plt.savefig(out_png)
		

	

def main():
    #X, Y = loadFittingDataP1.getData()
    #Y = Y.reshape((100,1))
    #w_opt = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), Y)
    gaussMean,gaussCov,quadBowlA,quadBowlb = loadParametersP1.getData()
    gaussMean = gaussMean.reshape((2,1))
    quadBowlb = quadBowlb.reshape((2,1))

    
    #####################################################################################
    # Effect of start guess
    #####################################################################################

    # Quadratic
    start_guess(objective=quadratic_objective, gradient=quadratic_gradient, obj_params=[quadBowlA,quadBowlb],
    	points=[[(80/3.)-10.,(80/3.)-10.],[(80/3.)-10.,(80/3.)+10.],[(80/3.),(80/3.)-20.],[(80/3.)-20.,(80/3.)-20.],[30.,30.]],
    	cols = ['b^','ko','r-','cv','g>'], h=10**(-2),cc=10**(1),func_type="Quadratic",out_png="quadratic_starting.png")

    #      #Notes, blue 16,15 converges faster than black 16,26 even tho it starts off worse, which makes sense due to
    # 		#co-dependence
    # 	#otherwise, close things pretty much

    # Gaussian
    start_guess(objective=gaussian_objective, gradient=gaussian_gradient, obj_params=[gaussMean,gaussCov],
    	points=[[5.,10.],[5.,15.],[15.,15.],[2.5,10.],[12.5,12.5]],
    	cols = ['b^','ko','r-','cv','g>'], h=10**(-2),cc=10**(-2),func_type="Gaussian",out_png="gaussian_starting.png")

    #####################################################################################
    # Effect of step size
    #####################################################################################

    # Quadratic
    step_size_effect(objective=quadratic_objective, gradient=quadratic_gradient, obj_params=[quadBowlA,quadBowlb],
    	steps=[0.001,0.005,0.01,0.05,0.1],
    	cols = ['b^','ko','r-','cv','g>'], start_guess=[(80/3.)-10,(80/3)-10],
    	cc=10**(1),func_type="Quadratic",out_png="quadratic_step_size.png")

    # Gaussian
    step_size_effect(objective=gaussian_objective, gradient=gaussian_gradient, obj_params=[gaussMean,gaussCov],
		steps=[0.001,0.005,0.01,0.05,0.1],
		cols = ['b^','ko','r-','cv','g>'], start_guess=[12,12],
		cc=10**(-2),func_type="Gaussian",out_png="guassian_step_size.png")

	#####################################################################################
    # Effect of Convergence Criteria
    #####################################################################################

    #Not so sure that I actually want to go with these plots, if I have time, convert to num iterations
    	# vs distance from the global minimum and then label the points in the plot

    # Quadratic
    covergence_criteria_effect(objective=quadratic_objective, gradient=quadratic_gradient, 
    	obj_params=[quadBowlA,quadBowlb],ccs=[0.001,0.01,0.1,1],
    	cols = ['b^','ko','r-','cv','g>'], start_guess=[(80/3.)-10,(80/3)-10],
    	step=10**(-2),func_type="Quadratic",out_png="quadratic_convergence_criteria.png")

    # Gaussian
    covergence_criteria_effect(objective=gaussian_objective, gradient=gaussian_gradient, 
    	obj_params=[gaussMean,gaussCov],ccs=[0.001,0.01,0.1,1],
    	cols = ['b^','ko','r-','cv','g>'], start_guess=[12,12],
    	step=10**(-2),func_type="Gaussian",out_png="gaussian_convergence_criteria.png")

   	#####################################################################################
    # Evolution of the norm of the gradient
    #####################################################################################

    plot_norm_of_gradient(objective=quadratic_objective, gradient=quadratic_gradient, 
    	obj_args= [quadBowlA,quadBowlb],start_guess=np.array([(80/3.)-10,(80/3)-10]), step=10**(-2),
    	cc=10**1, func_type="Quadratic", out_png="quadratic_gradient_norm.png")

    plot_norm_of_gradient(objective=gaussian_objective, gradient=gaussian_gradient, 
    	obj_args= [gaussMean,gaussCov],start_guess=np.array([12,12]), step=10**(-2),
    	cc=10**(-2), func_type="Gaussian", out_png="gaussian_gradient_norm.png")


   	#####################################################################################
    # 1.2 Central Difference Calculations
    #####################################################################################

    plot_finite_diff_vs_gradient(objective=quadratic_objective, gradient=quadratic_gradient, 
    	obj_args= [quadBowlA,quadBowlb], finite_steps = [0.1,1,10,1000,100000], cols = ['b-','k-','r-','c-','g-'],
    	start_guess=np.array([(80/3.)-100,(80/3)-100]), step=10**(-2),
    	cc=10**1, func_type="Quadratic", out_png="quadratic_finite_difference.png")
    #notes: a sweet spot occurs due to the addition and division terms
    	#but still really low in all cases, on the scale of 10^-10

    plot_finite_diff_vs_gradient(objective=gaussian_objective, gradient=gaussian_gradient, 
		obj_args= [gaussMean, gaussCov], finite_steps = [1,2.5,5,10,100], cols = ['b-','k-','r-','c-','g-'],
		start_guess=np.array([12,12]), step=10**(-2),
		cc=10**(-2), func_type="Gaussian", out_png="gaussian_finite_difference.png")
	#notes: scales more so with the values now but way higher error values




    

if __name__ == '__main__':
    main()
