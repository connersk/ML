from loadFittingDataP2 import getData
from regressData import regressAData, regressBData, validateData
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly
plotly.tools.set_credentials_file(username='ck9793', api_key='CU2pY2GAbchJm0Md9u3R')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
#import pdb; pdb.set_trace()
import gradient_descent


def ridge_weights(X, Y,M, lamb):
    return np.dot(np.dot(np.linalg.inv(np.add(lamb*np.identity(M+1),np.dot(X.T,X))),X.T),Y)

def polynomial(x,weights):
    assert len(np.shape(weights)) == 1
    yy = [w*x**ii for ii, w in enumerate(weights)]
    return np.sum(yy)

def R_squared(weights, X, Y,M):

	A = np.empty((len(X),M+1))
	for i in range(M+1):
		A[:,i] = X**i
	TSS = np.sum((Y - np.mean(Y))**2)
	RSS = gradient_descent.least_squares_objective(weights, A, Y)
	print TSS, RSS
	return (TSS - RSS)/TSS

def fit_polynomial_ridge(X, Y, M, lamb, out_png=None, return_weights=False):

    assert len(np.shape(X)) == 1
    A = np.empty((len(X),M+1))
    for i in range(M+1):
        A[:,i] = X**i
    #A = np.matrix(A)
    #Y = np.matrix(Y).T # Nx1 matrix
    weights = ridge_weights(A,Y,M,lamb)
    #weights = np.reshape(np.array(weights),M+1) # back to np array


    if out_png:
        plt.figure(1)
        plt.clf()
        plt.plot(X,np.array(Y),'o',color='blue')

        xp = np.linspace(0, 1, 100)
        y_model = np.cos(np.pi*xp) + 1.5*np.cos(2*np.pi*xp)
        plt.plot(xp, y_model, color='yellow')

        y_regress = [polynomial(xx, weights) for xx in xp]
        plt.plot(xp, y_regress, color='red')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Ridge Regression (M={})'.format(M))

        plt.savefig(out_png)

    if return_weights:
    	return weights

def main():

	X,Y = getData(False)


	#Making plots for 3.1
	# for M in (0,1,3,10):
	# 	for lamb in [10**ii for ii in range(-1,2)]:
	# 	    weights = fit_polynomial_ridge(X,Y,M,lamb,'ridge_m_%i_lambda_%s.png' % (M,lamb))

	#3.2

	Xa, Ya = regressAData()
	Xa = Xa.reshape((13))
	Ya = Ya.reshape((13))
	Xb, Yb = regressBData()
	Xb = Xb.reshape((10))
	Yb = Yb.reshape((10))
	Xval, Yval = validateData()
	Xval = Xval.reshape((22))
	Yval = Yval.reshape((22))

	m_lamba = []
	train_a = []
	test_a_on_b = []
	test_a_val = []
	train_b = []
	test_b_on_a = []
	test_b_val = []
	data_matrix = []
	data_matrix.append(['M, Lambda','Train A',"Train A/Test B","Train A/Validate","Train B","Train B/Test A","Train B/Validate"])
	for M in (3,5,50):
		for lamb in [10**ii for ii in range(1,3)]:

			m_lamba.append([M,lamb])

			weights_A = fit_polynomial_ridge(X=Xa,Y=Ya,M=M,lamb=lamb,return_weights=True)
			train_a.append(round(R_squared(weights_A, Xa, Ya,M),2))
			test_a_on_b.append(round(R_squared(weights_A, Xb, Yb,M),2))
			test_a_val.append(round(R_squared(weights_A, Xval, Yval,M),2))

			weights_B = fit_polynomial_ridge(X=Xb,Y=Yb,M=M,lamb=lamb,return_weights=True)
			train_b.append(round(R_squared(weights_B, Xb, Yb,M),2))
			test_b_on_a.append(round(R_squared(weights_B, Xa, Ya,M),2))
			test_b_val.append(round(R_squared(weights_B, Xval, Yval,M),2))

			data_matrix.append([str([M,lamb]),round(R_squared(weights_A, Xa, Ya,M),2),round(R_squared(weights_A, Xb, Yb,M),2),round(R_squared(weights_A, Xval, Yval,M),2),
			 round(R_squared(weights_B, Xb, Yb,M),2),round(R_squared(weights_B, Xa, Ya,M),2),round(R_squared(weights_B, Xval, Yval,M),2)])

	print data_matrix
	table = ff.create_table(data_matrix)
	py.iplot(table, filename='ridge_validation.png')	

	# print m_lamba
	# # print train_a
	# # print test_a_on_b
	# print test_a_val
	# # print train_b
	# # print test_b_on_a
	# print test_b_val





if __name__ == '__main__':
    main()