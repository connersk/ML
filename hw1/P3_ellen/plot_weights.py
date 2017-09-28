import regressData
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from gradient_descent import polynomial_design_matrix as pdm
def plot_weights(X,Y,M, out_png,title):
    lams = [0,1,10]
    phiX = pdm(X,M)
    plt.figure(1,(3,3))
    plt.clf()
    width = .3
    xx = np.arange(M+1)
    for ii,l in enumerate(lams):
        res = regressData.ridge(phiX,Y,l)
        w = res.coef_.reshape(M+1)
        plt.bar(xx+ii*width,w,width, label='$\lambda={}$'.format(l))
    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('i')
    plt.ylabel('$w_i$')
    plt.tight_layout()
    plt.savefig(out_png)

def main():
    X, Y = regressData.regressAData()
    for M in (1,2,3,4,5,8):
        title = 'Train A; M={}'.format(M)
        out_png = 'weights_{}_trainA'.format(M)
        plot_weights(X,Y,M,out_png,title)

    X, Y = regressData.regressBData()
    for M in (1,2,3,4,5,8):
        title = 'Train B; M={}'.format(M)
        out_png = 'weights_{}_trainB'.format(M)
        plot_weights(X,Y,M,out_png,title)

if __name__ == '__main__':
    main()


