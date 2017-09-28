from loadFittingDataP2 import getData
import fit_poly
import fit_poly_gd as gd
import fit_poly_sgd as sgd
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

def plot_weights(M, out_png):
    X, Y = getData(False)
    w_gd, _ = gd.fit_polynomial(X,Y,M)
    w_sgd, _ = sgd.fit_polynomial(X,Y,M)
    w = fit_poly.fit_polynomial(X,Y,M)
    w = w.reshape(M+1)
    w_gd = w_gd.reshape(M+1)
    w_sgd = w_sgd.reshape(M+1)

    plt.figure(1,(3,3))
    plt.clf()
    width = .3
    xx = np.arange(M+1)
    plt.bar(xx,w,width, label='analytic')
    plt.bar(xx+width, w_gd, width, label='BGD')
    plt.bar(xx+2*width, w_sgd, width, label='SGD')
    plt.title('M={}'.format(M))
    plt.legend(loc='best')
    plt.xlabel('i')
    plt.ylabel('$w_i$')
    plt.tight_layout()
    plt.savefig(out_png)

def main():
    for M in (2,3,4,5,8):
        out_png = 'weights_{}'.format(M)
        plot_weights(M,out_png)

if __name__ == '__main__':
    main()


