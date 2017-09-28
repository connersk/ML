from loadFittingDataP2 import getData
import fit_poly_gd as gd
import fit_poly_sgd as sgd
import matplotlib.pyplot as plt
import pickle
import os

def main():
    if os.path.exists('iter_vs_M.pkl'):
        with open('iter_vs_M.pkl','rb') as f:
            MM = pickle.load(f)
            iter_gd = pickle.load(f)
            iter_sgd = pickle.load(f)
    else:
        MM = range(11)
        X, Y = getData(False)
        iter_gd = []
        iter_sgd = []
        for M in MM:
            _, n = gd.fit_polynomial(X,Y,M)
            iter_gd.append(n)
            _, n = sgd.fit_polynomial(X, Y, M)
            iter_sgd.append(n)
        with open('iter_vs_M.pkl','wb') as f:
            pickle.dump(MM,f)
            pickle.dump(iter_gd,f)
            pickle.dump(iter_sgd,f)

    plt.figure(1, (4.5,3.5))
    plt.plot(MM, iter_gd, 'o', label='BGD', mfc='none')
    plt.plot(MM, iter_sgd, 'o', label='SGD', mfc='none')
    plt.xlabel('M')
    plt.ylabel('Number of iterations')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('iter_vs_M.png')

if __name__ == '__main__':
    main()


