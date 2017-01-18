import numpy as np
import pylab as pl

from scikits.learn import svm
from scikits.learn.base import BaseEstimator, ClassifierMixin

###############################################################################
# Define kernels

class RBF(object):
    """docstring for RBF"""
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, Y=None):
        XX = np.sum(X * X, axis=1)[:,np.newaxis]
        if Y is None:
            Y = X
            YY = XX.T
        else:
            YY = np.sum(Y * Y, axis=1)[np.newaxis,:]
        distances = XX + YY # Using broadcasting
        distances -= 2 * np.dot(X, Y.T)
        distances = np.maximum(distances, 0)
        return np.exp(- self.gamma * distances)

def linear(X, Y=None):
    """Linear kernel"""
    if Y is None:
        Y = X
    return np.dot(X, Y.T)

class MultiKernel(object):
    def __init__(self, kernels, gammas, X=None):
        self.kernels = kernels
        self.gammas = gammas
        self.X = X
        self.Ks = None
        if X is not None: # Precompute kernels
            self.Ks = [kernel(X) for kernel in kernels]

    def __call__(self, X, Y=None):
        """Construct kernel by linear combination"""
        K = 0
        if X is self.X and (Y is X or Y is None):
            for gamma, Ki in zip(self.gammas, self.Ks):
                if gamma > 0.0:
                    K += gamma * Ki
        else:
            for gamma, kernel in zip(self.gammas, self.kernels):
                if gamma > 0.0:
                    K += gamma * kernel(X, Y)
        return K

        ###############################################################################
# Multi Kernel SVC (2 classes only)

class MultiKernelSVC(BaseEstimator, ClassifierMixin):
    """Lp - Multiple Kernel Learning (MKL)
    2 classes only
    Parameters
    ----------
    kernel : string, optional
        List of precomputed kernels.
    p : float, optional
        ???
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    Notes
    -----
    Add ref Szafranski 2010 Machine Learning Journal
    """
    def __init__(self, kernels, p=1, maxit=10, C=1, verbose=False, tol=1e-5,
                 store_objective=False):
        self.kernels = kernels
        self.p = p
        self.maxit = maxit
        self.C = C
        self.verbose = verbose
        self.tol = tol
        self.store_objective = store_objective

    def fit(self, X, y, **params):
        """fit the MKL and learn the kernel"""
        self._set_params(**params)

        X = np.atleast_2d(X)
        y = y.ravel()

        classes = np.unique(y)
        n_classes = classes.size
        assert n_classes == 2

        y = np.array(y, dtype=np.int)
        y[y==classes[0]] = -1
        y[y==classes[1]] = 1

        p = float(self.p)
        kernels = self.kernels
        C = self.C

        n_kernels = len(self.kernels)
        # kernel weights
        gammas = (1.0 / n_kernels) ** (1.0 / p) * np.ones(n_kernels)

        # Construct kernel by linear combination
        multi_kernel = MultiKernel(kernels, gammas, X)
        Ks = multi_kernel.Ks

        norms = np.empty(n_kernels)
        maxit = self.maxit

        objective = []

        for it in range(maxit):
            if self.verbose:
                print "Gammas : %s" % multi_kernel.gammas

            svc = svm.SVC(kernel=multi_kernel, C=C)
            svc.fit(X, y)
            dual_coef_ = svc.dual_coef_.ravel()
            support_ = np.array(svc.support_, dtype=np.int).ravel() - 1

            # Update kernel weights
            for i, (gamma, K) in enumerate(zip(multi_kernel.gammas, Ks)):
                norms[i] = (gamma**2) * np.dot(dual_coef_,
                                    np.dot(K[support_][:,support_], dual_coef_))

            if self.store_objective:
                dual_obj = -0.5 * np.dot(dual_coef_,
                              np.dot(multi_kernel(X[support_]), dual_coef_)) + \
                              (dual_coef_ * y[support_]).sum()
                objective.append(dual_obj)

            # print norms
            norms = norms ** (1.0 / (1.0 + p))
            scaling = np.sum(norms ** p ) ** (1.0 / p)
            gammas_ = norms / scaling

            gammas_[gammas_ < 1e-6 * gammas_.max()] = 0.0

            if (gammas_ - multi_kernel.gammas).max() < self.tol:
                if self.verbose:
                    print "Converged after %d interations" % it
                break

            multi_kernel.gammas = gammas_
        else:
            if self.verbose:
                print "Did NOT converge after %d interations" % it

        self._svc = svc
        self.gammas_ = multi_kernel.gammas
        self.objective = objective
        return self

    def predict(self, X):
        return self._svc.predict(X)


if __name__ == '__main__':
    xx, yy = np.meshgrid(np.linspace(-5, 5, 40), np.linspace(-5, 5, 40))
    np.random.seed(0)
    X = np.random.randn(300, 2)
    y = np.logical_xor(X[:,0]>0, X[:,1]>0)
    # y = X[:,0]>0

    y = np.array(y, dtype=np.int)
    y[y==0] = -1

    # Define the kernels
    kernels = [RBF(10 ** k) for k in range(-5, 0)] # some RBF kernels
    kernels.append(linear) # Add linear kernel

    # fit the model
    clf = MultiKernelSVC(kernels=kernels, C=1e6, verbose=True, maxit=100,
                         tol=1e-5, p=1, store_objective=True)
    clf.fit(X, y)

    objective = clf.objective

    # pl.close('all')
    # pl.figure()
    # pl.plot(objective)
    # pl.xlabel('Iterations')
    # pl.ylabel('Dual objective')

    # # plot the line, the points, and the nearest vectors to the plane
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # # pl.close('all')
    # pl.figure()
    # pl.set_cmap(pl.cm.Paired)
    # pl.pcolormesh(xx, yy, Z)
    # pl.scatter(X[:,0], X[:,1], c=y)

    # pl.axis('tight')
    # pl.show()
