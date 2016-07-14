import numpy as np
from scipy.optimize import check_grad

## logistic regression 

def getAvgGradient(w, X, y, L):
    N,D = X.shape

    # f = 1/N*sum_t log(1 + exp(-w'ytxt)) + .5*L*||w||^2
    l = 1.0/N* np.log(1. + np.exp(-y*np.dot(X,w))).sum() + .5*L*(w**2).sum() 
    # df = 1/N*sum_t -ytxt/(1 + exp(w'ytxt)) + L*w 
    g = -1.0/N*np.dot(y/(1. + np.exp(np.dot(X,w)*y)),X).reshape((D,)) + L*w

    return (g,l)


def predict(w, X):
    N,D = X.shape
    ypred = np.ones((N,), dtype=int)
    ypred[np.dot(X,w)<0.] = -1

    return ypred

    
def init(D):
    # Init w
    w = np.zeros((D,),dtype = np.double)
    return w

def loss(w, X, y, L):
    _,l = getAvgGradient(w, X, y, L)
    return l
    
def grad(w, X, y, L):
    g,_ = getAvgGradient(w, X, y, L)
    return g

def self_test1():
    D = 100
    N = 1000
    L = 1e-6
    w = init(D)
    w = np.random.normal(size=w.size)
    X = np.random.normal(size=(N,D))
    y = 2*np.random.randint(2,size=(N,))-1
    err = check_grad(loss, grad, w, X, y, L)
    print err
    

