import numpy as np
from scipy.optimize import check_grad

## softmax
def getAvgGradient(w, X, y, L, K):
    N,D = X.shape
    W = w.reshape((K,D))
    
    XW = np.dot(X,W.T) # N x K
    XW -= np.tile(XW.max(axis=1).reshape((N,1)),(1,K))
    expXW = np.exp(XW) # N x K
    sumexpXW = expXW.sum(axis=1) # N x 1
    XWy = XW[range(N),y] # N x 1

    # f = -1/N*sum_t log(exp(w(yt)'xt)/sum_k exp(wk'xt)) + .5*l*||W||^2
    # = -1/N*sum_t [w(yt)'*xt - log(sum_k exp(wk'xt))] + .5*l*||W||^2
    # = -1/N*sum(sum(W(:,y).*X,1),2) + 1/N*sum(log(sumexpWX),2) + .5l*sum(sum(W.^2));
    l = -1.0/N*XWy.sum() + 1.0/N*np.log(sumexpXW).sum() +.5*L*(W**2).sum()#(axis=(0,1))

    # df/dwk = -1/N*sum(x(y==k,:),1) + 1/N*sum_t exp(wk'xt)*xt/(sum_k exp(wk'xt))] + L*wk
    G = np.zeros((K,D))
    for k in range(K):
        indk = np.where(y==k)[0]    
        G[k,:] = -1.0/N*X[indk,:].sum(axis=0).reshape((D,)) \
            + 1.0/N*np.dot(expXW[:,k]/sumexpXW,X).reshape((D,)) + L*W[k,:].reshape((D,))
    
    g = G.reshape((K*D,))
    return (g,l)


def predict(w, X, K):
    N,D = X.shape
    W = w.reshape((K,D))
    ypred = np.argmax(np.dot(X,W.T),axis=1)
    return ypred

    
def init(D,K):
    w = np.zeros((D*K,))
    return w

def loss(w, X, y, L, K):
    _,l = getAvgGradient(w, X, y, L, K)
    return l
    
def grad(w, X, y, L, K):
    g,_ = getAvgGradient(w, X, y, L, K)
    return g


def self_test1():
    D = 100
    K = 10
    N = 1000
    L = 1e-6
    w = init(D, K)
    w = np.random.normal(size=w.size)
    X = np.random.normal(size=(N,D))
    y = np.random.randint(K,size=(N,))
    err = check_grad(loss, grad, w, X, y, L, K)
    print err

