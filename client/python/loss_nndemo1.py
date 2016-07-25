import numpy as np
from scipy.optimize import check_grad

## Two-layer NN with ReLU
# Two-layer NN, with 200 units per layer with ReLu ai = max(0,oi)
# X - (W01) - Layer1 - (W12) - Layer2 - (W23) - Output
# ((D+1)*nh) + ((nh+1)*nh) + ((nh+1)*K)

nh = 200

def getAvgGradient(w, X, y, L, K):
    [N,D] = X.shape

    W01,b1,W12,b2,W23,b3 = parseParams(w,D,K)
    
    # Forward pass
    h1 = np.maximum(0, np.dot(X, W01) + np.tile(b1,(N,1))) # N x nh, ReLU
    h2 = np.maximum(0, np.dot(h1, W12) + np.tile(b2,(N,1))) # N x nh, ReLU
    scores = np.dot(h2, W23) + np.tile(b3,(N,1)) # N x K
    exp_scores = np.exp(scores-np.tile(np.max(scores,axis=1,keepdims=True),(1,K)))
    probs = exp_scores / np.tile(exp_scores.sum(axis=1,keepdims=True),(1,K)) # N x K
    l = -np.log(probs[range(N),y]).mean() + .5*L*((W01**2).sum()+(W12**2).sum()+(W23**2).sum())

    # Backward pass
    dscores = probs # N x K
    dscores[range(N),y] -= 1
    #dscores /= N

    dW23 = np.dot(h2.T, dscores)/N + L*W23 # nh x K
    db3 = np.sum(dscores, axis=0, keepdims=True)/N # nh x 1
    dh2 = np.dot(dscores, W23.T) # N x K x K x nh = N x nh
    dh2[h2 <= 0.] = 0.

    dW12 = np.dot(h1.T, dh2)/N + L*W12
    db2 = np.sum(dh2, axis=0, keepdims=True)/N
    dh1 = np.dot(dh2, W12.T)
    dh1[h1 <= 0.] = 0.

    dW01 = np.dot(X.T, dh1)/N + L*W01
    db1 = np.sum(dh1, axis=0, keepdims=True)/N

    g = np.concatenate((dW01.flatten(), db1.flatten(), dW12.flatten(), db2.flatten(), dW23.flatten(), db3.flatten()), axis=0)

    return (g, l)
    
    
    
def predict(w, X, K):

    N,D = X.shape
    W01,b1,W12,b2,W23,b3 = parseParams(w,D,K)
    
    # Forward pass
    h1 = np.maximum(0, np.dot(X, W01) + np.tile(b1,(N,1))) # N x nh, ReLU
    h2 = np.maximum(0, np.dot(h1, W12) + np.tile(b2,(N,1))) # N x nh, ReLU
    scores = np.dot(h2, W23) + np.tile(b3,(N,1)) # N x K
    #exp_scores = np.exp(scores-np.tile(np.max(scores,axis=1,keepdims=True),(1,K)))
    #probs = exp_scores / np.tile(exp_scores.sum(axis=1,keepdims=True),(1,K)) # N x K
    #ypred = np.argmax(probs,axis=1)
    ypred = np.argmax(scores,axis=1) 

    return ypred
    
    
def parseParams(w,D,K):

    cnt = 0
    W01 = w[:D*nh].reshape((D,nh))
    cnt += D*nh
    b1 = w[cnt:cnt+nh].reshape((1,nh))
    cnt += nh

    W12 = w[cnt:cnt+nh*nh].reshape((nh,nh))
    cnt += nh*nh
    b2 = w[cnt:cnt+nh].reshape((1,nh))
    cnt += nh

    W23 = w[cnt:cnt+nh*K].reshape((nh,K))
    cnt += nh*K
    b3 = w[cnt:cnt+K].reshape((1,K))
    cnt += K

    if (cnt != w.size):
        print 'Error: wrong param size'
        exit()
        
    return (W01,b1,W12,b2,W23,b3)


def init(D,K):

    d = (D+1)*nh + (nh+1)*nh + (nh+1)*K
    w = 1.e-1*np.random.normal(size=(d,))
    #w = np.zeros((d,))
    return w
    

def loss(w, X, y, L, K):
    _,l = getAvgGradient(w, X, y, L, K)
    return l
   
    
def grad(w, X, y, L, K):
    g,_ = getAvgGradient(w, X, y, L, K)
    return g


def self_test1():
    D = 100
    K = 2
    N = 10
    L = 1e-6

    # check parsing    
    W01 = np.random.randn(D,nh)
    b1 = np.random.randn(1,nh)
    W12 = np.random.randn(nh,nh)
    b2 = np.random.randn(1,nh)
    W23 = np.random.randn(nh,K)
    b3 = np.random.randn(1,K)

    w = np.concatenate((W01.flatten(), b1.flatten(), W12.flatten(), b2.flatten(), W23.flatten(), b3.flatten()), axis=0)
    W01_,b1_,W12_,b2_,W23_,b3_ = parseParams(w,D,K)
    print ((W01-W01_)**2).sum()/(W01**2).sum()
    print ((b1-b1_)**2).sum()/(b1**2).sum()
    print ((W12-W12_)**2).sum()/(W12**2).sum()
    print ((b2-b2_)**2).sum()/(b2**2).sum()
    print ((W23-W23_)**2).sum()/(W23**2).sum()
    print ((b3-b3_)**2).sum()/(b3**2).sum()

    w = init(D, K)
    w = 1e-0*np.random.normal(size=w.size)
    X = np.random.normal(size=(N,D))
    y = np.random.randint(K,size=(N,))
    err = check_grad(loss, grad, w, X, y, L, K)
    print err
     

