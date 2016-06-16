from firebase import firebase
from firebase_token_generator import create_token

import numpy as np
#import numpy.random
import time


### Change these
url = <Your firebase url>
uid = <arbitrary string>
secret = <Copy and paste the string from firebase db>
maxiter = 100

# For testing
dataDir = './DataFiles' # Where you put your files
testFeatures = 'MNISTTestImages.dat';
testLabels = 'MNISTTestLabels.dat';
Ntest = 1000
Dtest = 784
Ktest = 10

# For local training
naught = 1.
localTraining = False # Set this true for faster, local training.

params = {}
if localTraining: # local version for testing purposes.
    params = {}
    params['D'] = 784
    params['K'] = 10
    params['L']= 1e-6
    params['N'] = 10000
    params['clientBatchSize'] = 100
    params['featureSource'] = 'MNISTTrainImages.dat'
    params['labelSource'] = 'MNISTTrainLabels.dat'
    params['lossFunction'] = 'Softmax'
    params['noiseDistribution'] = 'NoNoise'
    params['noiseScale'] = 0.

###


'''
# w is the paramter vector (, which is from the K x D matrix W for softmax)
# X is the N x D array of N samples of D-dimensional features
# y is the N x 1 array of N samples
# The output of a loss function is the averaged gradient over N samples, and the loss value
'''

#Hinge loss
def computeGradLoss_hinge(w, X, y, L):
    N,D = X.shape
    ind = np.where(y*np.dot(X,w) < 1.)[0]
    if len(ind)==0:
        l = .5*L*(w**2).sum()
        g = L*w
    else:
        l = -1.0/len(ind)*(1.-y[ind]*np.dot(X[ind,:],w)).sum() + .5*L*(w**2).sum()
        g = -1.0/len(ind)*np.dot(y[ind],X[ind,:]).reshape((D,)) + L*w
   
    return (g,l)


#logistic regression 
def computeGradLoss_logreg(w, X, y, L):
    N,D = X.shape

    # f = 1/N*sum_t log(1 + exp(-w'ytxt)) + .5*L*||w||^2
    l = 1.0/N* np.log(1. + np.exp(-y*np.dot(X,w))).sum() + .5*L*(w**2).sum() 
    # df = 1/N*sum_t -ytxt/(1 + exp(w'ytxt)) + L*w 
    g = -1.0/N*np.dot(y/(1. + np.exp(np.dot(X,w)*y)),X).reshape((D,)) + L*w

    return (g,l)
 

#softmax
def computeGradLoss_softmax(w, X, y, K, L):
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


# Gaussian noise
def GenerateGaussianNoise(scale=1.,tsize=None):
    noise = np.random.normal(0., scale, tsize)
    return noise

# Laplace noise
def GenerateLaplaceNoise(scale=1.,tsize=None):
    U = np.random.uniform(-0.5, 0.5,tsize)
    noise = - np.sqrt(0.5)*scale*np.sign(U)*np.log(1. - 2.*np.abs(U))
    return noise


#Train model, and retrieve/upload w and loss
def trainModel():

    print 'Setting up firebase'
    ref = firebase.FirebaseApplication(url, None)
    users = firebase.FirebaseApplication(url+'/users', None)
    auth_payload = {"uid": uid}
    token = create_token(secret, auth_payload)
    user = '/users/'+uid+'/'

    print 'Pre-loading test data'
    Xtest,ytest = loadData(dataDir,testFeatures,testLabels,Ntest,Dtest,Ktest)
    

    while True:
        paramIter = -1
        weightIter = -1
        if not localTraining: # Read all params from server
            print ' '
            print 'Downloading parameters from server'
            paramIter = np.int(ref.get('/parameters/paramIter', None, params = {"auth":token}))
            params['D'] = np.int(ref.get('/parameters/D', None, params = {"auth":token}))
            params['K'] = np.int(ref.get('/parameters/K', None, params = {"auth":token}))
            params['L'] = np.double(ref.get('/parameters/L', None, params = {"auth":token}))
            params['N'] = np.int(ref.get('/parameters/N', None, params = {"auth":token}))
            params['clientBatchSize'] = np.int(ref.get('/parameters/clientBatchSize', None, params = {"auth":token}))
            params['featureSource'] = ref.get('/parameters/featureSource', None, params = {"auth":token})
            params['labelSource'] = ref.get('/parameters/labelSource', None, params = {"auth":token})
            params['lossFunction'] = ref.get('/parameters/lossFunction', None, params = {"auth":token})
            params['noiseDistribution'] = ref.get('/parameters/noiseDistribution', None, params = {"auth":token})
            params['noiseScale'] = np.double(ref.get('/parameters/noiseScale', None, params = {"auth":token}))

        print params
        
        print 'Loading training data'
        X,y = loadData(dataDir,params['featureSource'],params['labelSource'],params['N'],params['D'],params['K'])

        # Re-init w
        w = initW(params['lossFunction'],params['D'],params['K'])

        print 'Begin iteration'
        for gradIter in range(1,maxiter+1):
            print ' '
            print 'paramIter = ', str(paramIter)
            print 'weightIter = ', str(weightIter)
            print 'gradIter = ', str(gradIter),'/',str(maxiter)

            # Ready to send weights?
            reset = False
            print 'Checking server status'
            while not localTraining:
                if (gradIter==1): # beginning
                    break;
                print '.',
                time.sleep(1.) # sleep for 1 sec
                paramIter_server = np.int(ref.get('parameters/paramIter', None, params = {"auth":token}))
                if (paramIter_server > paramIter): # parameter has changed. Reset
                    reset = True
                    break

                gradientProcessed = ref.get(user+'gradientProcessed', None, params = {"auth":token})
                gradIter_server = np.int(ref.get(user+'gradIter', None, params = {"auth":token}))
                #print 'gradientProcessed:',str(gradientProcessed),',   gradIter_server:',str(gradIter_server)
                if (gradientProcessed and gradIter_server == gradIter-1):
                    break
            print ' '
            if reset:
                print 'Parameter changed !!!'
                break;

            # Randomly choose (clientBatchSize) samples
            ind = np.random.choice(range(params['N']),size=(params['clientBatchSize'],),replace=False)        
            tX = X[ind,:]
            ty = y[ind]

            # Fetch iteration number and weight 
            if localTraining:
                weightIter = gradIter
            else:
                #print 'Fetching weights'
                weightIter = np.int(ref.get('/trainingWeights/iteration', None, params = {"auth":token}))
                #print 'weightIter= ', weightIter
                w = np.array(ref.get('/trainingWeights/weights', None, params = {"auth":token}),dtype=np.double)

            # Use one of loss functions.
            # The output is the averaged gradient
            if (params['lossFunction']=='Hinge'):
                g,l = computeGradLoss_hinge(w,tX,ty,params['L'])
            elif (params['lossFunction']=='LogReg'):
                g,l = computeGradLoss_logreg(w,tX,ty,params['L'])
            elif (params['lossFunction']=='Softmax'):
                g,l = computeGradLoss_softmax(w,tX,ty,params['K'],params['L'])
            elif (params['lossFunction']=='NN'):
                g,l = computeGradLoss_NN(w,tX,ty,params['K'],params['L'])
            else:
                print 'Unknown loss type'
                return
            
            if (params['noiseDistribution']=='NoNoise'):
                noise = np.zeros(w.shape)
            elif (params['noiseDistribution']=='Gauss'):
                noise = GenerateGaussianNoise(params['noiseScale'], w.shape)
            elif (params['noiseDistribution']=='Laplace'):
                noise = GenerateLaplaceNoise(params['noiseScale'], w.shape)
            else:
                print 'Unknown noise type'
                return
                    
            g += noise
            
            if np.isnan(g).any():
                print 'Nan in gradient'
                return
            #print str((g**2).sum())
            print 'loss = ',str(l)

            if localTraining:
                # Simple learning rate
                w -= naught/gradIter*g 
            else:
                print 'Uploading gradients'
                gradJson = g.tolist()
                ref.put(user, 'paramIter', paramIter, params = {"auth":token})                 
                ref.put(user, 'weightIter', weightIter, params = {"auth":token})          
                ref.put(user, 'gradIter', gradIter, params = {"auth":token})
                ref.put(user, 'gradients', gradJson, params = {"auth":token}) 
                ref.put(user ,'gradientProcessed', False, params = {"auth":token})

        ## Iteration ended
        #if (gradIter==maxiter):
            testModel(w,Xtest,ytest,params['K'])


        if localTraining:
            break
            
# Test
def testModel(w,X,y,K):

    # Implemented for only Softmax. Change it later.
    N,D = X.shape
    W = w.reshape((K,D))

    ypred = np.argmax(np.dot(X,W.T),axis=1) # N x K
    ind_correct = np.where(ypred==y)[0]    
    ncorrect = ind_correct.size
    rate = float(ncorrect) / float(ypred.size)
    print 'accuracy = ', str(rate)


# Load data
def loadData(dataDir,featureSource,labelSource,N,D,K):
    # Load data
    X = np.loadtxt(dataDir+'/'+featureSource, delimiter=',', dtype=float)
    #print X.shape
    
    if (X.shape[0]!=N):
        print 'Wrong number of samples'
        #return
    if (X.shape[1]!=D):
        print 'Wrong feature dimension'
        return
    y = np.loadtxt(dataDir+'/'+labelSource, dtype=float).astype(int)
    if (y.size!=N):
        print 'Wrong number of labels'
        #return
    if (K==2):
        y[y==0] = -1
        if any((y!=1) & (y!=-1)):
            print 'Wrong labels'
            return
    if (K>2):
        if any((y<0) | (y>K-1)):
            print 'Wrong labels'
            return

    return (X,y)


def initW(lossFunction,D,K):
    # Init w
    if (lossFunction=='Hinge'):
        w = np.zeros((D,),dtype = np.double)
    elif (lossFunction=='LogReg'):
        w = np.zeros((D,),dtype = np.double)
    elif (lossFunction=='Softmax'):
        w = np.zeros((D*K,),dtype = np.double)
    elif (lossFunction=='NN'):
        pass#g = computeGradLoss_NN(w,tX,ty,params['K'],params['L'])
    else:
        print 'Unknown loss type'
        return

    return w

###############################################################################################
# Begining of main


trainModel() 

