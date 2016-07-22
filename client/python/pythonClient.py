'''
Copyright 2016 Crowd-ML team

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License
'''

from firebase import firebase
from firebase_token_generator import create_token

import numpy as np
#import numpy.random
import time

from scipy.optimize import check_grad

import loss_hinge
import loss_logreg
import loss_softmax
import loss_nndemo1


####### Change below
'''
url = <Your firebase url>
uid = <arbitrary string>
secret = <Copy and paste the string from firebase db>
'''
maxiter = 100

## For testing
dataDir = './DataFiles' # Where you put your files
testFeatures = 'MNISTTestImages.50.l2.dat';
testLabels = 'MNISTTestLabels.dat';
Ntest = 1000
Dtest = 50#784
Ktest = 10

## For local training
#naught = 10.
localTraining = True # Set this true for faster, local training.

params = {}
if localTraining: # local version for testing purposes.
    params = {}
    params['D'] = 50#784
    params['K'] = 10
    params['L']= 1e-6
    params['N'] = 60000
    params['naught'] = 10.
    params['clientBatchSize'] = 100
    params['localUpdateNum'] = 10
    params['featureSource'] = 'MNISTTrainImages.50.l2.dat'
    params['labelSource'] = 'MNISTTrainLabels.dat'
    params['lossFunction'] = 'NNdemo1'#'Softmax'
    params['noiseDistribution'] = 'NoNoise'
    params['noiseScale'] = 0.


'''
# w is the paramter vector (, which is from the K x D matrix W for softmax)
# X is the N x D array of N samples of D-dimensional features
# y is the N x 1 array of N samples
# The output of a loss function is the averaged gradient over N samples, and the loss value
'''

 


############################################################################################################
## Gaussian noise
def GenerateGaussianNoise(scale=1.,tsize=None):
    noise = np.random.normal(0., scale, tsize)
    return noise

## Laplace noise
def GenerateLaplaceNoise(scale=1.,tsize=None):
    U = np.random.uniform(-0.5, 0.5,tsize)
    noise = - np.sqrt(0.5)*scale*np.sign(U)*np.log(1. - 2.*np.abs(U))
    return noise


############################################################################################################
## Train model, and retrieve/upload w and loss
def trainModel():

    print 'Setting up firebase'
    if not localTraining:
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
            params['naught'] = np.int(ref.get('/parameters/naught', None, params = {"auth":token}))
            params['clientBatchSize'] = np.int(ref.get('/parameters/clientBatchSize', None, params = {"auth":token}))
            params['featureSource'] = ref.get('/parameters/featureSource', None, params = {"auth":token})
            params['labelSource'] = ref.get('/parameters/labelSource', None, params = {"auth":token})
            params['lossFunction'] = ref.get('/parameters/lossFunction', None, params = {"auth":token})
            params['noiseDistribution'] = ref.get('/parameters/noiseDistribution', None, params = {"auth":token})
            params['noiseScale'] = np.double(ref.get('/parameters/noiseScale', None, params = {"auth":token}))
            params['localUpdateNum'] = np.int(ref.get('/parameters/localUpdateNum', None, params = {"autho":token}))
        print params
        
        print 'Loading training data'
        X,y = loadData(dataDir,params['featureSource'],params['labelSource'],params['N'],params['D'],params['K'])

        # Re-init w
        if (params['lossFunction']=='Hinge'):
            w = loss_hinge.init(params['D'])
        elif (params['lossFunction']=='LogReg'):
            w = loss_logreg.init(params['D'])
        elif (params['lossFunction']=='Softmax'):
            w = loss_softmax.init(params['D'],params['K'])
        elif (params['lossFunction']=='NNdemo1'):
            w = loss_nndemo1.init(params['D'],params['K'])
        else:
            print 'Unknown loss type'
            exit()
        

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

            # Fetch iteration number and weight 
            if localTraining:
                weightIter = gradIter
            else:
                #print 'Fetching weights'
                weightIter = np.int(ref.get('/trainingWeights/iteration', None, params = {"auth":token}))
                #print 'weightIter= ', weightIter
                w = np.array(ref.get('/trainingWeights/weights', None, params = {"auth":token}),dtype=np.double)

            if params['localUpdateNum']<=0 : 
                # SGD mode: compute and send the gradient
                tX,ty = sampleData(X,y,params)
                g, l = computeNoisyGradient(w,tX,ty,params)
            else: # Parameter averaging mode: compute and send the parameters
                for cnt in range(params['localUpdateNum']):
                    tX,ty = sampleData(X,y,params)
                    g,l = computeNoisyGradient(w,tX,ty,params)

                    # Simple learning rate
                    #w -= naught/gradIter*g 
                    w -= params['naught']/np.sqrt(gradIter*params['localUpdateNum'])*g # keep gradIter fixed?

            print 'loss = ',str(l)


            if localTraining:
                if params['localUpdateNum']<=0:
                    # Simple learning rate
                    #w -= naught/gradIter*g 
                    w -= params['naught']/np.sqrt(gradIter)*g
                else:
                    pass # Do nothing
            
            else:
                print 'Uploading gradients'
                gradJson = g.tolist()
                ref.put(user, 'paramIter', paramIter, params = {"auth":token})                 
                ref.put(user, 'weightIter', weightIter, params = {"auth":token})          
                ref.put(user, 'gradIter', gradIter, params = {"auth":token})
                ref.put(user, 'gradients', gradJson, params = {"auth":token}) 
                ref.put(user ,'gradientProcessed', False, params = {"auth":token})

        ## Iteration ended
        if (gradIter==maxiter):
            testModel(w,Xtest,ytest,params['K'],params['lossFunction'])


        if localTraining:
            break


def sampleData(X,y,params):
    # Randomly choose (clientBatchSize) samples
    ind = np.random.choice(range(params['N']),size=(params['clientBatchSize'],),replace=False)        
    tX = X[ind,:]
    ty = y[ind]

    return (tX,ty)


def computeNoisyGradient(w,tX,ty,params):

    # Use one of loss functions.
    # The output is the averaged gradient
    if (params['lossFunction']=='Hinge'):
        g,l = loss_hinge.getAvgGradient(w,tX,ty,params['L'])
    elif (params['lossFunction']=='LogReg'):
        g,l = loss_logreg.getAvgGradient(w,tX,ty,params['L'])
    elif (params['lossFunction']=='Softmax'):
        g,l = loss_softmax.getAvgGradient(w,tX,ty,params['L'],params['K'])
    elif (params['lossFunction']=='NNdemo1'):
        g,l = loss_nndemo1.getAvgGradient(w,tX,ty,params['L'],params['K'])
    else:
        print 'Unknown loss type'
        exit()
    
    if (params['noiseDistribution']=='NoNoise'):
        noise = np.zeros(w.shape)
    elif (params['noiseDistribution']=='Gauss'):
        noise = GenerateGaussianNoise(params['noiseScale'], w.shape)
    elif (params['noiseDistribution']=='Laplace'):
        noise = GenerateLaplaceNoise(params['noiseScale'], w.shape)
    else:
        print 'Unknown noise type'
        exit()
            
    g += noise
    
    if np.isnan(g).any():
        print 'Nan in gradient'
        exit()
    return (g,l)
    
            
## Test
def testModel(w,X,y,K,lossFunction):

    if (lossFunction=='Hinge'):
        ypred = loss_hinge.predict(w,X)
    elif (lossFunction=='LogReg'):
        ypred = loss_logreg.predict(w,X)
    elif (lossFunction=='Softmax'):
        ypred = loss_softmax.predict(w,X,K)
    elif (lossFunction=='NNdemo1'):
        ypred = loss_nndemo1.predict(w,X,K)
    else:
        print 'Unknown loss type'
        exit()

    ind_correct = np.where(ypred==y)[0]    
    ncorrect = ind_correct.size
    rate = float(ncorrect) / float(ypred.size)
    print 'accuracy = ', str(rate)


## Load data
def loadData(dataDir,featureSource,labelSource,N,D,K):
    # Load data
    X = np.loadtxt(dataDir+'/'+featureSource, delimiter=',', dtype=float)
    #print X.shape
    
    if (X.shape[0]!=N):
        print 'Wrong number of samples'
        exit()
        #return
    if (X.shape[1]!=D):
        print 'Wrong feature dimension'
        exit()
        #return
    y = np.loadtxt(dataDir+'/'+labelSource, dtype=float).astype(int)
    if (y.size!=N):
        print 'Wrong number of labels'
        exit()
        #return
    if (K==2):
        y[y==0] = -1
        if any((y!=1) & (y!=-1)):
            print 'Wrong labels'
            exit()
    if (K>2):
        if any((y<0) | (y>K-1)):
            print 'Wrong labels'
            exit()

    return (X,y)


###############################################################################################
## Begining of main

'''
loss_hinge.self_test1()
loss_logreg.self_test1()
loss_softmax.self_test1()
loss_nndemo1.self_test1()
exit()
'''

trainModel() 


