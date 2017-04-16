# Server
## Pre-Setup -- Local Build

1. Download and install [Node.js and npm](http://nodejs.org/en/) which can be done here: [nodejs.org](http://nodejs.org/)
2. Setup Node Firebase API with the commands:

```bash
npm install firebase --save
npm install firebase-token-generator --save
npm install firebase-tools --save
```

3. The local environment is now setup proceed to **step 4**

## Pre-Setup -- Docker

**NOTE**: One may not be able to run both Docker and an Android/iOS emulator. Consider using on device builds with the Docker container. 

1. Install [Docker](http://docker.com/)
2. Build the docker image and run the container:

```bash
# Builds the Dockerfile
# Switch `pwd` to $pwd` if running on PowerShell
docker build -t crowdml-server .
docker run -it --rm -v `pwd`:/usr/src/app/ crowdml-server bash
```

3. The docker environment is now setup proceed to **step 4**

## Final setup and running of the server

4. Create a Firebase service account through the [console](console.firebase.google.com) via: Settings > Permissions > Service accounts > Create

    - **NOTE** make sure to:
        - [x] Select 'Furnish a new private key'
        - [x] Get *key type* as `JSON`
        - [x] Leave unselected 'Enable Google Apps Domain-wide Delegation'

5. Download the service account `.json` file and place it in `server/`
6. Paste desired testing data, feature and label files (found in `data/`), into the `server/DataFiles/` folder
7. Setup config, server and gradient constant values in the `Constants.js` file see [Constants Setup](#constants-setup) (*Notice*: You can copy the `Constants.js` file to a numerated file)
8. Run command within your setup environment:

```bash
node crowdML-server.js <Desired Constants File>
```

## What the server does

The server listens to any changes in the data values of the various user trees, and uses any updated gradient information to create new weight values. The Constants folder contains parameter information for both the server itself as well as all of the clients. Any information needed by the clients is sent to the parameters branch of firebase upon running the server code, which the clients can read. The server also updates various iterators and process checks, which is detailed in the firebase ReadMe. Depending on the learning rate algorithm chosen by the user, the server will subtract the adjusted gradient from the current weight to create a new weight which is uploaded to firebase. The Server allows for gradient mini-batches of a size chosen by the user, in which the gradients collected are stored and when the batch is filled, the average is sent to be applied to the weight values. The server also allows for accuracy testing, which when enabled will compare every X weight values uploaded against a chosen dataset to see what percentage of test feature sets return the correct result.


## Constants Setup

One can change the firebase URL, keys, etc. by editing the `Constants.js` file. This allows one to affect the following:
* Parameter iteration
* Service Account json file name
* database URL (this can be found on the database page of your project console)
* feature size (D)
* maximum weight batch and gradient batch sizes
* naught learning rate
* epsilon value for learning rate
* number of classes (K)
* descent algorithm (constant|simple|sqrt|adagrad|rmsProp)
* test features folder
* test labels folder
* number of test samples (NTest)
* type of test to run (multiTest|binary|none)
* frequency of tests run per weight sent
* regularization constant (L)
* gradient loss type (LogReg|Hinge|Softmax)
* feature folder name
* test folder name
* Noise type (None|Gaussian|Laplace)
* Noise variance (noiseScale)
* Number of training samples (N)
* Client batch size
