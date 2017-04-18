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
7. Tune config, server and gradient values in a `configuration.json` file located in `server/config/` see [Configuration Setup](#configuration-setup) (*Notice*: One supplies this file to setup the server)
8. Add ones credentials to a `credentials.json` file in `server/` with the format given by [Credentials Setup](#credentials-setup) 
8. Run the following command within your setup environment:

```bash
# ex: node crowdML-server.js crowdml config/configuration.json
node crowdML-server.js <Credentials Name> <Desired Configuration File>
```

## What the server does

The server listens to any changes in the data values of the various user trees, and uses any updated gradient information to create new weight values. The Constants folder contains parameter information for both the server itself as well as all of the clients. Any information needed by the clients is sent to the parameters branch of firebase upon running the server code, which the clients can read. The server also updates various iterators and process checks, which is detailed in the firebase ReadMe. Depending on the learning rate algorithm chosen by the user, the server will subtract the adjusted gradient from the current weight to create a new weight which is uploaded to firebase. The Server allows for gradient mini-batches of a size chosen by the user, in which the gradients collected are stored and when the batch is filled, the average is sent to be applied to the weight values. The server also allows for accuracy testing, which when enabled will compare every X weight values uploaded against a chosen dataset to see what percentage of test feature sets return the correct result.


## Configuration
### Credentials Setup

Create a `credentials.json` file within `server/` with the following format and filling in the `<...>` blanks with your credentials data:

```json
{
    "<name-of-credential-set>": {
        "serviceAccount": "<service-account-filename.json>",
        "databaseURL": "<firebase-url>"
    }
}
```

An example of a completed file is as follows. Multiple credentials can be added with different names:

```json
{
    "crowdml": {
        "serviceAccount": "Crowd-ML-XXXXXXXXXXXX.json",
        "databaseURL": "https://crowd-ml-XXXXX.firebaseio.com/"
    }
}
```

### Configuration Setup

One can change the firebase URL, keys, etc. by editing the `configuration.json` file. This allows one to affect the following:

Area | Field | Modifies
--- | --- | ---
Model | `descentAlg` | Descent algorithm (constant|simple|sqrt|adagrad|rmsProp)
 | `lossFunction` | Gradient loss type (LogReg|Hinge|Softmax)
 | `paramIter` | Parameter iteration
 | `maxIter` |
 | `D` | Feature size
 | `naughtRate` | Naught learning rate
 | `K` | Number of classes
 | `L` | Regularization constant
 | `N` | Number of training samples
 | `nh` |
 | `eps` | Epsilon value for learning rate
 | `maxWeightBatchSize` | Maximum weight batch
 | `maxGradBatchSize` | Maximum gradient batch sizes
 | `clientBatchSize` | Client batch size
 | `localUpdateNum` |
 | |
Privacy | `noiseDistribution` | Noise type (None|Gaussian|Laplace)
 | `noiseScale` | Noise variance
 | |
Data | `featureSource` | Feature file
 | `labelSource` | Label file
 | |
Tests | `testFeatures` | Test features file
 | `testLabels` | Test labels file
 | `testN` | Number of test samples (NTest)
 | `testType` | Type of test to run (multiTest|binary|none)
 | `testFrequency` | Frequency of tests run per weight sent

### Supported Opperations

Field | Supported Values
--- | ---
`descentAlg` | "constant", "adagrad", "simple", "sqrt", "rmsProp"
`lossFunction` | "LogReg", "Hinge", "Softmax", "SoftmaxNN"
`testType` | "None", "binaryTest", "multiTest", "NNTest"
`noiseDistribution` | "NoNoise", "Gaussian", "Laplace"
