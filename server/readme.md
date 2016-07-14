#### How to set up server-side app

1. Download and install Node which can be done here: https://nodejs.org/en/
2. Setup Node Firebase API with the commands:

   ```
  $npm install firebase --save
  $npm install firebase-token-generator --save
  $npm install firebase-tools
  ```
3. Go to your project page in the firebase website. Download firebase service account json object from <your firebase app>/Settings/Permissions/Service Accounts/Create service account' and place it in same directory as server. 

   *Make sure to select 'Furnish a new private key' and select the key type as JSON, and leave 'Enable Google Apps Domain-wide Delegation' unselected.*
4. Paste Testing data feature and label files (found in the data folder) into a folder labelled 'DataFiles' and place it in the Server directory.   
5. Set server and gradient constant values in Constants.js file
6. Download crowdML-server.js 
7. Run command

  ```
  $nodejs crowdML-server.js [Constants file name you wish to use]
  ```

#### What the server-side app does

 The server listens to any changes in the data values of the various user trees, and uses any updated gradient information to create new weight values. The Constants folder contains parameter information for both the server itself as well as all of the clients. Any information needed by the clients is sent to the parameters branch of firebase upon running the server code, which the clients can read. The server also updates various iterators and process checks, which is detailed in the firebase ReadMe. Depending on the learning rate algorithm chosen by the user, the server will subtract the adjusted gradient from the current weight to create a new weight which is uploaded to firebase. The Server allows for gradient mini-batches of a size chosen by the user, in which the gradients collected are stored and when the batch is filled, the average is sent to be applied to the weight values. The server also allows for accuracy testing, which when enabled will compare every X weight values uploaded against a chosen dataset to see what percentage of test feature sets return the correct result.


#### Setting up firebase URL, keys, etc

Editing the Constants.js file allows you to affect the:
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

