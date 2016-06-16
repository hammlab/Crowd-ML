This folder contains scripts for training with different training datasets and for generating a number of virtual devices on a single machine.


#### Tests

##### Test 1. (Run crowdML-server.js Constants1.js)
Binary MNIST classification using SVM (hinge loss) without noise and with simple gradient update.
(Look at [Constants1.js](./Constants1.js) to see the details.)

Results:
[figure for the test error up to 40 weight updates]



##### Test 2. (Run crowdML-server.js Constants2.js)
10-class MNIST data using Softmax without noise and with adagrad update rule.
(Look at [Constants2.js](./Constants2.js) to see the details.)

Results:
[figure for the test error up to 100 weight updates]



##### Test 3. 
With noise. 



