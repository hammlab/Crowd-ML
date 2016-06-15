	var firebase = require('firebase');
	var constants = require('./Constants')
	var test = require('./test')

	firebase.initializeApp({
  		serviceAccount: constants.serviceAccount,
  		databaseURL: constants.databaseURL
	});

	var db = firebase.database();

	var ref = db.ref();
	var weight = ref.child('trainingWeights');
	var params = ref.child('parameters');
	var users = ref.child('users');

	var D = constants.D;
	var maxWeightBatchSize = 1;
	var maxGradBatchSize = constants.maxGradBatchSize;
	var c = constants.naughtRate;
	var eps = constants.eps;
	var K = constants.K;
	var descentAlg = constants.descentAlg;
	var testType = constants.testType;
	var testFreq = constants.testFrequency;

	var testNum = 0;	
	var changeiter = 0;
	var D = constants.D;
	var weightBatchSize = 0;
	var weightBatch = [];
	var gradBatchSize = 0;
	var gradBatch = [];
	var iter = 1;
	var learningRate = c;

	var auth = firebase.auth();
	var token = auth.createCustomToken("Server");

	//reset weights to 0 for testing, initialize learning rate matrices
	var length = D;
	if (K > 2){
		length = D*K;}
	var adaG = new Array(length);
	var rms = new Array(length);

	var zeroWeight = new Array(length);
	for(i = 0; i < length; i++){
		zeroWeight[i] = 0;
		adaG[i] = 0;
		rms[i] = 0;}
	weight.update({
		weights: zeroWeight,
		iteration: iter
		});
	console.log('zeroes set');

	//send parameters to clients
	params.update({
		paramIter: constants.paramIter,
		L: constants.L,
		noiseScale: constants.noiseScale,
		noiseDistribution: constants.noiseDistribution,
		K: K,
		lossFunction: constants.lossFunction,
		labelSource: constants.labelSource,
		featureSource: constants.featureSource,
		D: D,
		N: constants.N,
		clientBatchSize: constants.clientBatchSize
		})
	console.log('parameters set');



	var currentWeight;
	var currentIter;

	weight.on("value", function(snapshot, prevChildKey) {
		var currentWeights = snapshot.val();
		currentWeight = currentWeights.weights;
		currentIter = currentWeights.iteration;
	});


	users.on("child_changed", function(snapshot) {
		changeiter++;
  		var user = snapshot.val();
        	var grad = user.gradients;
		var processed = user.gradientProcessed;
		var userWeightIter = user.weightIter;
		var userParamIter = user.paramIter;
		var uid = snapshot.key;
		var userID = users.child(uid);		
		if(uid && !processed){
			if(userWeightIter == iter && userParamIter == constants.paramIter){
				addToGradBatch(grad);
			}
			
			userID.update({
				gradientProcessed: true
				});
		}
		
    });


function addToGradBatch(gradient){
	gradBatch.push(gradient);
	gradBatchSize++;
	if(gradBatchSize == maxGradBatchSize){
		var avgGradient = [];
		for(i = 0; i < gradient.length; i++){
			var sum = 0;
			for(j = 0; j < maxGradBatchSize; j++){
				sum += gradBatch[j][i];
				}
			avgGradient[i] = sum/maxGradBatchSize;
			}
		gradBatchSize = 0;
		gradBatch = [];

		var newWeight = [];
		if(descentAlg=='constant'){
			learningRate = c;
			for (i = 0; i < length; i++) { 
				newWeight[i] = currentWeight[i] - (learningRate * gradient[i]);
			}
		}
		if(descentAlg=='simple'){
			learningRate = c/iter;
			for (i = 0; i < length; i++) { 
				newWeight[i] = currentWeight[i] - (learningRate * gradient[i]);
			}
		}
		else if(descentAlg=='sqrt'){
			learningRate = c/Math.sqrt(iter);
			for (i = 0; i < length; i++) { 
				newWeight[i] = currentWeight[i] - (learningRate * gradient[i]);
			}
		}
		else if(descentAlg=='adagrad'){
			for (i = 0; i < length; i++) { 
				adaG[i] += gradient[i]*gradient[i];
				learningRate = c/Math.sqrt(adaG[i]+eps);
				newWeight[i] = currentWeight[i] - (learningRate * gradient[i]);
			}
		}
		else if(descentAlg=='rmsProp'){
			for (i = 0; i < length; i++) { 
				rms[i] = 0.9*rms[i] + 0.1*gradient[i]*gradient[i];
				learningRate = c/Math.sqrt(rms[i]+eps);
				newWeight[i] = currentWeight[i] - (learningRate * gradient[i]);
			}
		}	

		addToWeightBatch(newWeight);
		
	}
}

function addToWeightBatch(weightArray){
	weightBatch.push(weightArray);
	weightBatchSize++;
	if(weightBatchSize == maxWeightBatchSize){
		var newWeight = [];
		for(i = 0; i < weightArray.length; i++){
			var sum = 0;
			for(j = 0; j < maxWeightBatchSize; j++){
				sum += weightBatch[j][i];
				}
			newWeight[i] = sum/maxWeightBatchSize;
			}

		testNum++;
		if(testNum == testFreq && testType == 'multiTest'){
			testNum = 0;
			test.multiTest(newWeight);}
		if(testNum == testFreq && testType == 'binary'){
			testNum = 0;
			test.binaryTest(newWeight);}

		iter++;
		weight.update({
			iteration: iter
			});
		weight.update({
			weights: newWeight
			});
		weightBatchSize = 0;
		weightBatch = [];
	}
}






	
