
	var paramIter = 0;
	
	var serviceAccount = <Enter json file name here>;	
	var databaseURL = <Enter Firebase database url here>;
	var D = 784;
	//var maxWeightBatchSize = 1;
	var maxGradBatchSize = 1;
	var naughtRate = 0.01;
	var eps = 0.00000001;
	var K = 10;
	var descentAlg = 'adagrad'; //constant, simple, sqrt, adagrad, rmsProp
	var testFeatures = 'MNISTTestImages.dat';
	var testLabels = 'MNISTTestLabels.dat';
	var testN = 1000;
	var testType = 'multiTest'; //none, binary, multiTest
	var testFrequency = 1;

	var L = 1e-6;
    	var noiseScale = 1;
    	var noiseDistribution = 'NoNoise';
    	var lossFunction = 'Softmax';
    	var labelSource = 'MNISTTrainLabels.dat';
    	var featureSource = 'MNISTTrainImages.dat';
	var N = 10000;
	var clientBatchSize = 10;

	exports.paramIter = paramIter;
	exports.serviceAccount = serviceAccount;
	exports.databaseURL = databaseURL;
	exports.D = D;
	//exports.maxWeightBatchSize = maxWeightBatchSize;
	exports.maxGradBatchSize = maxGradBatchSize;
	exports.naughtRate = naughtRate;
	exports.eps = eps;
	exports.K = K;
	exports.descentAlg = descentAlg;
	exports.testFeatures = testFeatures;
	exports.testLabels = testLabels;
	exports.testN = testN;
	exports.testType = testType;
	exports.testFrequency = testFrequency;
	exports.L = L;
	exports.noiseScale = noiseScale;
	exports.noiseDistribution = noiseDistribution;
	exports.lossFunction = lossFunction;
	exports.labelSource = labelSource;
	exports.featureSource = featureSource;
	exports.N = N;
	exports.clientBatchSize = clientBatchSize;

