
	var paramIter = 0;
	
	var serviceAccount = 'GD-Prototype-credentials.json';	
	var databaseURL = 'https://gd-prototype.firebaseio.com';
	var D = 784;
	//var maxWeightBatchSize = 1;
	var maxGradBatchSize = 1;
	var naughtRate = 10;
	var eps = 0.00000001;
	var K = 10;
	var descentAlg = 'adagrad'; //simple, sqrt, adagrad, rmsProp
	var testFeatures = 'MNISTTestImages.dat';
	var testLabels = 'MNISTTestLabels.dat';
	var testN = 1000;
	var testType = 'multiTest'; //none, binary, multitest
	var testFrequency = 1;

	var L = 1e-6;
    	var noiseScale = 1;
    	var noiseDistribution = 'NoNoise';
    	var lossFunction = 'SoftmaxNN';
    	var labelSource = 'MNISTTrainLabels.dat';
    	var featureSource = 'MNISTTrainImages.dat';
	var N = 60000;
	var clientBatchSize = 1;

	if(descentAlg != 'simple' && descentAlg != 'sqrt' && descentAlg != 'adagrad' && descentAlg != 'rmsProp')
	{
		throw new Error("Invalid descent Algorithm");
		}

	if(testType != 'None' && testType != 'binary' && testType != 'multiTest')
	{
		throw new Error("Invalid test Type");
		}

	if(noiseDistribution != 'NoNoise' && noiseDistribution != 'Gaussian' && noiseDistribution != 'Laplace')
	{
		throw new Error("Invalid Noise Type");
		}

	if(lossFunction != 'LogReg' && lossFunction != 'Hinge' && lossFunction != 'Softmax' && lossFunction != 'SoftmaxNN')
	{
		throw new Error("Invalid Loss Type");
		}
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

