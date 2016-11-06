
	var paramIter = 0;
	
	var serviceAccount = <insert service account file name>;	
	var databaseURL = <insert database URL>;
	var D = 785;
	var maxWeightBatchSize = 1;
	var localUpdateNum = 0; //set to 0 to disable
	var maxGradBatchSize = 1;
	var naughtRate = 0.01;
	var eps = 0.00000001;
	var K = 2;
	var descentAlg = 'simple'; //simple, sqrt, adagrad, rmsProp
	var testFeatures = 'binaryTestImages.dat';
	var testLabels = 'binaryTestLabels.dat';
	var testN = 220;
	var testType = 'binaryTest'; //none, binaryTest, multiTest, NNTest
	var testFrequency = 1;
	var maxIter = 50;

	var L = 1e-6;
    	var noiseScale = 1;
    	var noiseDistribution = 'NoNoise';
    	var lossFunction = 'Hinge';
    	var labelSource = 'binaryTrainLabels.dat';
    	var featureSource = 'binaryTrainImages.dat';
	var N = 12445;
	var clientBatchSize = 1;
	var nh = 1;

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

	if(lossFunction != 'LogReg' && lossFunction != 'Hinge' && lossFunction != 'Softmax')
	{
		throw new Error("Invalid Loss Type");
		}
	exports.paramIter = paramIter;
	exports.serviceAccount = serviceAccount;
	exports.databaseURL = databaseURL;
	exports.D = D;
	exports.maxWeightBatchSize = maxWeightBatchSize;
	exports.localUpdateNum = localUpdateNum;
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
	exports.maxIter = maxIter;
	exports.L = L;
	exports.noiseScale = noiseScale;
	exports.noiseDistribution = noiseDistribution;
	exports.lossFunction = lossFunction;
	exports.labelSource = labelSource;
	exports.featureSource = featureSource;
	exports.N = N;
	exports.clientBatchSize = clientBatchSize;
	exports.nh = nh;

