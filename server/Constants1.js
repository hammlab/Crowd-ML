var paramIter = 0;
var serviceAccount = 'Crowd-ML-e591994762fe.json';
var databaseURL = 'https://crowd-ml-6228f.firebaseio.com/';
var D = 785;
var maxWeightBatchSize = 1;
var localUpdateNum = 15;
var maxGradBatchSize = 1;
//var naughtRate = 0.00001;
var naughtRate = .001;
var eps = 0.00000001;
var K = 2;
var descentAlg = 'sqrt'; //simple, sqrt, adagrad, rmsProp
var testFeatures = 'binaryTestImages.dat';
var testLabels = 'binaryTestLabels.dat';
var testN = 220;
var testType = 'binaryTest'; //none, binary, multitest
var testFrequency = 1;
var maxIter = 1000;

var L = 1e-6;
var noiseScale = 1;
var noiseDistribution = 'NoNoise';
var lossFunction = 'Hinge';
var labelSource = 'binaryTrainLabels.dat';
var featureSource = 'binaryTrainImages.dat';
var N = 12665;
var clientBatchSize = 1;
var nh = 1;

// Node.js error exit code is 1, success is 0 (default)
var ERROR_CODE = 1

if(descentAlg != 'simple' && descentAlg != 'sqrt' && descentAlg != 'adagrad' && descentAlg != 'rmsProp'){
	// It's better to simply log the error then exit the process. Throwing
	// an error produces a stacktrace which obfuscates the problem.
	//throw new Error("Invalid descent Algorithm");

	console.log("Invalid descent algorithm: Expecting \'simple\', \'sqrt\', \'adagrad\', or \'rmsProp\'.");
	console.log("Instead found: " + descentAlg);
	process.exit(ERROR_CODE);
}

if(testType != 'None' && testType != 'binaryTest' && testType != 'multiTest' && testType != 'NNTest'){
	// See comment above
	//throw new Error("Invalid test Type");

	console.log("Invalid test type: Expecting \'None\', \'binaryTest\', \'multiTest\', or \'NNTest\'.");
	console.log("Instead found: " + testType);
	process.exit(ERROR_CODE);
}

if(noiseDistribution != 'NoNoise' && noiseDistribution != 'Gaussian' && noiseDistribution != 'Laplace'){
	// See comment above
	//throw new Error("Invalid Noise Type");

	console.log("Invalid noise type: Expecting \'NoNoise\', \'Gaussian\', or \'Laplace\'.");
	console.log("Instead found: " + noiseDistribution);
	process.exit(ERROR_CODE);
}

if(lossFunction != 'LogReg' && lossFunction != 'Hinge' && lossFunction != 'Softmax' && lossFunction != 'SoftmaxNN'){
	// See comment above
	//throw new Error("Invalid Loss Type");

	console.log("Invalid loss type: Expecting \'LogReg\', \'Hinge\', \'Softmax\', or \'SoftmaxNN\'.");
	console.log("Instead found: " + lossFunction);
	process.exit(ERROR_CODE);
}

exports.localUpdateNum = localUpdateNum;
exports.maxIter = maxIter;
exports.paramIter = paramIter;
exports.serviceAccount = serviceAccount;
exports.databaseURL = databaseURL;
exports.D = D;
exports.maxWeightBatchSize = maxWeightBatchSize;
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
exports.nh = nh;
