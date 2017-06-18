var firebase = require('firebase');

// Server requirements
var ERROR_CODE = 1;
var config = {};
var manage = {};

start()

function start() {
	loadConfig();
	validateConfig();
	if (config.testType !== "None") {
		manage.test = require('./tests/' + config.testType);
	} else {
		manage.test = { 
			accuracy: function accuracy(testWeight, constStr) {
				console.log('   Unknown test results. Configuration set to testType: None');
				console.log('');
			}
		};
	}
	setupFirebase();
	setupListeners();
}

function loadConfig() {
	var credentials = process.argv[2];
	if (!credentials) {
		console.log(new Error(
			"ERROR: Missing credentials argument:\n" +
			"  Expecting invocation of: node crowdML-server.js <credentials> <path-to-configuration>"));
		process.exit(ERROR_CODE);
	}
	
	var credentialsValues = require('./credentials.json');
	if (!credentialsValues[credentials]) {
		console.log(new Error(
			"ERROR: Missing credentials in credentials.json:\n" +
			"  Expecting credentials for supplied argument: " + credentials));
		process.exit(ERROR_CODE);
	}
	credentialsValues = credentialsValues[credentials];

	var cofigFile = process.argv[3];
	if (!cofigFile) {
		console.log(new Error(
			"ERROR: Missing configFile argument:\n" +
			"  Expecting invocation of: node crowdML-server.js <credentials> <path-to-configuration>"));
		process.exit(ERROR_CODE);
	}
	var configFilename = './' + cofigFile;
	config = require(configFilename);
	config.file = configFilename;

	// Credentials
	config.serviceAccount = credentialsValues.serviceAccount;
	config.databaseURL = credentialsValues.databaseURL;

	// Additions
	config.c = config.naughtRate;
	config.testFreq = config.testFrequency;
	config.weightBatchSize = 0;
	config.weightBatch = [];
	config.gradBatchSize = 0;
	config.gradBatch = [];
	config.iter = 1;
	config.testNum = 0;
	//TODO(tylermzeller) Not sure why this -1 is here. Consider removing.
	config.iterArray = [config.iter, -1];

	var D = config.D;
	var K = config.K;
	config.length = D;
	if (K > 2) {
		config.length = D * K;
	}
	if (config.lossFunction == 'SoftmaxNN') {
		config.length = D * nh + nh + nh * nh + nh + nh * K + K;
	}
	config.adaG = new Array(config.length);
	config.rms = new Array(config.length);
	config.initWeight = new Array(config.length);
	for(i = 0; i < config.length; i++){
		config.initWeight[i] = (Math.random() - 0.5);
		//initWeight[i] = 1;
		config.adaG[i] = 0;
		config.rms[i] = 0;
	}
	config.weightSet = [config.initWeight, config.iterArray];
}

function setupFirebase() {
	firebase.initializeApp({
		serviceAccount: config.serviceAccount,
		databaseURL: config.databaseURL
	});

	var ref = firebase.database().ref();
	manage.weights = ref.child('trainingWeights');
	manage.params = ref.child('parameters');
	manage.users = ref.child('users');

	var auth = firebase.auth();
	manage.serverToken = auth.createCustomToken("Server");
}

function validateConfig() {
	// Update the Server README with changes also
	var supportedDescentAlgs = ["constant", "adagrad", "simple", "sqrt", "rmsProp", "tf"];
	var supportedTestTypes = ["None", "binaryTest", "multiTest", "NNTest"];
	var supportedNoiseDistributions = ["NoNoise", "Gaussian", "Laplace"];
	var supportedLossFunctions = ["LogReg", "Hinge", "Softmax", "SoftmaxNN", "tf"];

	if (!supportedDescentAlgs.includes(config.descentAlg)) {
		console.log(new Error(
			"ERROR: Invalid descentAlg:\n" +
			"  Expecting one of the supportedDescentAlgs: " + supportedDescentAlgs + "\n" +
			"  Instead found: " + config.descentAlg));
		process.exit(ERROR_CODE);
	}

	if (!supportedTestTypes.includes(config.testType)) {
		console.log(new Error(
			"Error: Invalid testType:\n" +
			"  Expecting one of the supportedTestTypes: " + supportedTestTypes + "\n" +
			"  Instead found: " + config.testType));
		process.exit(ERROR_CODE);
	}

	if (!supportedNoiseDistributions.includes(config.noiseDistribution)) {
		console.log(new Error(
			"Error: Invalid noiseDistribution:\n" +
			"  Expecting one of the supportedNoiseDistributions: " + supportedNoiseDistributions + "\n" +
			"  Instead found: " + config.noiseDistribution));
		process.exit(ERROR_CODE);
	}

	if (!supportedLossFunctions.includes(config.lossFunction)) {
		console.log(new Error(
			"Error: Invalid lossFunction:\n" +
			"  Expecting one of the supportedLossFunctions: " + supportedLossFunctions + "\n" +
			"  Instead found: " + config.lossFunction));
		process.exit(ERROR_CODE);
	}
}

/*
 * 1. Initialized Values
 * 2. Setup Listeners
 */
function setupListeners() {
	// Initialize Values
	updateWeights(config.iter, [config.initWeight, config.iterArray]);
	console.log("[ Init: weights initialized   ]");

	// Send parameters to clients
	manage.params.update({
		// Model
		descentAlg: config.descentAlg,
		lossFunction: config.lossFunction,
		paramIter: config.paramIter,
		D: config.D,
		c: config.c,
		K: config.K,
		L: config.L,
		N: config.N,
		nh: config.nh,
		eps: config.eps,
		maxIter: config.maxIter,
		clientBatchSize: config.clientBatchSize,
		// Necessary for client-side weight calculation
		localUpdateNum: config.localUpdateNum,

		// Privacy
		noiseDistribution: config.noiseDistribution,
		noiseScale: config.noiseScale,

		// Data
		labelSource: config.labelSource,
		featureSource: config.featureSource,
	});
	console.log("[ Init: parameters set        ]");
	console.log("[ Init: complete              ]");

	// Setup Listeners
	manage.weights.on("value", function (snapshot, prevChildKey) {
		console.log("[ Weights changed             ]");
		var currentWeights = snapshot.val();
		var weightArrays = currentWeights.weights;
		config.currentWeight = weightArrays[0];
		config.currentIter = weightArrays[1][0];
	});
	console.log("[ Listeners: weights active   ]");

	manage.users.on("child_changed", function (snapshot) {
		console.log("[ User changed                ]");
		var user = snapshot.val();
		var grad = user.gradients;
		var processed = user.gradientProcessed;
		var userWeightIter = user.weightIter;
		var userParamIter = user.paramIter;
		var uid = snapshot.key;
		var userID = manage.users.child(uid);
		if (uid && !processed) {
			console.log('WeightIter: ' + userWeightIter + ' ' + config.iter);
			if (userWeightIter == config.iter && userParamIter == config.paramIter) {
				if (config.localUpdateNum == 0) {
					addToGradBatch(grad);
				} else {
					addToWeightBatch(grad);
				}
				console.log("[ Updating gradient processed ]");
				userID.update({
					gradientProcessed: true
				});
			}
		}
	});
	console.log("[ Listeners: users active     ]");
	console.log("[ Listeners: complete         ]");
	console.log("[ setupListeners complete     ]");
}

function updateWeights(iter, weights) {
	manage.weights.update({
		weights: weights,
		iteration: iter
	})
}

function addToGradBatch(gradient) {
	console.log("[ Adding to gradient batch    ]");
	config.gradBatch.push(gradient);
	config.gradBatchSize++;
	if (config.gradBatchSize == config.maxGradBatchSize) {
		var avgGradient = [];
		for (i = 0; i < gradient.length; i++) {
			var sum = 0;
			for (j = 0; j < config.maxGradBatchSize; j++) {
				sum += config.gradBatch[j][i];
			}
			avgGradient[i] = sum / config.maxGradBatchSize;
		}
		config.gradBatchSize = 0;
		config.gradBatch = [];

		console.log("[ Performing descent          ]");

		var newWeight = [];
		var c = config.c;
		var learningRate = c;
		var length = config.length;
		var eps = config.eps
		switch (config.descentAlg) {
			case 'constant':
				learningRate = c;
				for (i = 0; i < length; i++) {
					newWeight[i] = config.currentWeight[i] - (learningRate * avgGradient[i]);
				}
				break;
			case 'simple':
				learningRate = c / config.iter;
				for (i = 0; i < length; i++) {
					newWeight[i] = config.currentWeight[i] - (learningRate * avgGradient[i]);
				}
				break;
			case 'sqrt':
				learningRate = c / Math.sqrt(config.iter);
				for (i = 0; i < length; i++) {
					newWeight[i] = config.currentWeight[i] - (learningRate * avgGradient[i]);
				}
				break;
			case 'adagrad':
				for (i = 0; i < length; i++) {
					config.adaG[i] += gradient[i] * gradient[i];
					learningRate = c / Math.sqrt(config.adaG[i] + eps);
					newWeight[i] = config.currentWeight[i] - (learningRate * avgGradient[i]);
				}
				break;
			case 'rmsProp':
				for (i = 0; i < length; i++) {
					config.rms[i] = 0.9 * rms[i] + 0.1 * gradient[i] * gradient[i];
					learningRate = c / Math.sqrt(config.rms[i] + eps);
					newWeight[i] = config.currentWeight[i] - (learningRate * avgGradient[i]);
				}
				break;
			case 'tf':
				newWeight = config.currentWeight;
				break;
			default:
				console.log(new Error("ERROR: Didn't execute a valid descentAlg"));
		}
		addToWeightBatch(newWeight);
	}
}

function addToWeightBatch(weightArray) {
	console.log("[ Adding to weight batch      ]");
	config.weightBatch.push(weightArray);
	config.weightBatchSize++;
	if (config.weightBatchSize == config.maxWeightBatchSize) {
		var newWeight = [];
		for (i = 0; i < weightArray.length; i++) {
			var sum = 0;
			for (j = 0; j < config.maxWeightBatchSize; j++) {
				sum += config.weightBatch[j][i];
			}
			newWeight[i] = sum / config.maxWeightBatchSize;
		}

		config.testNum++;
		if (config.testNum == config.testFreq) {
			config.testNum = 0;
			console.log('Weight iteration ', config.iter);
			manage.test.accuracy(newWeight, '.' + config.file);
		}

		if (config.localUpdateNum > 0) {
			config.iter += config.localUpdateNum;
		} else {
			config.iter++;
		}

		console.log("[ Updating weights in DB      ]");
		config.iterArray = [config.iter, -1];
		updateWeights(config.iter, [newWeight, config.iterArray]);

		config.weightBatchSize = 0;
		config.weightBatch = [];
	}
}
