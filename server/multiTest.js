	function accuracy(testWeight, constStr){
		var constants = require(constStr)

		var D = constants.D;
		var K = constants.K;
		var data = 'DataFiles/';
		var testFeatures = data.concat(constants.testFeatures);
		var testLabels = data.concat(constants.testLabels);
		var N = constants.testN;

		var correct = 0;
		var error = 0;
		var labels = require('fs').readFileSync(testLabels).toString().split('\n')
		var features = require('fs').readFileSync(testFeatures).toString().split('\n')
		for(i = 0; i < N; i++){
			var classResults = [];
			line = labels[i];
			var label = parseFloat(line, 10);
			line = features[i];
			var featureStr = line.split(/,| /);
			function valid(str) {
	    	return str != "";
			}
			var featureClean = featureStr.filter(valid);
			var featureArray = [];
			for(var j=0; j<featureClean.length; j++) {
				featureArray[j] = parseFloat(featureClean[j], 10);}

			for(h = 0; h < K; h++){
				dot = 0;
				for(j = 0; j < D; j++){
					dot += featureArray[j]*testWeight[j + (h*D)];}
				classResults[h] = dot;
			}
			var bestGuess = 0;
			for(h = 0; h < K; h++){
				if(classResults[h]>classResults[bestGuess]){
					bestGuess = h;}
			}

			if(bestGuess == label){
				correct++;}
		}

		var accuracy = 100*correct/N;
		console.log('Accuracy: ', accuracy, '%')
		console.log('Correct : ', correct)
		console.log('Total   : ', N)
	}

	exports.accuracy = accuracy;
