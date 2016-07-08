	var constants = require('./Constants')

	var D = constants.D;
	var K = constants.K;
	var data = 'DataFiles/';
	var testFeatures = data.concat(constants.testFeatures);
	var testLabels = data.concat(constants.testLabels);
	var N = constants.testN;
	var nh = 200;

	function multiTest(testWeight){

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
	    			return str != "";}
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
	
		//var accuracy = 100*correct/N;
		//console.log('Accuracy: ', accuracy, '%')
		var accuracy = correct/N;
		console.log(accuracy)
	}

	function binaryTest(testWeight){

		var correct = 0;
		var labels = require('fs').readFileSync(testLabels).toString().split('\n')
		var features = require('fs').readFileSync(testFeatures).toString().split('\n')

		for(i = 0; i < N; i++){
			line = labels[i];
			var label = parseFloat(line, 10);	
			line = features[i];
			var featureStr = line.split(" ");
			var featureArray = [];
			for(var j=0; j<featureStr.length; j++) { 
				featureArray[j] = parseFloat(featureStr[j], 10);}
			dot = 0;		
			for(j = 0; j < D; j++){
				dot += featureArray[j]*testWeight[j];}		
			var predict = 0;
			if(dot > 0){
				predict = 1;}
			if(predict == label){
				correct++;}
		
		}
	
		var accuracy = 100*correct/N;
		console.log('Accuracy: ', accuracy, '%')
	}

	function NNTest(testWeight){

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
	    			return str != "";}
			var featureClean = featureStr.filter(valid);
			var featureArray = [];
			for(var j=0; j<featureClean.length; j++) { 
				featureArray[j] = parseFloat(featureClean[j], 10);}





			var W01 = []
			var b1 = []
			var W12 = []
			var b2 = []
			var W23 = []
			var b3 = []
			//Parse Parameters
			var count = 0;
			var end = count + D*nh;
			var place = 0;
			while(count < end){
			    W01[place] = testWeight[count];
			    place++;
			    count++;
			}
			end = count + nh;
			place = 0;
			while(count < end){
			    b1[place] = testWeight[count];
			    place++;
			    count++;
			}
			end = count + nh*nh;
			place = 0;
			while(count < end){
			    W12[place] = testWeight[count];
			    place++;
			    count++;
			}
			end = count + nh;
			place = 0;
			while(count < end){
			    b2[place] = testWeight[count];
			    place++;
			    count++;
			}
			end = count + nh*K;
			place = 0;
			while(count < end){
			    W23[place] = testWeight[count];
			    place++;
			    count++;
			}
			end = count + K;
			place = 0;
			while(count < end){
			    b3[place] = testWeight[count];
			    place++;
			    count++;
			}

			var dot;
			var h1 = [];
			place = 0;
			for(m = 0; m < nh; m++){
			    dot = 0;
			    for(j = 0; j < D; j++){
				dot += featureArray[j]*W01[m + j*(nh)];
			    }
			    if(dot + b1[m] > 0) {
				h1[place]=(dot + b1[m]);
			    }
			    else{
				h1[place]=0.0;
			    }
			}

			var h2 = new [];
			place = 0;
			for(m = 0; m < nh; m++){
			    dot = 0;
			    for(j = 0; j < nh; j++){
				dot += h1[j]*W12[m + j*(nh)];
			    }
			    if(dot + b2[m] > 0) {
				h2[place] = dot + b2[m];
			    }
			    else{
				h2[place] = 0.0;
			    }
			}

			var scores = []
			place = 0;
			for(m = 0; m < K; m++){
			    dot = 0;
			    for(j = 0; j < nh; j++){
				dot += h2[j]*W23[m + j*(K)];
			    }
			    scores[place] = (dot + b3[m]);
			}

			/*
			//dotMax used to prevent overflow
			var dotMax = 0;
			for(m = 0; m < K; m++){
			    dot = 0;
			    for(j = 0; j < nh; j++){
				dot += h2[j] * W23[j + (nh*m)];
			    }

			    if(dot > dotMax){
				dotMax = dot;
			    }
			}

			//denom = Σ(i:k) exp(Θ_i · X)
			double denom = 0;
			for(int i = 0; i < K; i++){
			    //dot product w_i·x
			    dot = 0;
			    for(int j = 0; j < nh; j++){
				dot += h2.get(j) * W23.get(j + (nh*i));
			    }

			    denom += Math.exp(dot - dotMax);
			}


			List<Double> probs = new ArrayList<Double>(K);

			for(int i = 0; i < K; i++) {

			    //prob_i = exp(Θ_i · X)/denom
			    dot = 0;
			    for (int j = 0; j < nh; j++) {
				dot += h2.get(j) * W23.get(j + (nh * i));
			    }

			    probs.add(Math.exp(dot - dotMax) / denom);
			}

			*/


			var bestGuess = 0;
			for(h = 0; h < K; h++){
				if(scores[h]>scores[bestGuess]){
					bestGuess = h;}
			}

			if(bestGuess == label){
				correct++;}		
		}
	
		var accuracy = correct/N;
		console.log(accuracy)
	}

	exports.binaryTest = binaryTest;
	exports.multiTest = multiTest;
	exports.NNTest = NNTest;
