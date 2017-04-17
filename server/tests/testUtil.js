function printResult(accuracy, correct, N, lessAccurate=null) {
    console.log('   Accuracy: ', accuracy, '%')
	if (lessAccurate != null) {
		console.log('   Less Accuracy: ', lessAccurate, '%')
	}
	console.log('   Correct : ', correct)
	console.log('   Total   : ', N)
	console.log('')
}

exports.printResult = printResult;
