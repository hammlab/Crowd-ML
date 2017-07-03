package osu.crowd_ml;

/*
Copyright 2017 Crowd-ML team


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
*/

import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowTrainingInterface;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import osu.crowd_ml.R;
import osu.crowd_ml.firebase.CrowdMLApplication;

public class TensorFlowTrainer implements Trainer {

    final static private int stepsToTest = 10;
    private Parameters params;
    private double[] weights;
    private int t;

    private boolean init = false;

    private static TensorFlowTrainer instance = null;

    private TensorFlowTrainingInterface trainingInterface = null;

    private TensorFlowTrainer(){
    }
    
    public static TensorFlowTrainer getInstance() {
        if (instance == null){
            instance = new TensorFlowTrainer();
        }
        return instance;
    }

    private TensorFlowTrainingInterface getTrainingInterface(){
        if (trainingInterface == null){
            AssetManager am = CrowdMLApplication.getAppContext().getAssets();
            String modelName = CrowdMLApplication.getAppContext().getString(R.string.model_name_TF);
            trainingInterface = new TensorFlowTrainingInterface(am, modelName);
        }
        return trainingInterface;
    }

    @Override
    public double[] train(final int numIterations) {
        int length = weights.length; // get the length of all the parameters
        String initName  = params.getTfInitOp();
        //String weightsIn = CrowdMLApplication.getAppContext().getString(R.string.weights_in_TF);

//        float[] w = new float[weights.length];
//        for (int j = 0; j < weights.length; j++) {
//            w[j] = (float)weights[j];
//        }

        float[] all_params = new float[length];

        // Initialize the training interface if this is the first round of training
        trainingInterface = getTrainingInterface();

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("beginTraining");

        if (!init) {
            Trace.beginSection("init_vars");
            trainingInterface.run(new String[]{}, new String[]{initName});
            Trace.endSection(); // init vars
            init = true;
        }

        for (int i = 0; i < numIterations; i++) {
            Log.d("train", "iteration " + (i + 1));
            runTrainingStep(all_params);

            if (i == 0 || (i+1) % stepsToTest == 0){
                runTest();
            }
        }

        Trace.endSection(); // "beginTraining"

        double[] newWeights = new double[length]; // TODO: include the bias
        for (int j = 0; j < length; j++){
            newWeights[j] = (double)all_params[j];
        }
        return newWeights;
    }

    private void runTest() {
        int K = params.getK();
        int D = params.getD();
        int testN = params.getTestN();
        String testName  = params.getTfTestOp();
        String samplesName  = params.getTfFeaturesName();
        String labelsName = params.getTfLabelsName();
        float[] testFeatures = TrainingDataIO.getInstance().getTFTestFeatures(testN, params);
        float[] testLabels = TrainingDataIO.getInstance().getTFTestingLabels(testN, params);

        Log.d("train", "feeding test samples and labels");

        // Copy the test data into TensorFlow.
        Trace.beginSection("feed");
        trainingInterface.feed(samplesName, testFeatures, testN, D);
        trainingInterface.feed(labelsName, testLabels, testN, K);
        Trace.endSection(); // feed

        // Run the inference call.
        Trace.beginSection("test");
        trainingInterface.run(new String[]{testName}, new String[]{});
        Trace.endSection(); // test

        // Copy the accuracy Tensor into the output array.
        float[] outputs = new float[1];
        Trace.beginSection("fetch");
        trainingInterface.fetch(testName, outputs);
        Trace.endSection(); // fetch
        Log.d("train", (outputs[0] * 100) + "%");
    }

    private void runTrainingStep(float[] all_params) {
        int batchSize = params.getClientBatchSize();
        int N = params.getN();
        int K = params.getK();
        int D = params.getD();
        String trainName = params.getTfTrainOp();
        String samplesName  = params.getTfFeaturesName();
        String labelsName = params.getTfLabelsName();
        List<String> trainableParameters = params.getTfParameters();
        float[] trainFeatureBatch;
        float[] trainLabelBatch;

        int[] indices = new int[batchSize];
        for (int j = 0; j < batchSize; j++){
            indices[j] = new Random().nextInt(N);
        }

        // Get the training feature
        trainFeatureBatch = TrainingDataIO.getInstance().getTFFeatureBatch(indices, params);
        // Get the training label
        trainLabelBatch = TrainingDataIO.getInstance().getTFLabelBatch(indices, params);

        Log.d("train", "feeding samples, weights, and labels");

        // Copy the training data into TensorFlow.
        Trace.beginSection("feed");
        trainingInterface.feed(samplesName, trainFeatureBatch, batchSize, D);
        //trainingInterface.feed(weightsIn, w, D, K); // TODO: inject bias
        trainingInterface.feed(labelsName, trainLabelBatch, batchSize, K);
        Trace.endSection();

        Log.d("train", "training step");

        // Run a single step of training
        Trace.beginSection("train");
        trainingInterface.run(trainableParameters.toArray(new String[0]), new String[]{trainName});
        Trace.endSection();

        Log.d("train", "fetching weights");

        Trace.beginSection("fetch");
        // Copy all trainable Tensor's into a 1-D Java array
        getTrainableParameters(trainableParameters, all_params);
        Trace.endSection();
    }

    private void getTrainableParameters(List<String> trainableParameters, float[] all_params) {
        int offset = 0;
        // Copy the weights Tensor into the weights array.
        for (String param : trainableParameters) {
            // Fetch the Tensor's value
            float[] temp = getTensorValue(param);
            concatParams(all_params, temp, offset);
            offset += temp.length;
        }
    }

    private void concatParams(float[] all_params, float[] temp, int offset) {
        for (int i = 0; i < temp.length; i++){
            all_params[offset + i] = temp[i];
        }
    }

    private float[] getTensorValue(String param) {
        // Get the length of this Tensor
        int length = trainingInterface.getNumElements(param);
        // Create a temporary array to hold the Tensor's value
        float[] temp = new float[length];
        // Fetch the Tensor's value
        trainingInterface.fetch(param, temp);
        return temp;
    }

    @Override
    public List<Double> getNoisyGrad() {
        // TODO(tylermzeller): How to get gradients from tensorflow?
        return null;
    }

    @Override
    public Trainer setIter(int t) {
        this.t = t;
        return getInstance();
    }

    @Override
    public Trainer setWeights(double[] weights) {
        this.weights = weights;
        return getInstance();
    }

    @Override
    public Trainer setParams(Parameters params) {
        this.params = params;
        return getInstance();
    }

    @Override
    public void destroy() {
        Log.d("destroy", "destroy this trainer.");
        trainingInterface.close();
        weights = null;
        params = null;
        instance = null;
    }
}
