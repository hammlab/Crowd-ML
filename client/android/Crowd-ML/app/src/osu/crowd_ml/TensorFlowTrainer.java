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

import java.util.List;
import java.util.Random;

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
        int batchSize = params.getClientBatchSize();
        int N = params.getN();
        int K = params.getK();
        int D = params.getD();
        int testN = params.getTestN();
        String initName  = CrowdMLApplication.getAppContext().getString(R.string.init_name_TF);
        String trainName = CrowdMLApplication.getAppContext().getString(R.string.train_name_TF);
        String testName  = CrowdMLApplication.getAppContext().getString(R.string.test_name_TF);
        String samplesName  = CrowdMLApplication.getAppContext().getString(R.string.input_name_TF);
        String labelsName = CrowdMLApplication.getAppContext().getString(R.string.label_name_TF);
        String weightsIn = CrowdMLApplication.getAppContext().getString(R.string.weights_in_TF);
        String weightsOp = CrowdMLApplication.getAppContext().getString(R.string.weights_op_TF);

        float[] w = new float[weights.length];
        for (int j = 0; j < weights.length; j++) {
            w[j] = (float)weights[j];
        }

        float[] all_params = new float[D * K + K];

        // Initialize the training interface if this is the first round of training
        trainingInterface = getTrainingInterface();

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("beginTraining");

        if (!init) {
            Trace.beginSection("init_vars");
            Log.d("train", "initializing all vars");
            trainingInterface.run(new String[]{}, new String[]{initName});
            Trace.endSection();
            init = true;
        }

        for (int i = 0; i < numIterations; i++) {

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
            trainingInterface.run(new String[]{weightsOp, "get_params"}, new String[]{trainName});
            Trace.endSection();

            Log.d("train", "fetching weights");

            // Copy the weights Tensor into the weights array.
            Trace.beginSection("fetch");
            trainingInterface.fetch(weightsOp, w);
            trainingInterface.fetch("get_params", all_params);
            Log.d("all params", all_params[0] + "");
            Trace.endSection();

            Log.d("train", "iteration " + i);

            if (i == 0 || (i+1) % stepsToTest == 0){
                float[] testFeatures = TrainingDataIO.getInstance().getTFTestFeatures(testN, params);
                float[] testLabels = TrainingDataIO.getInstance().getTFTestingLabels(testN, params);

                Log.d("train", "feeding test samples and labels");

                // Copy the test data into TensorFlow.
                Trace.beginSection("feed");
                trainingInterface.feed(samplesName, testFeatures, testN, D);
                trainingInterface.feed(labelsName, testLabels, testN, K);
                Trace.endSection();

                Log.d("train", "test the accuracy");

                // Run the inference call.
                Trace.beginSection("test");
                trainingInterface.run(new String[]{testName}, new String[]{});
                Trace.endSection();

                Log.d("train", "fetching the results");

                // Copy the accuracy Tensor into the output array.
                float[] outputs = new float[1];
                Trace.beginSection("fetch");
                trainingInterface.fetch(testName, outputs);
                Trace.endSection();
                Log.d("train", (outputs[0] * 100) + "%");

                Trace.endSection(); // "beginTraining"
            }
        }

        double[] newWeights = new double[D * K]; // TODO: include the bias
        for (int j = 0; j < D * K; j++){
            newWeights[j] = (double)w[j];
        }
        return newWeights;
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
