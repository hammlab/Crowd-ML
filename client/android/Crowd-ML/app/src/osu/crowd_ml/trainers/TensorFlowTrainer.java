package osu.crowd_ml.trainers;

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

import osu.crowd_ml.CrowdMLApplication;
import osu.crowd_ml.Parameters;
import osu.crowd_ml.R;
import osu.crowd_ml.TrainingDataIO;

public class TensorFlowTrainer implements Trainer {

    final static private int stepsToTest = 10;
    final static private int testN = 1000; // TODO: hardcoded for MNIST 10 class
    private Parameters params;
    private List<Double> weights;
    private int t;
    private boolean first;

    private static TensorFlowTrainer instance = null;

    private TensorFlowTrainingInterface trainingInterface = null;

    private TensorFlowTrainer(){
        this.first = true;
    }
    
    public static Trainer getInstance() {
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
    public List<Double> train(final int numIterations) {
        int batchSize = params.getClientBatchSize();
        int N = params.getN();
        int K = params.getK();
        int D = params.getD();
        String initName  = CrowdMLApplication.getAppContext().getString(R.string.init_name_TF);
        String trainName = CrowdMLApplication.getAppContext().getString(R.string.train_name_TF);
        String testName  = CrowdMLApplication.getAppContext().getString(R.string.test_name_TF);
        String feedName  = CrowdMLApplication.getAppContext().getString(R.string.input_name_TF);
        String fetchName = CrowdMLApplication.getAppContext().getString(R.string.label_name_TF);
        String weightsIn = CrowdMLApplication.getAppContext().getString(R.string.weights_in_TF);
        String weightsOp = CrowdMLApplication.getAppContext().getString(R.string.weights_op_TF);

        float[] w = new float[D * K];
        for (int j = 0; j < weights.size(); j++) {
            w[j] = (float)(double)weights.get(j);
        }

        // Initialize the training interface if this is the first round of training
        trainingInterface = getTrainingInterface();

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("beginTraining");

        if (first) {
            Trace.beginSection("init_vars");
            first = false;
            trainingInterface.run(new String[]{}, new String[]{initName});
        }
        //trainingInterface.feed(initName, new float[0], 0);

        Trace.endSection();

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

            // Copy the training data into TensorFlow.
            Trace.beginSection("feed");
            trainingInterface.feed(feedName, trainFeatureBatch, batchSize, D);
            trainingInterface.feed(weightsIn, w, D, K);
            trainingInterface.feed(fetchName, trainLabelBatch, batchSize, K);
            Trace.endSection();

            // Run a single step of training
            Trace.beginSection("train");
            trainingInterface.run(new String[]{weightsOp}, new String[]{trainName});
            Trace.endSection();

            // Copy the weights Tensor into the weights array.
            Trace.beginSection("fetch");
            trainingInterface.fetch(weightsOp, w);
            Trace.endSection();

            Log.d("TFTrainingInterface", i + " iteration");

            if (i == 0 || (i+1) % stepsToTest == 0){
                float[] testFeatures = TrainingDataIO.getInstance().getTFTestFeatures(testN, params);
                float[] testLabels = TrainingDataIO.getInstance().getTFTestingLabels(testN, params);
                // Copy the test data into TensorFlow.
                Trace.beginSection("feed");
                trainingInterface.feed(feedName, testFeatures, testN, D);
                trainingInterface.feed(fetchName, testLabels, testN, K);
                Trace.endSection();

                // Run the inference call.
                Trace.beginSection("test");
                trainingInterface.run(new String[]{testName}, new String[]{});
                Trace.endSection();

                // Copy the accuracy Tensor back into the output array.
                float[] outputs = new float[1];
                Trace.beginSection("fetch");
                trainingInterface.fetch(testName, outputs);
                Trace.endSection();
                Log.d("TFTrainingAccuracy", (outputs[0] * 100) + "%");

                Trace.endSection(); // "beginTraining"
            }
        }

        List<Double> newWeights = new ArrayList<>(D * K);
        for (int j = 0; j < D*K; j++){
            newWeights.add(j, (double)w[j]);
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
    public Trainer setWeights(List<Double> weights) {
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
        trainingInterface.close();
        weights = null;
        params = null;
        instance = null;
    }
}
