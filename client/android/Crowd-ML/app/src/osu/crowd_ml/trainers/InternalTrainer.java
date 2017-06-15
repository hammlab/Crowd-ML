package osu.crowd_ml.trainers;

import android.util.Log;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import osu.crowd_ml.BuildConfig;
import osu.crowd_ml.Parameters;
import osu.crowd_ml.TrainingDataIO;
import osu.crowd_ml.loss_functions.LossFunction;
import osu.crowd_ml.noise_distributions.Distribution;
import osu.crowd_ml.utils.ArrayUtils;

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

public class InternalTrainer implements Trainer {

    private static InternalTrainer instance = null;

    private List<Integer> order;
    private Parameters params;
    private List<Double> weights;
    private int t;
    private int length;

    private InternalTrainer(){}
    
    public static Trainer getInstance() {
        if (instance == null) {
            instance = new InternalTrainer();
        }
        return instance;
    }

    public List<Double> getNoisyGrad() {
        maintainSampleOrder(); // this line ensures that order is never null or empty
        List<Integer> oldOrder = new ArrayList<>(order);
        List<Double> noisyGrad = computeNoisyGrad();

        if (Thread.currentThread().isInterrupted()){
            order = oldOrder;
        }
        return noisyGrad;
    }

    /**
     * @param numIterations number of training steps to run.
     * @return The updated weight matrix.
     */
    @Override
    public List<Double> train(final int numIterations) {
        maintainSampleOrder(); // This line ensures order is never null or empty
        // Cache the old order list if we need to rollback changes.
        List<Integer> oldOrder = new ArrayList<>(order);
        for (int i = 0; i < numIterations; i++) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
            // Compute the gradient with random noise added
            List<Double> noisyGrad = computeNoisyGrad();

            // Return the updated weights
            weights = calcWeight(noisyGrad);

            Log.d("sendWeight", "local iter: " + (i + 1));
        }

        // Thread was stopped early
        if (Thread.currentThread().isInterrupted()) {
            order = oldOrder;
        }
        return weights;
    }

    @Override
    public Trainer setIter(int t) {
        this.t = t;
        return getInstance();
    }

    @Override
    public Trainer setWeights(List<Double> weights) {
        if (BuildConfig.DEBUG && weights.size() <= 0) throw new AssertionError();
        this.weights = weights;
        this.length = weights.size();
        return getInstance();
    }

    @Override
    public Trainer setParams(Parameters params) {
        this.params = params;
        return getInstance();
    }

    @Override
    public void destroy() {
        order = null;
        weights = null;
        params = null;
        instance = null;
    }

    private List<Double> calcWeight(List<Double> grad){

        double c = params.getC();
        double epsilson = params.getEps();
        String descentAlg = params.getDescentAlg();

        List<Double> newWeight = new ArrayList<>(length);
        double[] learningRate = null;
        if (descentAlg.equals("adagrad") || descentAlg.equals("rmsProp")){
            learningRate = new double[length];
        }

        for(int i = 0; i < length; i ++) {
            if (Thread.currentThread().isInterrupted()){
                break;
            }
            double deltaW;
            if (descentAlg.equals("constant")) {
                deltaW = c * grad.get(i);
            } else if (descentAlg.equals("simple")) {
                deltaW = (c / t) * grad.get(i);
            } else if (descentAlg.equals("sqrt")) {
                deltaW = (c / Math.sqrt(t)) * grad.get(i);
            } else if (descentAlg.equals("adagrad")) {
                double adagradRate = learningRate[i] + grad.get(i) * grad.get(i);
                learningRate[i] =  c / Math.sqrt(adagradRate + epsilson);
                deltaW = learningRate[i] * grad.get(i);
            } else if (descentAlg.equals("rmsProp")) {
                double rmsRate = 0.9 * learningRate[i] + 0.1 * grad.get(i) * grad.get(i);
                learningRate[i] = c / Math.sqrt(rmsRate + epsilson);
                deltaW = learningRate[i] * grad.get(i);
            } else {
                Log.e("InternalTrainer", "Invalid descent algorithm. Defaulting to \'simple\'.");
                deltaW = (c / t) * grad.get(i);
            }
            newWeight.add(i, weights.get(i) - deltaW);
        }

        return newWeight;
    }

    private List<Double> computeNoisyGrad(){
        // Init training sample batch
        int[] batchSamples = gatherBatchSamples();

        // TODO(tylermzeller) this is a bottleneck on physical devices. Buffered file I/O seems to
        // invoke the GC often.
        // Get training sample features.
        List<double[]> xBatch = TrainingDataIO.getInstance().readSamples(batchSamples, params);

        // Get training sample labels.
        List<Integer> yBatch = TrainingDataIO.getInstance().readLabels(batchSamples, params);

        // Compute average gradient vector
        List<Double> avgGrad = computeAverageGrad(xBatch, yBatch);

        // Init empty noisy gradient vector
        List<Double> noisyGrad = new ArrayList<>(length);

        // Add random noise probed from the client's noise distribution.
        Distribution dist = params.getNoiseDistribution();
        for (int i = 0; i < length; i++) {
            double avg = avgGrad.get(i);
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
            noisyGrad.add(i, dist.noise(avg, params.getNoiseScale()));
        }

        return noisyGrad;
    }

    private int[] gatherBatchSamples() {
        int batchSize = params.getClientBatchSize();
        int[] batchSamples = new int[batchSize];

        Random r = new Random(); // rng

        // Loop batchSize times
        for (int i = 0; i < batchSize; i++) {
            // Calling this method here ensures that the order list is never empty. When the order
            // list becomes empty, a new epoch of training occurs as the list is repopulated with
            // random int values in the range [0, N).
            maintainSampleOrder();

            // get a random index in the range [0, |order|) to query the order list.
            int q = r.nextInt(order.size());

            // Remove the value at index q and add it to the current batch of samples.
            batchSamples[i] = order.remove(q);
        }
        ArrayUtils.sort(batchSamples);
        return batchSamples;
    }

    private List<Double> computeAverageGrad(List<double[]> X, List<Integer> Y) {
        int batchSize = params.getClientBatchSize();
        LossFunction loss = params.getLossFunction();
        int D = params.getD();
        int K = params.getK();
        double L = params.getL();
        int nh = params.getNH();

        // Init average gradient vector
        List<Double> avgGrad = new ArrayList<>(Collections.nCopies(length, 0.0d));

        // For each sample, compute the gradient averaged over the whole batch.
        double[] x;
        List<Double> grad;
        for(int i = 0; i < batchSize; i++){
            // Periodically check if this thread has been interrupted. See the javadocs on
            // threading for best practices.
            if (Thread.currentThread().isInterrupted()){
                break;
            }
            x = X.get(i); // current sample feature
            int y = Y.get(i); // current label

            // Compute the gradient.
            grad = loss.gradient(weights, x, y, D, K, L, nh);

            // Add the current normalized gradient to the avg gradient vector.
            for(int j = 0; j < length; j++) {
                avgGrad.set(j, (avgGrad.get(j) + grad.get(j)) / batchSize);
            }
        }
        return avgGrad;
    }

    /**
     * Maintains the sample order list.
     *
     * The sample order list is queried for random indices of training samples without replacement
     * (until all values are removed, that is).
     *
     * If order is null or empty, the list will be filled with int values in the range [0, N), then
     * shuffled.
     */
    private void maintainSampleOrder() {
        // Step 1. Ensure the order list is initialized.
        if (order == null) {
            order = new ArrayList<>();
        }

        // Step 2. If the order list is empty, fill with values in the range [0, N).
        if (order.isEmpty()) {
            for (int i = 0; i < params.getN(); i++) //create sequential list of input sample #s
                order.add(i);

            // Step 3. Randomize order.
            Collections.shuffle(order);
        }
    }
}
