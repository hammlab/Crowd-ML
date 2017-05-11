package osu.crowd_ml.utils;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import osu.crowd_ml.Parameters;
import osu.crowd_ml.TrainingWeights;
import osu.crowd_ml.loss_functions.LossFunction;
import osu.crowd_ml.noise_distributions.Distribution;

public class DataComputer {


    // Bebugging
    private boolean VERBOSE_DEBUG = true;

    private List<Integer> order;
    private int paramIter;
    private Distribution dist;
    private int K;
    private LossFunction loss;
    private String labelSource;
    private String featureSource;
    private int D;
    private int N;
    private int batchSize;
    private double noiseScale;
    private double L;
    private int nh;
    private int localUpdateNum;
    private double c;
    private double eps;
    private String descentAlg;
    private int maxIter;
    private volatile int t;
    private List<Double> learningRate;
    private List<Double> weights;
    private int length;

    // Context used to get files
    private Context context;

    public DataComputer(Context context) {
        // Get Context
        this.context = context;
    }

    public void setParameters(Parameters params) {
        paramIter = params.getParamIter();
        dist = params.getNoiseDistribution();
        K = params.getK();
        loss = params.getLossFunction();
        labelSource = params.getLabelSource();
        featureSource = params.getFeatureSource();
        D = params.getD();
        N = params.getN();
        batchSize = params.getClientBatchSize();
        noiseScale = params.getNoiseScale();
        L = params.getL();
        nh = params.getNH();
        localUpdateNum = params.getLocalUpdateNum();
        c = params.getC();
        eps = params.getEps();
        descentAlg = params.getDescentAlg();
        maxIter = params.getMaxIter();

        // Call to setup the length
        loss.setLength(params);
        length = loss.getLength();

        // Must call setLength() before these lines
        if (descentAlg.equals("adagrad") || descentAlg.equals("rmsProp"))
            learningRate = new ArrayList<>(Collections.nCopies(length, 0.0d));

        // Must clear the sample order list of previous contents.
        order = new ArrayList<>();
    }

    public void setWeights(TrainingWeights weightVals) {
        weights = weightVals.getWeights().get(0);
        t = weightVals.getIteration();
    }


    /**
     * Returns the noisy gradients.
     *
     * @return noiseyGradients as a List<Double>
     */
    public List<Double> getNoisyGradients() {
        return computeNoisyGrad();
    }

    public List<Double> getWeights() {
        // Cache old parameters, in case we need to rollback changes
        int oldT = t;
        List<Integer> oldOrder = new ArrayList<>(order);

        // Get current weights.
        //List<Double> weights = weightVals.getWeights().get(0);
        // Calc new weights using local update num.
        for (int i = 0; i < localUpdateNum; i++) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
            weights = internalWeightCalc();
            //t++;
            if (VERBOSE_DEBUG)
                Log.d("sendWeight", "local iter: " + i + 1);
        }

        return weights;
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
    public void maintainSampleOrder(){

        // Step 1. Ensure the order list is initialized.
        if (order == null){
            order = new ArrayList<>();
        }

        // Step 2. If the order list is empty, fill with values in the range [0, N).
        if (order.isEmpty()){
            for (int i = 0; i < N; i++) //create sequential list of input sample #s
                order.add(i);

            // Step 3. Randomize order.
            Collections.shuffle(order);
        }
    }


    private List<Integer> gatherBatchSamples(){
        List<Integer> batchSamples = new ArrayList<>();

        int i = 0; // counter
        Random r = new Random(); // rng

        // Loop batchSize times
        while(i < batchSize) {
            // Calling this method here ensures that the order list is never empty. When the order
            // list becomes empty, a new epoch of training occurs as the list is repopulated with
            // random int values in the range [0, N).
            maintainSampleOrder();

            // get a random index in the range [0, |order|) to query the order list.
            int q = r.nextInt(order.size());

            // Remove the value at index q and add it to the current batch of samples.
            batchSamples.add(order.remove(q));
            i++; // counter
        }
        return batchSamples;
    }

    private List<Double> computeAverageGrad(List<double[]> X, List<Integer> Y, List<Double> weights){
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

//            for(int j = 0; j < length; j++) {
//                avgGrad.set(j, avgGrad.get(j) / batchSize);
//            }

        }

        return avgGrad;
    }

    /**
     * Compute the gradient with random noise added.
     *
     * @return noiseyGradients as a List<Double>
     */
    private List<Double> computeNoisyGrad(){
        // Init training sample batch
        List<Integer> batchSamples = gatherBatchSamples();

        // TODO(tylermzeller) this is a bottleneck on physical devices. Buffered file I/O seems to
        // invoke the GC often.
        // Get training sample features.
        List<double[]> xBatch = readSamples(batchSamples);

        // Get training sample labels.
        List<Integer> yBatch = readLabels(batchSamples);

        // Compute average gradient vector
        List<Double> avgGrad = computeAverageGrad(xBatch, yBatch, weights);

        // Init empty noisy gradient vector
        List<Double> noisyGrad = new ArrayList<>(length);

        // Add random noise probed from the client's noise distribution.
        for (double avg : avgGrad) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
            // TODO(davidsoller): This changes the size of the ArrayList after we initialized it with a certain size
            // Change to .add(index, element) as per https://developer.android.com/reference/java/util/ArrayList.html
            noisyGrad.add(dist.noise(avg, noiseScale));
        }

        return noisyGrad;
    }

    private List<Double> internalWeightCalc(){
        // Compute the gradient with random noise added
        List<Double> noisyGrad = computeNoisyGrad();

        // Periodically check if this thread has been interrupted. See the javadocs on
        // threading for best practices.
        if (Thread.currentThread().isInterrupted()){
            return noisyGrad;
        }

        // Return the updated weights
        return InternalServer.calcWeight(weights, noisyGrad, learningRate, t, descentAlg, c, eps);
    }

    public List<double[]> readSamples(List<Integer> sampleBatch){
        List<double[]> xBatch = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(featureSource)));
            String line;
            int counter = 0;
            double[] sampleFeatures = new double[D];
            String[] features;
            int max = Collections.max(sampleBatch);
            while ((line = br.readLine()) != null && counter <= max){
                // Periodically check if this thread has been interrupted. See the javadocs on
                // threading for best practices.
                if (Thread.currentThread().isInterrupted()) {
                    break;
                }

                if(sampleBatch.contains(counter)){

                    features = line.split(",| ");

                    // TODO: why is this list necessary?
                    //List<String> featureList = new ArrayList<>(Arrays.asList(features));
                    //featureList.removeAll(Arrays.asList(""));
                    for(int i = 0; i < D; i++) {
                        sampleFeatures[i] = Double.parseDouble(features[i]);
                        //sampleFeatures[i] = Double.parseDouble(features[i]);
                    }
                    xBatch.add(sampleFeatures);
                }
                counter++;
            }
        } catch (IOException e) {
            // TODO(tylermzeller) probably a better way to handle this.
            e.printStackTrace();
        }
        return xBatch;
    }

    public List<Integer> readLabels(List<Integer> sampleBatch){
        List<Integer> yBatch = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(labelSource)));
            String line;
            int counter = 0;
            while ((line = br.readLine()) != null && counter <= Collections.max(sampleBatch)){
                // Periodically check if this thread has been interrupted. See the javadocs on
                // threading for best practices.
                if (Thread.currentThread().isInterrupted()) {
                    break;
                }

                if(sampleBatch.contains(counter)){
                    line = line.trim();
                    int sampleLabel = Integer.parseInt(line);
                    if(sampleLabel == 0 && loss.lossType().equals("binary")){
                        sampleLabel = -1;
                    }
                    yBatch.add(sampleLabel);
                }
                counter++;
            }
        } catch (IOException e) {
            // TODO(tylermzeller) probably a better way to handle this.
            e.printStackTrace();
        }
        return yBatch;
    }

}
