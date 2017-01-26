package osu.crowd_ml;

/*
    Copyright 2016 Crowd-ML team


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

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.IBinder;
import android.support.annotation.Nullable;
import android.util.Log;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class BackgroundDataSend extends Service {

    private static final int DEFAULT_BATCH_SIZE = 1;

    final static FirebaseDatabase database = FirebaseDatabase.getInstance();
    final static DatabaseReference ref = database.getReference();
    final static DatabaseReference weights = ref.child("trainingWeights");
    final static DatabaseReference parameters = ref.child("parameters");
    DatabaseReference userValues;

    private String UID;
    private List<Integer> order;
    private TrainingWeights weightVals;
    private Parameters params;
    private UserData userCheck;
    private int gradientIteration = 0;
    private int dataCount = 0;
    private boolean init = false;

    private ValueEventListener userListener;
    private ValueEventListener paramListener;
    private ValueEventListener weightListener;

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
    private int t = 1;
    private List<Double> learningRate;

    private int length;

    @Nullable @Override public IBinder onBind(Intent intent) {
        return null;
    }

    @Override public void onCreate() {
        super.onCreate();
        // Step 1. Get shared preferences.
        SharedPreferences preferences = getSharedPreferences("UserPreferences", Context.MODE_PRIVATE);

        // Step 2. Extract necessary information
        UID = preferences.getString("uid", "");
        dataCount = preferences.getInt("batchSize", DEFAULT_BATCH_SIZE);

        // Step 3. Get database references.
        userValues = ref.child("users").child(UID);

        // Step 4. Initialize necessary data.
        weightVals = new TrainingWeights();
        userCheck = new UserData();
        params = new Parameters();
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        super.onStartCommand(intent, flags, startId);

        // Step 1. Add parameters listener.
        paramListener = parameters.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
                Log.d("onDataChange", "Got parameters");
                onParameterDataChange(dataSnapshot);
            }

            @Override public void onCancelled(DatabaseError error) {
                // Parameter listener error
                Log.d("BackgroundDataSend", "Parameter listener error");
            }
        });

        // Step 2. Add weight listener.
        weightListener = weights.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
                // TODO: Need some way to signal when weights and parameters are set for training to resume.
                Log.d("onDataChange", "Got weights");
                weightVals = dataSnapshot.getValue(TrainingWeights.class);
            }

            @Override public void onCancelled(DatabaseError error) {
                // Weight event listener error
                //message.setText("Weight event listener error");
                Log.d("BackgroundDataSend", "Weights listener error");
            }
        });

        return START_STICKY;
    }

    @Override public void onDestroy() {
        super.onDestroy();
        if (paramListener != null)
            parameters.removeEventListener(paramListener);

        if (weightListener != null)
            weights.removeEventListener(weightListener);

        if (userListener != null)
            userValues.removeEventListener(userListener);

        paramListener = null;
        userListener = null;
        weightListener = null;
    }

    private void onParameterDataChange(DataSnapshot dataSnapshot){
        // Step 1. Get all parameters from the data snapshot
        params = dataSnapshot.getValue(Parameters.class);

        // Step 2. Set parameters.
        setParameters();
        setLength();

        if(loss.lossType().equals("binary") && K > 2){
            // Error: Binary classifier used on non-binary data
            dataCount = -1;
        }

        // Must call setLength() before these lines
        learningRate = new ArrayList<>(Collections.nCopies(length, 0.0d));

        // Must clear the sample order list of previous contents.
        order = new ArrayList<>();

        // Calling this method after clearing the order list (the above line) will initialize
        // it to a list of size N consisting of int values in the range [0, N) in random order.
        maintainSampleOrder();

        // Step 3. TODO: why do we add this here? This adds a new listener every time the parameters are updated
        addUserListener();
    }

    private void setLength(){
        length = D;

        if(loss.lossType().equals("multi")){
            length = D*K;
        }
        if(loss.lossType().equals("NN")){
            length = D*nh + nh + nh*nh + nh + nh*K + K;
        }
    }

    private void setParameters(){
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

        // Added to stop infinite send loop
        dataCount = maxIter;
    }

    private void addUserListener(){
        if (userListener != null) {
            userValues.removeEventListener(userListener);
            userListener = null;
        }

        userListener = userValues.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
                Log.d("onDataChange", "Got user values.");
                //internetServices(); wait for wifi
                userCheck = dataSnapshot.getValue(UserData.class);
                if(!init){
                    init = true;
                    initUser();
                }

                if (dataCount > 0 && userCheck.getGradientProcessed() && userCheck.getGradIter() == gradientIteration) {
                    if (localUpdateNum == 0){
                        sendGradient();
                    } else if (localUpdateNum > 0){
                        sendWeight();
                    }
                }
            }

            @Override public void onCancelled(DatabaseError firebaseError) {
                // Error
            }
        });
    }

    //allows for newly created users to initialize values
    private void initUser() {
        List<Double> initGrad = weightVals.getWeights().get(0);
        sendUserValues(initGrad, false, gradientIteration, weightVals.getIteration(), paramIter);
    }

    /**
     * Maintains the sample order list.
     * The sample order list is queried for random indices of training samples without replacement
     * (until all values are removed, that is).
     *
     * If order is null, order will be initialized to an empty list. If order is empty, the list
     * will be filled with int values in the range [0, N), then shuffled.
     */
    private void maintainSampleOrder(){
        if (order == null){
            order = new ArrayList<>();
        }

        if (order.isEmpty()){
            for (int i = 0; i < N; i++) //create sequential list of input sample #s
                order.add(i);
            Collections.shuffle(order); //randomize order
        }
    }

    private void sendGradient(){
        // Get current weights.
        List<Double> weights = weightVals.getWeights().get(0);

        // Compute the gradient with random noise added.
        List<Double> noisyGrad = computeNoisyGrad(weights);

        Log.d("sendGradient", "Sending gradient.");
        // Send the gradient to the server.
        sendUserValues(noisyGrad, false, ++gradientIteration, weightVals.getIteration(), paramIter);

        // Decrease iteration
        dataCount--;
    }

    private void sendWeight(){
        // Get current weights.
        List<Double> weights = weightVals.getWeights().get(0);

        // Calc new weights using local update num.
        for (int i = 0; i < localUpdateNum; i++) {
            weights = internalWeightCalc(weights);
            t++;
        }

        // Send the gradient to the server.
        sendUserValues(weights, false, ++gradientIteration, t, paramIter);

        // Decrease iteration.
        dataCount--;
    }

    private void sendUserValues(List<Double> gradientsOrWeights, boolean gradientProcessed, int gradIter, int weightIter, int paramIter){
        userValues.setValue(
            new UserData(gradientsOrWeights, gradientProcessed, gradIter, weightIter, paramIter)
        );
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
        for(int i = 0; i < batchSize; i++){
            double[] x = X.get(i); // current sample feature
            int y = Y.get(i); // current label

            // Compute the gradient.
            List<Double> grad = loss.gradient(weights, x, y, D, K, L, nh);

            // Add the current normalized gradient to the avg gradient vector.
            for(int j = 0; j < length; j++)
                avgGrad.set(j, (avgGrad.get(j) + grad.get(j)) / batchSize);
        }

        return avgGrad;
    }

    private List<Double> computeNoisyGrad(List<Double> weights){
        // Init training sample batch
        List<Integer> batchSamples = gatherBatchSamples();//new ArrayList<>();

        // Get training sample features.
        List<double[]> xBatch = readSamples(batchSamples);

        // Get training sample labels.
        List<Integer> yBatch = readLabels(batchSamples);

        // Init gradient vector
        List<Double> avgGrad = computeAverageGrad(xBatch, yBatch, weights);

        // Init empty noisy gradient vector
        List<Double> noisyGrad = new ArrayList<>();

        // Add random noise probed from the client's noise distribution.
        for (double avg : avgGrad)
            noisyGrad.add(dist.noise(avg, noiseScale));

        return noisyGrad;
    }

    private List<Double> internalWeightCalc(List<Double> weights){
        // Compute the gradient with random noise added
        List<Double> noisyGrad = computeNoisyGrad(weights);

        // Set the learningRate if the descent alg is adagrad or rmsProp
//        if(descentAlg.equals("adagrad")){
//            for(int j = 0; j < length; j++) {
//                double learningRate = learningRate.get(j) + noisyGrad.get(j) * noisyGrad.get(j);
//                learningRate.set(j, c / Math.sqrt(learningRate + eps));
//            }
//        } else if (descentAlg.equals("rmsProp")){
//            for(int j = 0; j < length; j++) {
//                double learningRate = 0.9 * learningRate.get(j) + 0.1 * noisyGrad.get(j) * noisyGrad.get(j);
//                learningRate.set(j, c / Math.sqrt(learningRate + eps));
//            }
//        }

        // Return the updated weights
        return InternalServer.calcWeight(weights, noisyGrad, learningRate, t, descentAlg, c, eps);
    }

    public List<double[]> readSamples(List<Integer> sampleBatch){
        List<double[]> xBatch = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(featureSource)));
            String line;
            int counter = 0;
            while ((line = br.readLine()) != null && counter <= Collections.max(sampleBatch)){
                if(sampleBatch.contains(counter)){
                    double[] sampleFeatures = new double[D];

                    // TODO: redundant escape?
                    String[] features = line.split(",| ");

                    // TODO: why is this list necessary?
                    List<String> featureList = new ArrayList<>(Arrays.asList(features));
                    featureList.removeAll(Arrays.asList(""));
                    for(int i = 0; i < D; i++) {
                        sampleFeatures[i] = Double.parseDouble(featureList.get(i));
                        //sampleFeatures[i] = Double.parseDouble(features[i]);
                    }
                    xBatch.add(sampleFeatures);
                }
                counter++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            dataCount = -1;
        } catch (IOException e) {
            e.printStackTrace();
        }

        return xBatch;

    }

    public List<Integer> readLabels(List<Integer> sampleBatch){
        List<Integer> yBatch = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(labelSource)));
            String line;
            int counter = 0;
            while ((line = br.readLine()) != null && counter <= Collections.max(sampleBatch)){
                line = line.trim();
                if(sampleBatch.contains(counter)){
                    int sampleLabel = (int)Double.parseDouble(line);
                    if(sampleLabel == 0 && loss.lossType().equals("binary")){
                        sampleLabel = -1;
                    }
                    yBatch.add(sampleLabel);
                }
                counter++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            dataCount = -1;
        } catch (IOException e) {
            e.printStackTrace();
        }

        return yBatch;
    }
}
