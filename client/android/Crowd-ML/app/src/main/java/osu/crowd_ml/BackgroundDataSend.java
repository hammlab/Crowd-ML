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
import android.os.Bundle;
import android.os.IBinder;
import android.support.annotation.Nullable;
import android.util.Log;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

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
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class BackgroundDataSend extends Service {

    private static final int DEFAULT_BATCH_SIZE = 1;

    final static FirebaseDatabase database = FirebaseDatabase.getInstance();
    final static DatabaseReference ref = database.getReference();
    final static DatabaseReference weights = ref.child("trainingWeights");
    final static DatabaseReference parameters = ref.child("parameters");
    DatabaseReference userValues;

    private UserData userData;
    private String UID;
    private List<Integer> order;
    private TrainingWeights weightVals;
    private Parameters params;
    private UserData userCheck;
    private int gradientIteration = 0;
    private int dataCount = 0;
    private boolean ready = false;
    private boolean autosend = true;
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
    private List<Double> learningRateDenom;

    private List<double[]> xBatch = new ArrayList<>();
    private List<Integer> yBatch = new ArrayList<>();

    private int length;

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        //Toast.makeText(this, "Service created.", Toast.LENGTH_SHORT).show();
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

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        super.onStartCommand(intent, flags, startId);

        // Step 1. Add parameters listener.
        paramListener = parameters.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                onParameterDataChange(dataSnapshot);
            }

            @Override
            public void onCancelled(DatabaseError error) {
                // Parameter listener error
                Log.d("BackgroundDataSend", "Parameter listener error");
            }
        });

        // Step 2. Add weight listener.
        weightListener = weights.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                weightVals = dataSnapshot.getValue(TrainingWeights.class);
                Log.d("Weight size", weightVals.getWeights().get(0).size() + "");
            }

            @Override
            public void onCancelled(DatabaseError error) {
                // Weight event listener error
                //message.setText("Weight event listener error");
            }
        });

        // Step 3. Send the training data.
        sendTrainingData();

        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        parameters.removeEventListener(paramListener);
        weights.removeEventListener(weightListener);
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
        dataCount = 0;

        if(loss.lossType().equals("binary") && K > 2){
            // Error: Binary classifier used on non-binary data
            dataCount = -1;
        }

        // Must call setLength() before this line
        learningRateDenom = new ArrayList<>(Collections.nCopies(length, 0.0d));

        System.out.println("Learning rate denom: " + learningRateDenom.size());

        // Step 3. TODO: why do we add this here? This adds a new listener every time the parameters are update
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

    // TODO: Consider removing
    private void fill(List<Double> l, double val){
        for(int i = 0; i < length; i++){
            l.add(val);
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
    }

    private void addUserListener(){
        if (userListener != null) {
            userValues.removeEventListener(userListener);
            userListener = null;
        }

        userListener = userValues.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                //internetServices(); wait for wifi
                userCheck = dataSnapshot.getValue(UserData.class);
                if(!init){
                    init = true;
                    initUser();
                }
                if (dataCount > 0 && userCheck.getGradientProcessed() && userCheck.getGradIter() == gradientIteration){
                    if (localUpdateNum == 0){
                        sendGradient();
                    } else if (localUpdateNum > 0){
                        ready = false;
                        userData = new UserData();
                        List<Double> weight = weightVals.getWeights().get(0);
                        userData.setParamIter(paramIter);
                        userData.setWeightIter(t);
                        for (int i = 0; i < localUpdateNum; i++) {
                            weight = internalWeightCalc(weight, t, i);
                            t++;
                        }
                        // System.out.println("new weight " + weight);
                        userData.setGradIter(++gradientIteration);
                        userData.setGradientProcessed(false);
                        userData.setGradients(weight);
                        userValues.setValue(userData);
                        dataCount--;
                        ready = true;
                    }
                }

                if (dataCount == 0) {
                    ready = true;

                    //Auto-send used for testing
                    if (autosend){
                        autosend = false;

                        order = new ArrayList<>();
                        for (int i = 0; i < N; i++) { //create sequential list of input sample #s
                            order.add(i);
                        }
                        Collections.shuffle(order); //randomize order
                        dataCount = maxIter;
                        System.out.println("maxIter " + dataCount);
                        if (dataCount > N / batchSize) {
                            dataCount = 0;
                        }

                        //message.setText("Sending Data");
                        ready = false;
                        //internetServices(); // TODO: Why wait here ... ?

                        // TODO: Why check dataCount here, if we check it above?
                        if (dataCount > 0 && dataCount <= N/batchSize && localUpdateNum == 0) {
                            // Sending data
                            //message.setText("Sending Data");
                            ready = false; //TODO: isn't ready already false?
                            //internetServices(); // TODO: ...  and here as well?
                            sendGradient();
                        }

                        if (dataCount > 0  && localUpdateNum > 0) {
                            // Sending data
                            //message.setText("Sending Data");
                            ready = false;

                            userData = new UserData();
                            List<Double> weight = weightVals.getWeights().get(0);
                            Log.d("Weight size 2", weight.size() + "");
                            userData.setParamIter(paramIter);
                            userData.setWeightIter(t);

                            // Calc new weights using local update num
                            for (int i = 0; i < localUpdateNum; i++) {
                                weight = internalWeightCalc(weight, t, i);
                                t++;
                            }

                            userData.setGradIter(++gradientIteration);
                            userData.setGradientProcessed(false);
                            userData.setGradients(weight);
                            userValues.setValue(userData);
                            dataCount--;

                            ready = true;

                        }
                    }
                }
            }

            @Override
            public void onCancelled(DatabaseError firebaseError) {
                // Error
            }
        });
    }

    //allows for newly created users to initialize values
    private void initUser(){
        userData = new UserData();
        userData.setParamIter(paramIter);
        double weightIter = weightVals.getWeights().get(1).get(0);// TODO Why not getIteration()?
        userData.setWeightIter(weightIter);
        userData.setGradientProcessed(false);
        List<Double> initGrad = weightVals.getWeights().get(0);
        userData.setGradients(initGrad);
        userData.setGradIter(gradientIteration);
        userValues.setValue(userData);
    }

    private void sendTrainingData(){
        order = new ArrayList<>();
        for (int i = 0; i < N; i++) { //create sequential list of input sample #s
            order.add(i);
        }
        Collections.shuffle(order); //randomize order

        // Sending data
        if (ready && dataCount > 0){
            if (dataCount <= N/batchSize && localUpdateNum == 0) {
                ready = false;
                // TODO
                //internetServices(); wait for wifi
                sendGradient();
            } else if (dataCount <= N/(batchSize*localUpdateNum) && localUpdateNum > 0) {
                ready = false;

                userData = new UserData();
                List<Double> weights = weightVals.getWeights().get(0);
                userData.setParamIter(paramIter);
                userData.setWeightIter(t);

                for (int i = 0; i < localUpdateNum; i++) {
                    weights = internalWeightCalc(weights, t, i);
                    t++;
                }

                userData.setGradIter(++gradientIteration);
                userData.setGradientProcessed(false);
                userData.setGradients(weights);
                userValues.setValue(userData);
                dataCount--;

                ready = true;
            }
        }
    }

    private void sendGradient(){
        userData = new UserData();
        userData.setParamIter(paramIter);
        double weightIter = weightVals.getWeights().get(1).get(0); //TODO Why not getIteration()?
        userData.setWeightIter(weightIter);

        List<Integer> batchSamples = new ArrayList<>();
        List<Double> currentWeights = weightVals.getWeights().get(0);
        int batchSlot = 0;
        while(dataCount > 0 && batchSlot < batchSize) {
            batchSamples.add(order.get((batchSize*(dataCount-1) + batchSlot)));
            batchSlot++;
        }

        dataCount--;
        xBatch = readSamples(batchSamples);
        yBatch = readLabels(batchSamples);
        List<Double> avgGrad = new ArrayList<>(length);
        for(int i = 0; i < length; i ++){
            avgGrad.add(0.0);
        }

        for(int i = 0; i < batchSize; i++){
            double[] X = xBatch.get(i);
            int Y = yBatch.get(i);
            List<Double> grad = loss.gradient(currentWeights, X, Y, D, K, L, nh);

            double sum;
            for(int j = 0; j < length; j++) {
                sum = avgGrad.get(j) + grad.get(j);
                avgGrad.set(j,sum);
            }
        }

        double total;
        for(int i = 0; i < length; i++) {
            total = avgGrad.get(i);
            avgGrad.set(i, total/batchSize);
        }

        List<Double> noisyGrad = new ArrayList<>(length);
        for (int j = 0; j < length; j++){
            noisyGrad.add(dist.noise(avgGrad.get(j), noiseScale));
        }

        System.out.println("sendGradient");
        userData.setGradientProcessed(false);
        userData.setGradients(noisyGrad);
        userData.setGradIter(++gradientIteration);
        userValues.setValue(userData);
        avgGrad.clear();
    }

    private List<Double> internalWeightCalc(List<Double> weights, float weightIter, int localUpdateIter){

        List<Integer> batchSamples = new ArrayList<>();
        int batchSlot = 0;
        while(dataCount > 0 && batchSlot < batchSize) {
            //batchSamples.add(order.get((batchSize*localUpdateNum*(dataCount-1) + batchSlot*(localUpdateIter+1))));
            batchSamples.add(order.get((batchSize*localUpdateNum*(dataCount-1) + batchSlot*(localUpdateIter+1))));
            batchSlot++;
        }

        xBatch = readSamples(batchSamples);
        yBatch = readLabels(batchSamples);
        List<Double> avgGrad = new ArrayList<>(length);
        for(int i = 0; i < length; i ++){
            avgGrad.add(0.0);
        }

        for(int i = 0; i < batchSize; i++){
            double[] X = xBatch.get(i);
            int Y = yBatch.get(i);
            Log.d("Weight size 3", weights.size() + "");
            List<Double> grad = loss.gradient(weights, X, Y, D, K, L, nh);

            double sum;
            for(int j = 0; j < length; j++) {
                sum = avgGrad.get(j) + grad.get(j);
                avgGrad.set(j,sum); // TODO: Consider dividing by batchSize here
            }
        }

        // TODO: This loop may not be necessary. May just have to divide by batchSize above
        double sum;
        for(int i = 0; i < length; i++) {
            sum = avgGrad.get(i);
            avgGrad.set(i, sum/batchSize);
        }

        List<Double> noisyGrad = new ArrayList<>(length);
        for (int j = 0; j < length; j++){
            noisyGrad.add(dist.noise(avgGrad.get(j), noiseScale));
        }

        if(descentAlg.equals("adagrad")){
            for(int j = 0; j < length; j++){
                double learningRate = learningRateDenom.get(j) + noisyGrad.get(j)*noisyGrad.get(j);
                learningRateDenom.set(j, learningRate);
            }
        }
        else if (descentAlg.equals("rmsProp")){
            for(int j = 0; j < length; j++){
                double learningRate = 0.9 * learningRateDenom.get(j) + 0.1 * noisyGrad.get(j)*noisyGrad.get(j);
                learningRateDenom.set(j, learningRate);
            }
        }

        return InternalServer.calcWeight(weights, learningRateDenom, noisyGrad, weightIter, descentAlg, c, eps);
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
