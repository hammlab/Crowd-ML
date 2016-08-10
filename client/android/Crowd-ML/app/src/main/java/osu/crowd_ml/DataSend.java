package osu.crowd_ml;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;


import com.firebase.client.Firebase;
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

public class DataSend extends AppCompatActivity {

    final static FirebaseDatabase database = FirebaseDatabase.getInstance();
    final static DatabaseReference ref = database.getReference();
    final static DatabaseReference weights = ref.child("trainingWeights");
    final static DatabaseReference parameters = ref.child("parameters");
    DatabaseReference userValues;


    private UserData userData;
    private String email;
    private String password;
    private String UID;
    private List<Integer> order;
    private TrainingWeights weightVals;
    private Parameters params;
    private UserData userCheck;
    private int gradientIteration = 0;
    private int dataCount = 0;
    private boolean ready = false;

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
    private int t = 1;

    private List<double[]> xBatch = new ArrayList<double[]>();
    private List<Integer> yBatch = new ArrayList<Integer>();

    private int length;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_data);

        Intent login = getIntent();
        Bundle b = login.getExtras();
        if (b != null) {
            email = (String) b.get("EMAIL");
            password = (String) b.get("PASSWORD");
            UID = (String) b.get("UID");
        }

        userValues = ref.child("users").child(UID);

        Firebase.setAndroidContext(this);

    }

    @Override
    protected void onStart() {
        super.onStart();
        weightVals = new TrainingWeights();
        userCheck = new UserData();
        params = new Parameters();
        final TextView message = (TextView) findViewById(R.id.messageDisplay);


        parameters.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                params = dataSnapshot.getValue(Parameters.class);
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


                dataCount = 0;

                length = D;
                if(loss.lossType().equals("multi")){
                    length = D*K;
                }
                if(loss.lossType().equals("NN")){
                    length = D*nh + nh + nh*nh + nh + nh*K + K;
                }
                if(loss.lossType().equals("binary") && K > 2){
                    message.setText("Binary classifier used on non-binary data");
                    dataCount = -1;
                }

                checkoutListener();
            }

            @Override
            public void onCancelled(DatabaseError error) {
                message.setText("Weight event listener error");
            }

        });




        weights.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                weightVals = dataSnapshot.getValue(TrainingWeights.class);
            }

            @Override
            public void onCancelled(DatabaseError error) {
                message.setText("Weight event listener error");
            }

        });

        Button mSendTrainingData = (Button) findViewById(R.id.sendTrainingData);
        mSendTrainingData.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                order = new ArrayList<>();
                for (int i = 0; i < N; i++) { //create sequential list of input sample #s
                    order.add(i);
                }
                Collections.shuffle(order); //randomize order
                EditText countField = (EditText) findViewById(R.id.inputCount);
                String numStr = countField.getText().toString();
                try {
                    dataCount = Integer.parseInt(numStr);
                } catch (NumberFormatException e) {
                    message.setText("Not a number");
                    dataCount = 0;
                }
                if(dataCount > N/batchSize){
                    message.setText("Input count too high");
                }

                if (ready && dataCount > 0 && dataCount <= N/batchSize && localUpdateNum == 0) {
                    message.setText("Sending Data");
                    ready = false;
                    sendGradient();
                }

                if (ready && dataCount > 0 && dataCount <= N/(batchSize*localUpdateNum) && localUpdateNum > 0) {
                    message.setText("Sending Data");
                    ready = false;

                    userData = new UserData();
                    List<Double> oldWeight = weightVals.getWeights();
                    List<Double> newWeight = new ArrayList<Double>(length);
                    userData.setParamIter(paramIter);
                    userData.setWeightIter(t);
                    System.out.println("old weight "+oldWeight);
                    for (int i = 0; i < localUpdateNum; i++) {
                        newWeight = internalWeightCalc(oldWeight, t, i);
                        t++;
                        oldWeight = newWeight;
                        System.out.println("weight update "+oldWeight);
                    }
                    System.out.println("new weight "+newWeight);
                    gradientIteration++;
                    userData.setGradIter(gradientIteration);
                    userData.setGradientProcessed(false);
                    userData.setGradients(newWeight);
                    userValues.setValue(userData);
                    dataCount--;

                    ready = true;

                }


            }
        });

        Button mCancel = (Button) findViewById(R.id.cancel);
        mCancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dataCount = 0;
                ready = true;
                message.setText("Waiting for data");
            }
        });



    }

    public void checkoutListener(){
        final TextView message = (TextView) findViewById(R.id.messageDisplay);
        userValues.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                userCheck = dataSnapshot.getValue(UserData.class);
                if (dataCount > 0 && userCheck.getGradientProcessed() && userCheck.getGradIter()== gradientIteration && localUpdateNum == 0) {
                    sendGradient();
                }
                else if(dataCount > 0 && userCheck.getGradientProcessed() && userCheck.getGradIter()== gradientIteration && localUpdateNum > 0){
                    ready = false;
                    userData = new UserData();
                    List<Double> oldWeight = weightVals.getWeights();
                    List<Double> newWeight = new ArrayList<Double>(length);
                    userData.setParamIter(paramIter);
                    userData.setWeightIter(t);
                    for (int i = 0; i < localUpdateNum; i++) {
                        newWeight = internalWeightCalc(oldWeight, t, i);
                        t++;
                        oldWeight = newWeight;
                    }
                    System.out.println("new weight "+newWeight);
                    gradientIteration++;
                    userData.setGradIter(gradientIteration);
                    userData.setGradientProcessed(false);
                    userData.setGradients(newWeight);
                    userValues.setValue(userData);
                    dataCount--;
                    ready = true;
                }
                if (dataCount == 0) {
                    ready = true;
                    message.setText("Waiting for data");
                }
            }

            @Override
            public void onCancelled(DatabaseError firebaseError) {

            }
        }

        );

    }

    public void sendGradient(){
        userData = new UserData();
        userData.setParamIter(paramIter);

        List<Integer> batchSamples = new ArrayList<Integer>();
        List<Double> currentWeights = weightVals.getWeights();
        userData.setWeightIter(weightVals.getCurrentIteration());
        int batchSlot = 0;
        while(dataCount > 0 && batchSlot < batchSize) {
            batchSamples.add(order.get((batchSize*(dataCount-1) + batchSlot)));
            batchSlot++;
        }
            dataCount--;
            xBatch = readSample(batchSamples);
            yBatch = readType(batchSamples);
        List<Double> avgGrad = new ArrayList<Double>(length);
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

        List<Double> noisyGrad = new ArrayList<Double>(length);
        for (int j = 0; j < length; j++){
            noisyGrad.add(dist.noise(avgGrad.get(j), noiseScale));
        }

        userData.setGradientProcessed(false);
        userData.setGradients(noisyGrad);
        gradientIteration++;
        userData.setGradIter(gradientIteration);
        userValues.setValue(userData);
        avgGrad.clear();
    }


    public List<Double> internalWeightCalc(List<Double> weights, int weightIter, int localUpdateIter){

        List<Integer> batchSamples = new ArrayList<Integer>();
        List<Double> currentWeights = weights;
        int batchSlot = 0;
        while(dataCount > 0 && batchSlot < batchSize) {
            batchSamples.add(order.get((batchSize*localUpdateNum*(dataCount-1) + batchSlot*(localUpdateIter+1))));
            batchSlot++;
        }
        xBatch = readSample(batchSamples);
        yBatch = readType(batchSamples);
        List<Double> avgGrad = new ArrayList<Double>(length);
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


        double sum;
        for(int i = 0; i < length; i++) {
            sum = avgGrad.get(i);
            avgGrad.set(i, sum/batchSize);
        }

        List<Double> noisyGrad = new ArrayList<Double>(length);
        for (int j = 0; j < length; j++){
            noisyGrad.add(dist.noise(avgGrad.get(j), noiseScale));
        }

        InternalServer server = new InternalServer();
        List<Double> newWeight = server.calcWeight(currentWeights, noisyGrad, weightIter, descentAlg, c, eps);

        return newWeight;

    }

    public List<double[]> readSample(List<Integer> sampleBatch){
        final TextView message = (TextView) findViewById(R.id.messageDisplay);
        List<double[]> xBatch = new ArrayList<double[]>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(featureSource)));
            String line = null;
            int counter = 0;
            while ((line = br.readLine()) != null && counter <= Collections.max(sampleBatch)){
                if(sampleBatch.contains(counter)){
                    double[] sampleFeatures = new double[D];
                    String[] features = line.split(",|\\ ");
                    List<String> featureList = new ArrayList<String>(Arrays.asList(features));
                    featureList.removeAll(Arrays.asList(""));
                    for(int i = 0;i < D; i++)
                    {
                        sampleFeatures[i] = Double.parseDouble(featureList.get(i));
                    }
                    xBatch.add(sampleFeatures);
                }

                counter++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            message.setText("Sample file not found");
            dataCount = -1;
        } catch (IOException e) {
            e.printStackTrace();
            message.setText("Sample IO exception");
        }

        return xBatch;

    }

    public List<Integer> readType(List<Integer> sampleBatch){
        final TextView message = (TextView) findViewById(R.id.messageDisplay);
        int sampleLabel = 0;
        List<Integer> yBatch = new ArrayList<Integer>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(labelSource)));
            String line = null;
            int counter = 0;
            while ((line = br.readLine()) != null && counter <= Collections.max(sampleBatch)){
                String cleanLine = line.trim();
                if(sampleBatch.contains(counter)){
                    sampleLabel = (int)Double.parseDouble(cleanLine);
                    if(sampleLabel == 0 && loss.lossType().equals("binary")){
                        sampleLabel = -1;
                    }
                    yBatch.add(sampleLabel);
                }
                counter++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            message.setText("Type file not found");
            dataCount = -1;
        } catch (IOException e) {
            e.printStackTrace();
            message.setText("Type IO exception");
        }


        return yBatch;
    }

}
