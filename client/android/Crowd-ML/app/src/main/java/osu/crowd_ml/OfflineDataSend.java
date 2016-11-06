package osu.crowd_ml;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

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

public class OfflineDataSend extends AppCompatActivity {

    private int dataCount = 0;
    private boolean ready = false;

    private List<Integer> order;
    private List<Double> weightVals;

    private Distribution dist = new NoNoise();
    private int K = 10;
    private LossFunction loss = new Softmax();
    private String trainLabelSource = "MNISTTrainLabels.dat";
    private String trainFeatureSource = "MNISTTrainImages.dat";
    private String testLabelSource = "MNISTTestLabels.dat";
    private String testFeatureSource = "MNISTTestImages.dat";
    private int D = 784;
    private int N = 60000;
    private int testN = 1000;
    private int batchSize = 50;
    private double noiseScale = 1;
    private double L = 1e-6;
    private int nh = 75;
    private double c = 10;
    private double eps = 0.00000001;
    private String descentAlg = "sqrt";
    private int t = 1;
    private List<Double> learningRateDenom = new ArrayList<Double>();;

    private List<double[]> xBatch = new ArrayList<double[]>();
    private List<Integer> yBatch = new ArrayList<Integer>();
    List<double[]> testFeatures = new ArrayList<double[]>(testN);
    List<Integer> testLabels = new ArrayList<Integer>(testN);

    private int length;
    private AccTest test;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_offline);
    }

    @Override
    protected void onStart() {
        super.onStart();
        final TextView message = (TextView) findViewById(R.id.messageDisplay);
        if (loss.lossType().equals("binary")) {
            length = D;
            test = new BinaryTest();
        } else if (loss.lossType().equals("multi")) {
            length = D * K;
            test = new MultiTest();
        } else if (loss.lossType().equals("NN")) {
            length = D * nh + nh + nh * nh + nh + nh * K + K;
            test = new NNTest();
        }
        if (loss.lossType().equals("binary") && K > 2) {
            message.setText("Binary classifier used on non-binary data");
            dataCount = -1;
        }

        for(int i = 0; i < length; i ++){
            learningRateDenom.add(0.0);
        }

        weightVals = new ArrayList<Double>(length);
        for (int i = 0; i < length; i++) {
            weightVals.add(Math.random() - 0.5);
        }

        List<Integer> allTestSamples = new ArrayList<Integer>(testN);
        for(int i = 0; i < testN; i++){
            allTestSamples.add(i);
        }
        testFeatures = readSample(allTestSamples);
        testLabels = readType(allTestSamples);

        ready = true;
        message.setText("Waiting For Data");


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
                if (dataCount > N / batchSize) {
                    message.setText("Input count too high");
                }


                if (ready && dataCount > 0 && dataCount <= N / (batchSize)) {
                    message.setText("Sending Data");
                    ready = false;

                    List<Double> newWeight = new ArrayList<Double>(length);
                    while (dataCount > 0) {
                        newWeight = internalWeightCalc(weightVals, t);
                        t++;
                        weightVals = newWeight;
                        dataCount--;
                    }
                    System.out.println("new weight " + newWeight);
                    Double acc = test.accuracy(OfflineDataSend.this, weightVals, testLabels, testFeatures, testN, D, K, nh);
                    int end = Math.min(acc.toString().length(),6);
                    String results = "Accuracy: "+acc.toString().substring(0, end);
                    message.setText(results);

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




    public List<Double> internalWeightCalc(List<Double> weights, int weightIter){

        List<Integer> batchSamples = new ArrayList<Integer>();
        List<Double> currentWeights = weights;
        int batchSlot = 0;
        while(dataCount > 0 && batchSlot < batchSize) {
            batchSamples.add(order.get((batchSize*(dataCount-1) + batchSlot)));
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
        if(descentAlg.equals("adagrad")){
            for(int j = 0; j < length; j++){
                double learningRate = learningRateDenom.get(j) + noisyGrad.get(j)*noisyGrad.get(j);
                learningRateDenom.set(j, learningRate);
            }
        }
        else if (descentAlg.equals("rmsProp")){
            for(int j = 0; j < length; j++){
                double learningRate = 0.9*learningRateDenom.get(j) + 0.1*noisyGrad.get(j)*noisyGrad.get(j);
                learningRateDenom.set(j, learningRate);
            }
        }
        List<Double> newWeight = server.calcWeight(currentWeights, learningRateDenom, noisyGrad, weightIter, descentAlg, c, eps);

        return newWeight;

    }

    public List<double[]> readSample(List<Integer> sampleBatch){
        final TextView message = (TextView) findViewById(R.id.messageDisplay);
        List<double[]> xBatch = new ArrayList<double[]>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(trainFeatureSource)));
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
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(trainLabelSource)));
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
