package osu.crowd_ml;

import android.content.Context;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
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
public class BinaryTest implements AccTest{

    public double accuracy(Context context, List<Double> weightVals, String testLabelSource, String testFeatureSource, int testN, int D, int K, int nh){

        int correct = 0;
        for(int i = 0; i < testN; i++){
            double[] X = readSample(i, testN, D, context, testFeatureSource);
            Integer Y = readType(i, testN, D, context, testFeatureSource);

            double dot = 0;
            for(int j = 0; j < D; j++){
                dot += X[j]*weightVals.get(j);}

            int predict = 0;
            if(dot > 0){
                predict = 1;}
            if(predict == Y){
                correct++;}



        }

        double accuracy = 100*correct/testN;
        return accuracy;

    }
        double[] readSample(int sample, int testN, int D, Context context, String testFeatureSource){
            double[] sampleFeatures = new double[D];
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(testFeatureSource)));
            String line = null;
            int counter = 0;
            boolean found = false;
            while ((line = br.readLine()) != null && counter <= testN && !found){
                if(sample == counter){

                    String[] features = line.split(",|\\ ");
                    List<String> featureList = new ArrayList<String>(Arrays.asList(features));
                    featureList.removeAll(Arrays.asList(""));
                    for(int i = 0;i < D; i++)
                    {
                        sampleFeatures[i] = Double.parseDouble(featureList.get(i));
                    }
                    found = true;
                }

                counter++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return sampleFeatures;

    }

    Integer readType(int sample, int testN, int D, Context context, String testLabelSource){
        int sampleLabel = 0;
        List<Integer> yBatch = new ArrayList<Integer>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(testLabelSource)));
            String line = null;
            int counter = 0;
            boolean found = false;
            while ((line = br.readLine()) != null && counter <= testN && !found){
                String cleanLine = line.trim();
                if(sample == counter){
                    sampleLabel = (int)Double.parseDouble(cleanLine);
                    if(sampleLabel == 0){
                        sampleLabel = -1;
                    }
                    found = true;
                }
                counter++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


        return sampleLabel;
    }
}
