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
public class NNTest implements AccTest{

    public double accuracy(Context context, List<Double> weightVals, List<Integer> testLabels, List<double[]> testFeatures, int testN, int D, int K, int nh){

        int correct = 0;
        for(int i = 0; i < testN; i++){
            double[] X = testFeatures.get(i);
            Integer Y = testLabels.get(i);

            int length = D*nh + nh + nh*nh + nh + nh*K + K;

            List<Double> W01 = new ArrayList<Double>(D*nh);
            List<Double> b1 = new ArrayList<Double>(nh);
            List<Double> W12 = new ArrayList<Double>(nh*nh);
            List<Double> b2 = new ArrayList<Double>(nh);
            List<Double> W23 = new ArrayList<Double>(nh*K);
            List<Double> b3 = new ArrayList<Double>(K);

            //Parse Parameters
            int count = 0;
            int end = count + D*nh;
            while(count < end){
                W01.add(weightVals.get(count));
                count++;
            }
            end = count + nh;
            while(count < end){
                b1.add(weightVals.get(count));
                count++;
            }
            end = count + nh*nh;
            while(count < end){
                W12.add(weightVals.get(count));
                count++;
            }
            end = count + nh;
            while(count < end){
                b2.add(weightVals.get(count));
                count++;
            }
            end = count + nh*K;
            while(count < end){
                W23.add(weightVals.get(count));
                count++;
            }
            end = count + K;
            while(count < end){
                b3.add(weightVals.get(count));
                count++;
            }

            //Forward Pass

            double dot;
            double sum;
            List<Double> h1 = new ArrayList<Double>(nh);
            for(int h = 0; h < nh; h++){
                dot = 0;
                for(int j = 0; j < D; j++){
                    dot += X[j]*W01.get(h + j*(nh));
                }
                sum = dot + b1.get(h);
                if(sum > 0) {
                    h1.add(sum);
                }
                else{
                    h1.add(0.0);
                }
            }

            List<Double> h2 = new ArrayList<Double>(nh);
            for(int h = 0; h < nh; h++){
                dot = 0;
                for(int j = 0; j < nh; j++){
                    dot += h1.get(j)*W12.get(h + j*(nh));
                }
                sum = dot + b2.get(h);
                if(sum > 0) {
                    h2.add(sum);
                }
                else{
                    h2.add(0.0);
                }
            }

            List<Double> scores = new ArrayList<Double>(K);
            for(int h = 0; h < K; h++){
                dot = 0;
                for(int j = 0; j < nh; j++){
                    dot += h2.get(j)*W23.get(h + j*(K));
                }
                sum = dot + b3.get(h);
                scores.add(sum);
            }
            int bestGuess = 0;
            for(int h = 0; h < K; h++){
                if(scores.get(h)>scores.get(bestGuess)){
                    bestGuess = h;}
            }

            if(bestGuess == Y){
                correct++;}

        }

        double accuracy = 100*correct/testN;
        return accuracy;

    }

}