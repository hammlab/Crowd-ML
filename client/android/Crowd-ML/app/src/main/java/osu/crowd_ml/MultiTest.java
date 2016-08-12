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
public class MultiTest implements AccTest{

    public double accuracy(Context context, List<Double> weightVals, List<Integer> testLabels, List<double[]> testFeatures, int testN, int D, int K, int nh){

        int correct = 0;
        double dot;
        for(int i = 0; i < testN; i++){
            double[] X = testFeatures.get(i);
            Integer Y = testLabels.get(i);
            double[] classResults = new double[10];
            for(int h = 0; h < K; h++){
                dot = 0;
                for(int j = 0; j < D; j++){
                    dot += X[j]*weightVals.get(j + (h*D));}
                classResults[h] = dot;
            }
            int bestGuess = 0;
            for(int h = 0; h < K; h++){
                if(classResults[h]>classResults[bestGuess]){
                    bestGuess = h;}
            }

            if(bestGuess == Y){
                correct++;}
        }





        double accuracy = 100*correct/testN;
        return accuracy;

    }

}