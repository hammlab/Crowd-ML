package osu.crowd_ml.tests;

import android.content.Context;

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
public class BinaryTest implements ModelTest {

    public double accuracy(Context context, List<Double> weightVals, List<Integer> testLabels, List<double[]> testFeatures, int testN, int D, int K, int nh){

        int correct = 0;
        for(int i = 0; i < testN; i++){
            double[] X = testFeatures.get(i);
            Integer Y = testLabels.get(i);

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
}
