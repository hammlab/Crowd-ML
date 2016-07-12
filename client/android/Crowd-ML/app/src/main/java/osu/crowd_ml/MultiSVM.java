package osu.crowd_ml;

import java.util.ArrayList;
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

public class MultiSVM implements LossFunction{

    public String lossType() {
        return "multi";
    }

    public List<Double> gradient(List<Double> weights, double[] X, int Y, int D, int K, double L, int nh){

        List<Double> grad = new ArrayList<Double>(D*K);
        for(int i = 0; i < D*K; i ++){
            grad.add(i,0.0);
        }

        double dot = 0;
        for(int j = 0; j < D; j++){
            dot += X[j] * weights.get(j + (D*Y));
        }
        double counterDot = 0;
        double maxDot = 0;
        for(int k = 0; k < K; k++) {
            if(k != Y) {
                for (int j = 0; j < D; j++) {
                    counterDot += X[j] * weights.get(j + (D * k));
                }
            }
            if(counterDot > maxDot){
                maxDot = counterDot;
            }
        }

        //regularization constants
        double[] regular = new double[D * K];
        for(int i = 0; i < D * K; i++){
            regular[i] = 2 * weights.get(i) * L;
        }

        for(int j = 0; j < D; j++){
            if(dot < 1+maxDot){
                grad.set(j + (D*Y), -1 * X[j]);
            }
            else{
                grad.set(j + (D*Y), 0.0);
            }
        }

        //apply regularization
        for(int i = 0; i < D * K; i++){
            grad.set(i, grad.get(i)+ regular[i]);
        }



        return grad;
    }
}
