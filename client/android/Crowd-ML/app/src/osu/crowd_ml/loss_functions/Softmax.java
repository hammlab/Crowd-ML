package osu.crowd_ml.loss_functions;

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

public class Softmax implements LossFunction {

    public String lossType() {
        return "multi";
    }

    public List<Double> gradient(List<Double> weights, float[] X, int Y, int D, int K, double L, int nh){

        List<Double> grad = new ArrayList<>(D*K);
        for(int i = 0; i < D*K; i++){
            grad.add(i, 0.0);
        }
        //Weights are read and gradients stored in column-major vectorization form for [W0,W1,W2,...,Wk]

        //dotMax used to prevent overflow
        double dot = 0;
        double dotMax = 0;
        for(int i = 0; i < K; i++){
            //dot product w_i·x
            dot = 0;
            for(int j = 0; j < D; j++){
                dot += X[j] * weights.get(j + (D*i));
            }

            if(dot > dotMax){
                dotMax = dot;
            }
        }

        //denom = Σ(i:k) exp(Θ_i · X)
        double denom = 0;
        for(int i = 0; i < K; i++){
            //dot product w_i·x
            dot = 0;
            for(int j = 0; j < D; j++){
                dot += X[j] * weights.get(j + (D*i));
            }

            denom += Math.exp(dot - dotMax);
        }

        //regularization constants
        double[] regular = new double[D * K];
        for(int i = 0; i < D * K; i++){
            regular[i] = 2 * weights.get(i) * L;
        }

        double prob;

        for(int i = 0; i < K; i++) {

            //prob_i = exp(Θ_i · X)/denom
            dot = 0;
            for (int j = 0; j < D; j++) {
                dot += X[j] * weights.get(j + (D * i));
            }
            prob = Math.exp(dot - dotMax) / denom;
            int match = 0;
            if(i == Y){
                match = 1;
            }
            //∇_0_i = -X(1{i = y} - prob_i)
            for (int j = 0; j < D; j++) {
                grad.set(j + (D * i), -1 * X[j] * (match - prob));
            }
        }

        //apply regularization
        for(int i = 0; i < D * K; i++){
            grad.set(i, grad.get(i)+ regular[i]);
        }

        return grad;
    }
}
