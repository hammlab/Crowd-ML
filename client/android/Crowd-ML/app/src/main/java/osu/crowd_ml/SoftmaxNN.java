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

public class SoftmaxNN implements LossFunction {

    public String lossType() {
        return "NN";
    }

    public List<Double> gradient(List<Double> weights, double[] X, int Y, int D, int K, double L, int nh){
        int length = D*nh + nh + nh*nh + nh + nh*K + K;
        List<Double> grad = new ArrayList<Double>(length);

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
            W01.add(weights.get(count));
            count++;
        }
        end = count + nh;
        while(count < end){
            b1.add(weights.get(count));
            count++;
        }
        end = count + nh*nh;
        while(count < end){
            W12.add(weights.get(count));
            count++;
        }
        end = count + nh;
        while(count < end){
            b2.add(weights.get(count));
            count++;
        }
        end = count + nh*K;
        while(count < end){
            W23.add(weights.get(count));
            count++;
        }
        end = count + K;
        while(count < end){
            b3.add(weights.get(count));
            count++;
        }

        //Forward Pass

        double dot;
        List<Double> h1 = new ArrayList<Double>(nh);
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < D; j++){
                dot += X[j]*W01.get(i + j*(nh));
            }
            if(dot + b1.get(i) > 0) {
                h1.add(dot + b1.get(i));
            }
            else{
                h1.add(0.0);
            }
        }

        List<Double> h2 = new ArrayList<Double>(nh);
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += h1.get(j)*W12.get(i + j*(nh));
            }
            if(dot + b2.get(i) > 0) {
                h2.add(dot + b2.get(i));
            }
            else{
                h2.add(0.0);
            }
        }

        List<Double> scores = new ArrayList<Double>(K);
        for(int i = 0; i < K; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += h2.get(j)*W23.get(i + j*(K));
            }
            scores.add(dot + b3.get(i));
        }

        System.out.println("scores");
        System.out.println(scores);
        System.out.println("correct");
        System.out.println(Y);

        //dotMax used to prevent overflow
        double dotMax = 0;
        for(int i = 0; i < K; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += h2.get(j) * W23.get(j + (nh*i));
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
            for(int j = 0; j < nh; j++){
                dot += h2.get(j) * W23.get(j + (nh*i));
            }

            denom += Math.exp(dot - dotMax);
        }


        List<Double> probs = new ArrayList<Double>(K);

        for(int i = 0; i < K; i++) {

            //prob_i = exp(Θ_i · X)/denom
            dot = 0;
            for (int j = 0; j < nh; j++) {
                dot += h2.get(j) * W23.get(j + (nh * i));
            }

            probs.add(Math.exp(dot - dotMax) / denom);
        }

        //Backward Pass

        List<Double> dW01 = new ArrayList<Double>(D*nh);
        List<Double> db1 = new ArrayList<Double>(nh);
        List<Double> dh1 = new ArrayList<Double>(nh);
        List<Double> dW12 = new ArrayList<Double>(nh*nh);
        List<Double> db2 = new ArrayList<Double>(nh);
        List<Double> dh2 = new ArrayList<Double>(nh);
        List<Double> dW23 = new ArrayList<Double>(nh*K);
        List<Double> db3 = new ArrayList<Double>(K);

        for(int i = 0; i < nh; i++){
            for(int j = 0; j < K; j++){
                dW23.add(probs.get(j)*h2.get(i) + L*W23.get(i*K + j));
            }
        }
        for(int i = 0; i < K; i++){
            db3.add(probs.get(i));
        }
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < K; j++){
                dot += probs.get(j) * W23.get(j + i*(K));
            }
            if(dot > 0){
                dh2.add(dot);
            }
            else{
                dh2.add(0.0);
            }
        }

        for(int i = 0; i < nh; i++){
            for(int j = 0; j < nh; j++){
                dW12.add(dh2.get(j)*h1.get(i) + L*W12.get(i*nh + j));
            }
        }
        for(int i = 0; i < nh; i++){
            db2.add(dh2.get(i));
        }
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += dh2.get(j) * W12.get(j + i*(nh));
            }
            if(dot > 0){
                dh1.add(dot);
            }
            else{
                dh1.add(0.0);
            }
        }

        for(int i = 0; i < D; i++){
            for(int j = 0; j < nh; j++){
                dW01.add(dh1.get(j)*X[i] + L*W01.get(i*nh + j));
            }
        }
        for(int i = 0; i < nh; i++){
            db1.add(dh1.get(i));
        }

        grad.addAll(dW01);
        grad.addAll(db1);
        grad.addAll(dW12);
        grad.addAll(db2);
        grad.addAll(dW23);
        grad.addAll(db3);


        return grad;
    }




}
