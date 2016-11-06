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

public class Hinge implements LossFunction{

    public String lossType() {
        return "binary";
    }

    public List<Double> gradient(List<Double> weights, double[] X, int Y, int D, int K, double L, int nh){
        List<Double> test = new ArrayList<Double>();
        for(int j = 0;j < X.length; j++){
            test.add(X[j]);
        }
        System.out.println("function X");
        System.out.println(test);


        List<Double>  grad = new ArrayList<Double>(D);
        for (int i = 0; i < D; i++) {
            grad.add(0.0);
        }

        //dot product w*x
        double dot = 0;
        for (int j = 0; j < D; j++) {
            dot += weights.get(j) * X[j];
        }
        System.out.println("dot");
        System.out.println(dot);
        System.out.println("Y*dot");
        System.out.println(Y*dot);

        double[] regular = new double[D];
        for(int i = 0; i < D; i++){
            regular[i] = 2*weights.get(i)*L;
        }

        for(int i = 0; i < D; i++){
            if(Y*dot < 1){
                grad.set(i, (-1)*Y*X[i]);// + regular[i]);
            }
            else{
                grad.set(i, 0.0);// + regular[i]);
            }
        }
        System.out.println("loss function");
        System.out.println(grad);

        return grad;
    }
}
