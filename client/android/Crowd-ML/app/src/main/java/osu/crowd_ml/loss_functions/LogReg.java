package osu.crowd_ml.loss_functions;

import java.util.ArrayList;
import java.util.List;

import osu.crowd_ml.Parameters;

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

public class LogReg implements LossFunction {

    protected int length;

    public String lossFunctionName() {
        return "LogReg";
    }

    public String lossType() {
        return "binary";
    }

    public int getLength() {
        return length;
    }
    public void setLength(Parameters params) {
        length = params.getD();
    }

    public List<Double> gradient(List<Double> weights, double[] X, int Y, int D, int K, double L, int nh){
        List<Double> grad = new ArrayList<Double>(D);
        for (int i = 0; i < D; i++) {
            grad.add(0.0);
        }

        //dot product w*x
        double dot = 0;
        for (int j = 0; j < D; j++) {
            dot += weights.get(j) * X[j];
        }

        //-yi exp(-yi w·xi) / (1+ exp(-yi w·xi))
        double gradMultiplier = (-Y) * Math.exp((-Y) * dot) / (1 + (Math.exp((-Y) * dot)));

        double[] regular = new double[D];
        for(int i = 0; i < D; i++){
            regular[i] = 2*weights.get(i)*L;
        }

        //-yi xi exp(-yi w·xi) / (1+ exp(-yi w·xi))
        for (int j = 0; j < D; j++) {
            grad.set(j,X[j] * gradMultiplier + regular[j]);
        }

        return grad;
    }
}
