package osu.crowd_ml;

import android.util.Log;

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

class InternalServer {

    static List<Double> calcWeight(List<Double> oldWeights, List<Double> grad, List<Double> learningRate, float t, String descentAlg, double c, double eps){

        List<Double> newWeight = new ArrayList<>(oldWeights.size());

        for(int i = 0; i < oldWeights.size(); i ++){
            double deltaW;
            if(descentAlg.equals("constant")){
                deltaW = c * grad.get(i);
            } else if(descentAlg.equals("simple")){
                deltaW = (c / t) * grad.get(i);
            } else if(descentAlg.equals("sqrt")){
                deltaW = (c / Math.sqrt(t)) * grad.get(i);
            } else if(descentAlg.equals("adagrad")){
                double adagradRate = learningRate.get(i) + grad.get(i) * grad.get(i);
                learningRate.set(i, c / Math.sqrt(adagradRate + eps));
                deltaW = learningRate.get(i) * grad.get(i);
            } else if(descentAlg.equals("rmsProp")) {
                double rmsRate = 0.9 * learningRate.get(i) + 0.1 * grad.get(i) * grad.get(i);
                learningRate.set(i, c / Math.sqrt(rmsRate + eps));
                deltaW = learningRate.get(i) * grad.get(i);
            } else {
                Log.e("InternalServer", "Invalid descent algorithm. Defaulting to \'simple\'.");
                deltaW = (c / t) * grad.get(i);
            }
            newWeight.add(oldWeights.get(i) - deltaW);
        }

        return newWeight;
    }
}
