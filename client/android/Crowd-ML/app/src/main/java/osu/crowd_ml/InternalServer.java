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

public class InternalServer {

    public List<Double> calcWeight(List<Double> oldWeights, List<Double> learningRateDenom, List<Double> grad, float t, String descentAlg, double c, double eps){

        List<Double> newWeight = new ArrayList<Double>(oldWeights.size());

        if(descentAlg.equals("constant")){
            for(int i = 0; i < oldWeights.size(); i ++){
                newWeight.add(oldWeights.get(i) - (c)*grad.get(i));
            }
        }

        if(descentAlg.equals("simple")){
            for(int i = 0; i < oldWeights.size(); i ++){
                newWeight.add(oldWeights.get(i) - (c/t)*grad.get(i));
            }
        }

        if(descentAlg.equals("sqrt")){
            for(int i = 0; i < oldWeights.size(); i ++){
                newWeight.add(oldWeights.get(i) - (c/Math.sqrt(t))*grad.get(i));
            }
        }

        if(descentAlg.equals("adagrad")){
            for(int i = 0; i < oldWeights.size(); i ++){
                newWeight.add(oldWeights.get(i) - (c/Math.sqrt(learningRateDenom.get(i) + eps))*grad.get(i));
            }
        }

        return newWeight;

    }
}
