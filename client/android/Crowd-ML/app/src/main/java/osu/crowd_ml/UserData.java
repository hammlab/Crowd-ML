package osu.crowd_ml;

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

public class UserData {
    List<Double> gradients;
    boolean gradientProcessed;
    int gradIter;
    double weightIter;
    int paramIter;


    public UserData() {

    }

    public List<Double> getGradients() {
        return gradients;
    }

    public void setGradients(List<Double> gradientValues) {
        gradients = gradientValues;
    }

    public boolean getGradientProcessed(){
        return gradientProcessed;
    }

    public void setGradientProcessed(boolean checkVal) {
        gradientProcessed = checkVal;
    }

    public int getGradIter(){
        return gradIter;
    }

    public void setGradIter(int iter) {
        gradIter = iter;
    }

    public double getWeightIter(){
        return weightIter;
    }

    public void setWeightIter(double iter) {
        weightIter = iter;
    }

    public int getParamIter(){
        return paramIter;
    }

    public void setParamIter(int iter) {
        paramIter = iter;
    }


}