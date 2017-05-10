package osu.crowd_ml;

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

import osu.crowd_ml.loss_functions.Hinge;
import osu.crowd_ml.loss_functions.LogReg;
import osu.crowd_ml.loss_functions.LossFunction;
import osu.crowd_ml.loss_functions.MultiSVM;
import osu.crowd_ml.loss_functions.Softmax;
import osu.crowd_ml.loss_functions.SoftmaxNN;
import osu.crowd_ml.noise_distributions.Distribution;
import osu.crowd_ml.noise_distributions.Gaussian;
import osu.crowd_ml.noise_distributions.Laplace;
import osu.crowd_ml.noise_distributions.NoNoise;

public class Parameters {
    private int paramIter;
    private double L;
    private int K;
    private double noiseScale;
    private Distribution noiseDistribution;
    private LossFunction lossFunction;
    private String labelSource;
    private String featureSource;
    private int D;
    private int N;
    private int nh;
    private int localUpdateNum;
    private int clientBatchSize;
    private int maxIter;
    private double c;
    private double eps;
    private String descentAlg;


    public Parameters() {

    }
    public int getParamIter(){
        return paramIter;
    }
    public void setParamIter(int iter){
        paramIter = iter;
    }

    public double getL(){
        return L;
    }
    public void setL(double regConst){
        L = regConst;
    }

    public double getNoiseScale(){
       return noiseScale;
    }
    public void setNoiseScale(double var){
        noiseScale = var;
    }

    public Distribution getNoiseDistribution(){
        return noiseDistribution;
    }
    public void setNoiseDistribution(String noiseDistributionName){
        switch (noiseDistributionName) {
            case "NoNoise":
                noiseDistribution = new NoNoise();
                break;
            case "Gaussian":
                noiseDistribution = new Gaussian();
                break;
            case "Laplace":
                noiseDistribution = new Laplace();
                break;
            default:
                noiseDistribution = new NoNoise();
                break;
        }
    }

    public int getK(){
        return K;
    }
    public void setK(int kVal){
        K = kVal;
    }

    public LossFunction getLossFunction(){
        return lossFunction;
    }
    public void setLossFunction(String lossFunctionName){
        switch (lossFunctionName) {
            case "LogReg":
                lossFunction = new LogReg();
                break;
            case "Hinge":
                lossFunction = new Hinge();
                break;
            case "Softmax":
                lossFunction = new Softmax();
                break;
            case "MultiSVM":
                lossFunction = new MultiSVM();
                break;
            case "SoftmaxNN":
                lossFunction = new SoftmaxNN();
                break;
            default:
                lossFunction = new LogReg();
                break;
        }
    }

    public String getLabelSource(){
        return labelSource;
    }
    public void setLabelSource(String label){
        labelSource = label;
    }

    public String getFeatureSource(){
        return featureSource;
    }
    public void setFeatureSource(String features){
        featureSource = features;
    }

    public int getD(){
        return D;
    }
    public void setD(int dVal){
        D = dVal;
    }

    public int getN(){
        return N;
    }
    public void setN(int nVal){
        N = nVal;
    }

    public int getNH(){
        return nh;
    }
    public void setNH(int nhVar){
        nh = nhVar;
    }

    public int getClientBatchSize(){
        return clientBatchSize;
    }
    public void setClientBatchSize(int bSize){
        clientBatchSize = bSize;
    }

    public int getLocalUpdateNum() { return localUpdateNum;}
    public void setLocalUpdateNum(int size){ localUpdateNum = size;}

    public double getC() {return c;}
    public void setC ( double cVal){ c = cVal;}

    public double getEps() {return eps;}
    public void setEps(double epsVal){eps = epsVal;}

    public String getDescentAlg() {
        return descentAlg;
    }

    public void setDescentAlg(String descentAlgVal) {descentAlg = descentAlgVal;}

    public int getMaxIter() {return maxIter;}

    public void setMaxIter(int max){maxIter = max;}
}
