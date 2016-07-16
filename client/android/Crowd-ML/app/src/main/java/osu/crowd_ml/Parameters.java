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

public class Parameters {
    private int paramIter;
    private double L;
    private int K;
    private double noiseScale;
    private String noiseDistribution;
    private String lossFunction;
    private String labelSource;
    private String featureSource;
    private int D;
    private int N;
    private int nh;
    private int localUpdateNum;
    private double c;
    private double eps;
    private String descentAlg;

    private int clientBatchSize;

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
        Distribution dist = new NoNoise();
        if(noiseDistribution.equals("NoNoise")){
            dist = new NoNoise();
        }
        else if(noiseDistribution.equals("Gaussian")){
            dist = new Gaussian();
        }
        else if(noiseDistribution.equals("Laplace")){
            dist = new Laplace();
        }
        return dist;
    }
    public void setNoiseDistribution(String dist){
        noiseDistribution = dist;
    }

    public int getK(){
        return K;
    }
    public void setK(int kVal){
        K = kVal;
    }

    public LossFunction getLossFunction(){
        LossFunction loss = new LogReg();
        if(lossFunction.equals("LogReg")){
            loss = new LogReg();
        }
        else if(lossFunction.equals("Hinge")){
            loss = new Hinge();
        }
        else if(lossFunction.equals("Softmax")){
            loss = new Softmax();
        }
        else if(lossFunction.equals("MultiSVM")){
            loss = new MultiSVM();
        }
        else if(lossFunction.equals("SoftmaxNN")){
            loss = new SoftmaxNN();
        }
        return loss;
    }
    public void setLossFunction(String loss){
        lossFunction = loss;
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

    public void setDescentAlg(String descentAlgVal) {
        descentAlg = descentAlgVal;
    }
}
