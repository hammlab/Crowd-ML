package osu.crowd_ml;

import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import osu.crowd_ml.firebase.CrowdMLApplication;
import osu.crowd_ml.loss_functions.LossFunction;
import osu.crowd_ml.utils.ArrayUtils;

/*
Copyright 2017 Crowd-ML team


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

public class TrainingDataIO {

    public static synchronized TrainingDataIO getInstance() {
        if(instance == null) instance = new TrainingDataIO();
        return instance;
    }

    private static TrainingDataIO instance;

    private TrainingDataIO() {}

    /**
     * This method is a left over from the original CrowdML codebase. Consider replacing with
     * updated code.
     *
     * @param sampleBatch
     * @param params
     * @return
     */
    public List<double[]> readSamples(int[] sampleBatch, final Parameters params) {
        int D = params.getD();
        String featureSrc = params.getFeatureSource();
        List<double[]> xBatch = new ArrayList<>(sampleBatch.length);
        try {
            // TODO(tylermzeller): provide an AssetManager to this class for reading source files
            BufferedReader br = new BufferedReader(new InputStreamReader(
                    CrowdMLApplication.getAppContext().getAssets().open(featureSrc)));
            String line;
            int counter = 0;
            double[] sampleFeatures = new double[D];
            String[] features;
            int max = sampleBatch[sampleBatch.length - 1];
            while ((line = br.readLine()) != null && counter <= max){
                // Periodically check if this thread has been interrupted. See the javadocs on
                // threading for best practices.
                if (Thread.currentThread().isInterrupted()) {
                    break;
                }

                if (ArrayUtils.binarySearch(sampleBatch, counter) >= 0) {
                    features = line.split(",| ");

                    for(int i = 0; i < D; i++) {
                        sampleFeatures[i] = Double.parseDouble(features[i]);
                    }
                    xBatch.add(sampleFeatures);
                }
                counter++;
            }
        } catch (IOException e) {
            // TODO(tylermzeller) probably a better way to handle this.
            e.printStackTrace();
        }
        return xBatch;
    }

    /**
     * This method is a left over from the original CrowdML codebase. Consider replacing with
     * updated code.
     *
     * @param sampleBatch
     * @param params
     * @return
     */
    public List<Integer> readLabels(int[] sampleBatch, final Parameters params){
        List<Integer> yBatch = new ArrayList<>();
        String labelSrc = params.getLabelSource();
        LossFunction loss = params.getLossFunction();

        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(CrowdMLApplication.getAppContext().getAssets().open(labelSrc)));
            String line;
            int counter = 0;
            while ((line = br.readLine()) != null && counter <= sampleBatch[sampleBatch.length - 1]){
                // Periodically check if this thread has been interrupted. See the javadocs on
                // threading for best practices.
                if (Thread.currentThread().isInterrupted()) {
                    break;
                }

                if (ArrayUtils.binarySearch(sampleBatch, counter) >= 0) {
                    line = line.trim();
                    int sampleLabel = Integer.parseInt(line);
                    if(sampleLabel == 0 && loss.lossType().equals("binary")){
                        sampleLabel = -1;
                    }
                    yBatch.add(sampleLabel);
                }
                counter++;
            }
        } catch (IOException e) {
            // TODO(tylermzeller) probably a better way to handle this.
            e.printStackTrace();
        }
        return yBatch;
    }

    public float[] getTFFeatureBatch(int[] indices, Parameters params) {
        int D = params.getD();
        int batchSize = params.getClientBatchSize();
        String featureSrc = params.getFeatureSource();
        // android studio was giving me some bs error about using Arrays.asList ???
        List<Integer> idcs = new ArrayList<>();
        for (int id : indices){
            idcs.add(id);
        }

        float[] trainingFeatures = new float[batchSize * D];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(
                    CrowdMLApplication.getAppContext().getAssets().open(featureSrc)));

            // do reading, usually loop until end of file reading
            int count = 0;
            int samples = 0;
            String line;
            String[] features;
            while ((line = reader.readLine()) != null) {
                // we've gathered all the batch samples
                if (samples == idcs.size()){
                    break;
                }

                //process line
                if (idcs.contains(count)){
                    features = line.split(",| ");
                    for(int i = 0; i < D; i++) {
                        trainingFeatures[samples * D + i] = Float.parseFloat(features[i]);
                    }
                    samples++;
                }
                count++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }

        return trainingFeatures;
    }

    public float[] getTFTestFeatures(int testN, Parameters params) {
        Log.d("readTestingFeatures","Begin");
        int D = params.getD();
        float[] testFeatures = new float[testN * D];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(
                    CrowdMLApplication.getAppContext().getAssets().open("MNISTTestImages.dat")));

            // do reading, usually loop until end of file reading
            int count = 0;
            String line;
            String[] features;
            while ((line = reader.readLine()) != null) {
                //process line
                features = line.split(",| ");
                for(int i = 0; i < D; i++) {
                    testFeatures[count * D + i] = Float.parseFloat(features[i]);
                    //sampleFeatures[i] = Double.parseDouble(features[i]);
                }
                count++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }

        return testFeatures;
    }

    public float[] getTFLabelBatch(int[] indices, Parameters params) {
        // android studio was giving me some bs error about using Arrays.asList ???
        List<Integer> idcs = new ArrayList<>();
        for (int id : indices){
            idcs.add(id);
        }

        int K = params.getK();
        int batchSize = params.getClientBatchSize();
        String labelSrc = params.getLabelSource();
        float[] trainingLabels = new float[batchSize * K];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(
                    CrowdMLApplication.getAppContext().getAssets().open(labelSrc)));

            // do reading, usually loop until end of file reading
            int count = 0;
            int sample = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                if (sample == idcs.size()){
                    break;
                }
                //process line
                if (idcs.contains(count)){
                    line = line.trim();
                    int sampleLabel = Integer.parseInt(line);
                    // For 1-hot encoding
                    trainingLabels[K * sample + sampleLabel] = 1;
                    sample++;
                }
                count++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        return trainingLabels;
    }

    public float[] getTFTestingLabels(int testN, Parameters params) {
        int K = params.getK();
        float[] testLabels = new float[testN * K];
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(
                    CrowdMLApplication.getAppContext().getAssets().open("MNISTTestLabels.dat")));

            // do reading, usually loop until end of file reading
            int count = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                //process line
                line = line.trim();
                int sampleLabel = Integer.parseInt(line);
                // For 1-hot encoding
                testLabels[K * count + sampleLabel] = 1;
                count++;
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        return testLabels;
    }

    public static int[] toOneHot(int num, int size){
        int[] oneHot = new int[size];
        oneHot[num - 1] = 1;
        return oneHot;
    }
}
