/**
 
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
 
 
 FireBase_iOS_Client_Demo
 
 **/


#import "SoftMax.h"
@interface SoftMax()
@end

@implementation SoftMax

-(float *) computeSoftMax :(const float *)trainingFeature :(const float *)trainingLabel :(const float *)w :(long) D :(int) classes :(double) regConstant
{
    //If it is a binary class, change the variable to 1 in order to ensure the code below not to work
    //for multiple classes form.
    if(classes <= 2){
        classes = 1;
    }
    
    float *grad = (float *) malloc(D*classes * sizeof(float));
    for(int i = 0; i < D*classes; i++){
        *(grad + i) = 0.0;
    }
    
    double dot = 0.0;
    double denom = 0.0;
    double max = -DBL_MAX;
    float *ai = (float *) malloc(classes * sizeof(float));
    
    //Store x dot w, and find the max dot product
    
    for(int i = 0; i < classes; i++){
        dot = 0;
        for(int j = 0; j < D; j++){
            dot += *(trainingFeature + j) * *(w + (j + (D * i)));
        }
        *(ai + i) = dot;
        max = MAX(dot, max);
    }
    
    //Compute the denominator of softmax function
    for(int i = 0; i < classes; i ++){
        denom += exp(*(ai + i) - max);
    }
    
    
    //Compute gradients and add regularization
    double prob = 0.0;
    int y = (int)*trainingLabel;
    if (y == 0 && classes <= 2)
        y = -1;
    
    for(int i = 0; i < classes; i++){
        dot = *(ai + i);
        prob = (exp(dot - max))/denom;
        
        double match = 0.0;
        if(i == y){
            match = 1.0;
        }
        for(int j = 0; j < D; j++){
            *(grad+ (j + (D * i))) = -1 * *(trainingFeature + j) * (match - prob) + 2 * *(w + (j + (D * i))) * regConstant;
        }
    }
    
    
    free(ai);
    return grad;
}

/**
 Calculate accuracy for multi(10) classes
 **/
- (float)calculateTrainAccuracyWithWeightSoftMax:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int)DFeatureSize :(int)classes :(int)Ntest

{
    
    long truePositive = 0;
    
    if(classes == 2){
        
        float *labelVector;
        labelVector = [self readTrainingLabelFile:labelName :fileType :Ntest];
        
        float **featureVector;
        featureVector = [self readTrainingFeatureFile:featureName :fileType : DFeatureSize :Ntest];
        
        for(int i = 0; i < Ntest; i++) {
            double h = 0;
            for(int j = 0; j <DFeatureSize; j++) {
                h += *(*(featureVector + i) + j) * *(w + j);
            }
            
            if(h > 0 && *(labelVector + i) > 0) {
                truePositive += 1;
            }
            
            if(h < 0 && *(labelVector + i) < 1) {
                truePositive += 1;
            }
            
        }
        
        free(labelVector);
        
        for(int i = 0; i < Ntest; i++) {
            free(*(featureVector + i));
        }
        
        free(featureVector);
        
        return 1.0 * truePositive / Ntest;
    }else{
        
        //Load test files
        float *labelVector;
        labelName = @"MNISTTestLabels";
        labelVector = [self readTrainingLabelFile:labelName :fileType :Ntest];
        
        float **featureVector;
        featureName = @"MNISTTestImages";
        featureVector = [self readTrainingFeatureFile:featureName :fileType : DFeatureSize :Ntest];
        
        for(int i = 0; i < Ntest; i++){
            float *ai=(float *) malloc(classes * sizeof(float));
            int label = (int)*(labelVector + i);

            float *featureArray=(float *) malloc(DFeatureSize * sizeof(float));
            for(int a = 0; a < DFeatureSize; a++){
                *(featureArray + a) = *(*(featureVector + i) + a);
            }

            
            for(int h = 0; h < classes; h++){
                double dot = 0;
                for(int j = 0; j < DFeatureSize; j++){
                    dot += *(featureArray + j) * *(w + (j + h * DFeatureSize));
                }
                *(ai + h) = dot;
            }
            
            int bestGuess = 0;
            for(int h = 0; h < classes; h++){
                if(*(ai + h) > *(ai + bestGuess)){
                    bestGuess = h;
                }
            }

            if(bestGuess == label){
                truePositive++;
            }
            free(ai);
            free(featureArray);
        }
        
        free(labelVector);
        
        for(int i = 0; i < Ntest; i++) {
            free(*(featureVector + i));
        }
        
        free(featureVector);
        
        return 100.0*truePositive/Ntest;
        
    }
}


/**
 Read label file
 **/
- (float *) readTrainingLabelFile: (NSString *)labelSource : (NSString *) filetype :(long)trainingModelSize {
    
    NSString *fileContent = [self readFileContentWithPath:labelSource type: filetype encoding:NSUTF8StringEncoding];
    
    NSArray *listContent = [fileContent componentsSeparatedByString:@"\n"];
    NSInteger labelSize = [listContent count];
    
    while([[listContent objectAtIndex:labelSize - 1] length] == 0)
        labelSize -= 1;
    
    //trainingModelSize = labelSize;
    
    float *labelVector = (float *) malloc(trainingModelSize * sizeof(float));
    
    for (int i = 0; i < trainingModelSize; i++) {
        *(labelVector + i) = [[listContent objectAtIndex:i] floatValue];
    }
    
    return labelVector;
}


/**
 Read feature file
 **/
- (float **) readTrainingFeatureFile: (NSString *)FeatureSource : (NSString *) filetype : (int) DfeatureSize :(long)trainingModelSize  {
    
    
    NSString *fileContent = [self readFileContentWithPath:FeatureSource type:filetype encoding:NSUTF8StringEncoding];
    
    NSArray *listContent = [fileContent componentsSeparatedByString:@"\n"];
    NSInteger labelSize = [listContent count];
    
    //Eliminate empty lines at the end of files
    while([[listContent objectAtIndex:labelSize - 1] length] == 0)
        labelSize -= 1;
    
    if(trainingModelSize != labelSize)
        //NSLog(@"Error: Label size and Feature size should be the same! ");
        
        if(trainingModelSize <= 0)
            return NULL;
    
    
    NSString *sep = @" ,";
    NSCharacterSet *set = [NSCharacterSet characterSetWithCharactersInString:sep];
    NSArray *features = [[listContent objectAtIndex:0] componentsSeparatedByCharactersInSet:set];
    float **featureVectors = (float **) malloc(trainingModelSize * sizeof(float *));
    
    for (int i = 0; i < trainingModelSize; i++) {
        float *featureVector = (float *) malloc(DfeatureSize * sizeof(float));
        features = [[listContent objectAtIndex:i] componentsSeparatedByCharactersInSet:set];
        for (int j = 0; j < DfeatureSize; j++) {
            *(featureVector + j) = [[features objectAtIndex:j] floatValue];
        }
        
        *(featureVectors + i) = featureVector;
    }
    
    return featureVectors;
}

- (NSString *) readFileContentWithPath: (NSString *)filePath
                                  type: (NSString *)fileType
                              encoding: (NSStringEncoding)encoding
{
    NSError *error;
    
    NSString *pathAndType = [[NSBundle mainBundle] pathForResource:filePath ofType:fileType];
    
    NSString *fileContent = [NSString stringWithContentsOfFile:pathAndType encoding:encoding error:&error];
    
    if(error) {
        NSLog(@"Error reading file: %@", error.localizedDescription);
        NSLog(@"Error reading file: %@ ___ %@", filePath,fileType);
    }
    
    return fileContent;
}





@end
