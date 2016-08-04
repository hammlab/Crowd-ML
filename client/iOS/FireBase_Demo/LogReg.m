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


#import "LogReg.h"
#import <Foundation/Foundation.h>
@import Accelerate;

@interface LogReg()

@end
@implementation LogReg

//This function is used for log regression
- (float *) computeLossLog :(const float *)trainingFeature :(const float *)trainingLabel :(const float *)w :(long) D :(double) regConstant
{
    
    //Dot product
    float h = 0;
    
    //Categorize
    float y = -1;
    if (*trainingLabel > 0)
        y = 1;
    
    //Compute dot product
    for(int i = 0; i < D; i++){
        h += *(trainingFeature + i) * *(w + i);
       // NSLog(@"method: %d, %f, feature: %f",i, *(w+i),*(trainingFeature + i));
    }
    
    
    double temp = exp( y * -1 * h);
    //NSLog(@"exp: %f", y * -1 * h);
    //NSLog(@"temp = %f",temp);

    temp = y * exp( y * -1 * h) / (1 + exp( y * -1 * h)) * -1;
    //NSLog(@"y*temp = %f, 1+temp = %f",y * temp,(1 + temp));
    
    float lambda = (float)regConstant;
    
    float *grad = (float *) malloc(D * sizeof(float));
    float doubleTemp = (float)temp;
    
    vDSP_vsmul(trainingFeature, 1, &doubleTemp, grad, 1, D);
    
    //Regularization
    vDSP_vsma(w, 1, &lambda, grad, 1, grad, 1, D);
    
    
    
    return grad;
}

/**
 Calculate accuracy for binary class
 **/
- (float)calculateTrainAccuracyWithWeightBinary:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize :(int)classes :(long)trainingModelSize :(long)featureSize

{
    if(classes == 2){
    
        long truePositive = 0;
   
        float *labelVector;
        labelVector = [self readTrainingLabelFile:labelName :fileType :trainingModelSize];
    
        float **featureVector;
        featureVector = [self readTrainingFeatureFile:featureName :fileType : DFeatureSize :trainingModelSize];
    
        for(int i = 0; i < trainingModelSize; i++) {
            double h = 0;
            for(int j = 0; j <featureSize; j++) {
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
        
        for(int i = 0; i < trainingModelSize; i++) {
            free(*(featureVector + i));
        }
        
        free(featureVector);
        
        return 1.0 * truePositive / trainingModelSize;
    }else{
        NSLog(@"LogReg only works for binary classes.");
        return 0.0;
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
