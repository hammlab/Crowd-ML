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


#import "NeuralNetwork.h"
@interface NeuralNetwork()
@end

@implementation NeuralNetwork


- (float *) computeNN :(const float *)trainingFeature :(const float *)trainingLabel :(const float *)w :(long)D :(int)classes :(double) regConstant :(float) L :(int)nh :(int)clientBatchSize
{
    
    int length = ((int)D + 1) * nh + (nh + 1) * nh + (nh + 1) *classes;
    float *gradloss = (float *) malloc(length * sizeof(float));
    
    int lengthw01 = (int)D * nh;
    int lengthw12 = nh*nh;
    int lengthw23 = nh*classes;
    int lengthb1 = nh;
    int lengthb2 = nh;
    int lengthb3 = classes;
    
    
    NSMutableArray *w01 = [NSMutableArray array];
    NSMutableArray *b1 = [NSMutableArray array];
    NSMutableArray *w12 = [NSMutableArray array];
    NSMutableArray *b2 = [NSMutableArray array];
    NSMutableArray *w23 = [NSMutableArray array];
    NSMutableArray *b3 = [NSMutableArray array];
    
    //parseParams
    //W01:
    int cnt = 0;
    for(int i = 0; i < lengthw01; i++){
        [w01 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
        cnt++;
    }
    
    //b1
    for(int i = 0; i < lengthb1; i++){
        [b1 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
        cnt++;
    }
    
    //W12
    for(int i = 0; i < lengthw12; i++){
        [w12 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
        cnt++;
    }
    
    
    //b2
    for(int i = 0; i < lengthb2; i++){
        [b2 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
        cnt++;
    }
    
    
    //W23
    for(int i = 0; i< lengthw23; i++){
        [w23 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
        cnt++;
    }
    
    //b3
    for(int i = 0; i < lengthb3; i++){
        [b3 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
        cnt++;
    }
    
    
    //getAvgGradient:
    //Forward pass
    float *h1 = (float *) malloc(nh * sizeof(float));
    float *h2 = (float *) malloc(nh * sizeof(float));
    
    //h1
    float dot = 0;
    for(int i = 0; i < nh; i++){
        dot = 0;
        for(int j = 0; j < D; j++){
            dot += *(trainingFeature + j) * [[w01 objectAtIndex:(i + j*nh)] floatValue];
        }
        float sum = dot + [[b1 objectAtIndex:i] floatValue];
        *(h1 + i) = MAX(sum, 0.0);
    }
    
    //h2
    for(int i = 0; i < nh; i++){
        dot = 0;
        for(int j = 0; j < nh; j++){
            dot += *(h1 + j) * [[w12 objectAtIndex:(i + j*nh)]floatValue];
        }
        float sum = dot + [[b2 objectAtIndex:i] floatValue];
        *(h2 + i) = MAX(sum, 0.0);
        
    }
    
    float max = -FLT_MAX;
    float *ai = (float *) malloc(classes * sizeof(float));
    float *scores = (float *) malloc(classes * sizeof(float));
    for(int i = 0; i < classes; i++){
        dot = 0;
        for(int j = 0; j < nh; j++){
            dot += *(h2 + j) * [[w23 objectAtIndex:(i + classes*j)]floatValue];
            
        }
        *(ai + i) = dot;
        max = MAX(dot, max);
        *(scores + i) = dot + [[b3 objectAtIndex: i]floatValue];
    }
    
    float denom = 0;
    float *prob = (float *) malloc(classes * sizeof(float));
    for(int i =0; i < classes; i++){
        dot = *(ai + i);
        denom += expf(dot - max);
        
    }
    for(int i = 0; i < classes; i++){
        dot = *(ai + i);
        *(prob + i) = expf(dot - max)/denom;
    }
    
    //Backward pass
    NSMutableArray *dw01 = [NSMutableArray array];
    NSMutableArray *db1 = [NSMutableArray array];
    NSMutableArray *dw12 = [NSMutableArray array];
    NSMutableArray *db2 = [NSMutableArray array];
    NSMutableArray *dw23 = [NSMutableArray array];
    NSMutableArray *db3 = [NSMutableArray array];
    NSMutableArray *dh1 = [NSMutableArray array];
    NSMutableArray *dh2 = [NSMutableArray array];
    
    /*
    float *dscores = (float *) malloc(classes * sizeof(float));
    for(int i = 0; i < classes; i++){
        if( i == (int)*trainingLabel){
            *(dscores + i) = *(prob + i) - 1;
        }else{
            *(dscores + i) = *(prob + i);
        }
    }
     */
    
    float *dscores = (float *) malloc(classes * sizeof(float));
    for(int i = 0; i < classes; i++){
        for(int k = 0; k < clientBatchSize; k++){
            if( i == (int)*(trainingLabel + k)){
                *(dscores + i) = *(prob + i) - 1;
            }else{
                *(dscores + i) = *(prob + i);
            }
        }
    }
    
    
    //dw23
    for(int i = 0; i < nh; i++){
        for(int j = 0; j < classes; j++){
            float value = *(h2 + i) * *(dscores + j) + L * [[w23 objectAtIndex:(j*nh + i)]floatValue];
            [dw23 addObject: [NSNumber numberWithFloat:value]];
            
        }
    }
    
    //db3
    for(int i = 0; i < classes; i++){
        [db3 addObject: [NSNumber numberWithFloat:*(dscores + i)]];
  
    }
    
    //dh2
    for(int i =0; i < nh; i++){
        dot = 0;
        for(int j = 0; j < classes; j++){
            dot += *(dscores + j) * [[w23 objectAtIndex:(i*classes + j)]floatValue];
        }
        if(*(h2 + i) <= 0){
            [dh2 addObject: [NSNumber numberWithFloat:0.0]];
        }else{
            [dh2 addObject: [NSNumber numberWithFloat:dot]];
            
        }
        
    }
    
    //dw12
    for(int i = 0; i < nh; i++){
        for(int j = 0; j < nh; j++){
            float value = *(h1 + i) * [[dh2 objectAtIndex:j] floatValue] + L * [[w12 objectAtIndex:(i + nh * j)]floatValue];
            [dw12 addObject: [NSNumber numberWithFloat:value]];
        }
    }
    
    //db2
    for(int i = 0; i < nh; i++){
        [db2 addObject: [dh2 objectAtIndex:i]];
        
        
    }
    
    //dh1
    for(int i = 0; i < nh; i++){
        dot = 0;
        for(int j = 0; j < nh; j++){
            dot += [[dh2 objectAtIndex: j] floatValue] * [[w12 objectAtIndex:(j + i * nh)]floatValue];
        }
        if(*(h1 + i) <= 0){
            [dh1 addObject: [NSNumber numberWithFloat:0.0f]];
            
        }else{
            [dh1 addObject: [NSNumber numberWithFloat:dot]];
            
        }
    }
    
    //dw01
    for(int i = 0; i < nh; i++){
        for(int j = 0; j < D; j++){
            float value = *(trainingFeature + j) * [[dh1 objectAtIndex:i] floatValue] + L * [[w01 objectAtIndex: ((j * nh) + i)]floatValue];
            
            [dw01 addObject: [NSNumber numberWithFloat:value]];
            
        }
    }
    
    
    //db1
    for(int i = 0; i < nh; i++){
        [db1 addObject: [dh1 objectAtIndex: i]];
        
    }
    
    //add dw01
    int ind= 0;
    for(int i = 0; i < lengthw01; i++){
        *(gradloss + ind) = [[dw01 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add db1
    for(int i = 0; i < lengthb1; i++){
        *(gradloss + ind) = [[db1 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add dw12
    for(int i = 0; i < lengthw12; i++){
        *(gradloss + ind) = [[dw12 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add db2
    for(int i = 0; i < lengthb2; i++){
        
        *(gradloss + ind) = [[db2 objectAtIndex: i] floatValue];
        ind++;
        
        
    }
    
    //add dw23
    for(int i = 0; i < lengthw23; i++){
        *(gradloss + ind) = [[dw23 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add db3
    for(int i = 0; i < lengthb3; i++){
        *(gradloss + ind) = [[db3 objectAtIndex: i] floatValue];
        ind++;
        
        
    }
    
    /*
    for(int i = 0; i < length; i++){
        NSLog(@"gradient: %d: %f",i, *(gradloss + i));
    }
     */
    
    free(prob);
    free(scores);
    free(h1);
    free(h2);
    free(ai);
    
    return gradloss;
}



/**
 Calculate accuracy for multi(10) classes
 **/
- (float)calculateTrainAccuracyWithWeightNN:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int)DFeatureSize :(int)classes :(long)Ntest :(long)featureSize :(int)nh

{
    
    featureName = @"MNISTTestImages";
    labelName = @"MNISTTestLabels";
    fileType = @"dat";
 
    float *labelVector;
    labelVector = [self readTrainingLabelFile:labelName :fileType :Ntest];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile:featureName :fileType : DFeatureSize :Ntest];

    
    int correct = 0;
    int lesscorrect = 0;

    for(int a = 0; a < Ntest; a++){
        int lengthw01 = (int)DFeatureSize * nh;
        int lengthw12 = nh*nh;
        int lengthw23 = nh*classes;
        int lengthb1 = nh;
        int lengthb2 = nh;
        int lengthb3 = classes;
        
        
        NSMutableArray *w01 = [NSMutableArray array];
        NSMutableArray *b1 = [NSMutableArray array];
        NSMutableArray *w12 = [NSMutableArray array];
        NSMutableArray *b2 = [NSMutableArray array];
        NSMutableArray *w23 = [NSMutableArray array];
        NSMutableArray *b3 = [NSMutableArray array];
        
        //parseParams
        //W01:
        int cnt = 0;
        for(int i = 0; i < lengthw01; i++){
            [w01 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
            cnt++;
        }
        
        //b1
        for(int i = 0; i < lengthb1; i++){
            [b1 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
            cnt++;
        }
        
        //W12
        for(int i = 0; i < lengthw12; i++){
            [w12 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
            cnt++;
        }
        
        
        //b2
        for(int i = 0; i < lengthb2; i++){
            [b2 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
            cnt++;
        }
        
        
        //W23
        for(int i = 0; i< lengthw23; i++){
            [w23 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
            cnt++;
        }
        
        //b3
        for(int i = 0; i < lengthb3; i++){
            [b3 addObject:[NSNumber numberWithFloat:*(w+cnt)]];
            cnt++;
        }
        
     
    
        //Forward pass
        float *h1 = (float *) malloc(nh * sizeof(float));
        float *h2 = (float *) malloc(nh * sizeof(float));
        
        //h1
        float dot = 0;
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < DFeatureSize; j++){
                dot += *(*(featureVector + a) + j) * [[w01 objectAtIndex:(i + j*nh)] floatValue];
            }
            float sum = dot + [[b1 objectAtIndex:i] floatValue];
            *(h1 + i) = MAX(sum, 0.0);
        }
        
        //h2
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += *(h1 + j) * [[w12 objectAtIndex:(i + j*nh)]floatValue];
            }
            float sum = dot + [[b2 objectAtIndex:i] floatValue];
            *(h2 + i) = MAX(sum, 0.0);
            
        }
        
        //float max = -FLT_MAX;
        //float *ai = (float *) malloc(classes * sizeof(float));
        float *scores = (float *) malloc(classes * sizeof(float));
        for(int i = 0; i < classes; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += *(h2 + j) * [[w23 objectAtIndex:(i + classes*j)]floatValue];
                
            }
            //*(ai + i) = dot;
            //max = MAX(dot, max);
            *(scores + i) = dot + [[b3 objectAtIndex: i]floatValue];
        }
        
        int bestGuess = 0;
        for(int h = 0; h < classes; h++){
            if(*(scores + h) > *(scores + bestGuess)){
                bestGuess = h;
            }
        }
        
        int lessGuess = 0;
        for(int h = 0; h < classes; h++){
            if( *(scores + h) < *(scores + lessGuess)){
                lessGuess = h;
            }
        }
        
        if(bestGuess == *(labelVector + a)){
            correct++;
        }
        if(lessGuess == *(labelVector + a)){
            lesscorrect++;
        }
        
        free(h1);
        free(h2);
        free(scores);
    }
    
    float accuracy = 100.0 * correct/Ntest;
    NSLog(@"Accuracy: %f, Correct: %d",accuracy, correct);
    float lessaccuracy = 100.0 * lesscorrect/Ntest;
    NSLog(@"lessAccuracy: %f, lessCorrect: %d",lessaccuracy, lesscorrect);
    
    free(labelVector);
    
    for(int i = 0; i < Ntest; i++) {
        free(*(featureVector + i));
    }
    
    free(featureVector);
    
    return accuracy;
    
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
