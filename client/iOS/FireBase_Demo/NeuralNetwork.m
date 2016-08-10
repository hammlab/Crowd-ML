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
    float *grad = (float *) malloc(length * sizeof(float));
    
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
    [self parsePara:w :lengthw01 :lengthw12 :lengthw23 :lengthb1 :lengthb2 :lengthb3 :w01 :b1 :w12 :b2 :w23 :b3];
    
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
    
    
    float *dscores = (float *) malloc(classes * sizeof(float));
    for(int i = 0; i < classes; i++){
        if( i == (int)*trainingLabel){
            *(dscores + i) = *(prob + i) - 1;
        }else{
            *(dscores + i) = *(prob + i);
        }
    }
    
    /*
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
 */
    

    //dw23
    /////
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
    ////
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
    ////////////
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
        *(grad + ind) = [[dw01 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add db1
    for(int i = 0; i < lengthb1; i++){
        *(grad + ind) = [[db1 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add dw12
    for(int i = 0; i < lengthw12; i++){
        *(grad + ind) = [[dw12 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add db2
    for(int i = 0; i < lengthb2; i++){
        
        *(grad + ind) = [[db2 objectAtIndex: i] floatValue];
        ind++;
        
        
    }
    
    //add dw23
    for(int i = 0; i < lengthw23; i++){
        *(grad + ind) = [[dw23 objectAtIndex: i] floatValue];
        ind++;
        
    }
    
    //add db3
    for(int i = 0; i < lengthb3; i++){
        *(grad + ind) = [[db3 objectAtIndex: i] floatValue];
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
    
    return grad;
}



/*
- (float *) computeNN :(float **)trainingFeature :(const float *)trainingLabel :(const float *)w :(long)D :(int)classes :(double) regConstant :(float) L :(int)nh :(int)clientBatchSize
{
    
    int length = ((int)D + 1) * nh + (nh + 1) * nh + (nh + 1) *classes;
    float *grad = (float *) malloc(length * sizeof(float));
    
    
 
    //parseParams
    float w01[D][nh];
    //W01:
    int cnt = 0;
    for(int i = 0; i < D; i++){
        for(int j = 0; j < nh; j++){
            w01[i][j] = *(w + cnt);
            cnt++;
        }
    }
    
    //b1
    float b1[1][nh];
    for(int i = 0; i < nh; i++){
        b1[1][i] = *(w + cnt);
        cnt++;
    }
    
    //W12
    float w12[nh][nh];
    for(int i = 0; i < nh; i++){
        for(int j = 0; j < nh; j++){
            w12[i][j] = *(w + cnt);
            cnt++;
        }
    }
    
    
    //b2
    float b2[1][nh];
    for(int i = 0; i < nh; i++){
        b2[1][i] = *(w + cnt);
        cnt++;
    }
    
    
    //W23
    float w23[nh][classes];
    for(int i = 0; i< nh; i++){
        for(int j = 0; j < classes; j++){
            w23[i][j] = *(w + cnt);
            cnt++;
        }
    }
    
    //b3
    float b3[1][classes];
    for(int i = 0; i < classes; i++){
        b3[1][i] = *(w + cnt);
        cnt++;
    }

    
    int N = clientBatchSize;
    //getAvgGradient:
    //Forward pass
    float **h1 = (float **) malloc(N * sizeof(float));
    float **h2 = (float **) malloc(N * sizeof(float));
    
    //h1
    float dot = 0;
    for(int a = 0; a < N; a++){
        float *oneh1 = (float *)malloc(nh * sizeof(float));
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < D; j++){
                dot += *(*(trainingFeature + a) + j) * w01[j][i];
            }
            float sum = dot + b1[1][i];
            *(oneh1 + i) = MAX(sum, 0.0);
        }
        *(h1 + a) = oneh1;
        free(oneh1);
    }
    
    
    
    //h2
    for(int a = 0; a < N; a++){
        float *oneh2 = (float *)malloc(nh * sizeof(float));
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += *(*(h1+a) + j) * w12[j][i];
            }
            float sum = dot + b2[1][i];
            *(oneh2 + i) = MAX(sum, 0.0);
        }
        *(h2 + a) = oneh2;
        free(oneh2);

    }
    
    
    float *max = (float *) malloc(classes * sizeof(float));
    float *ai = (float *) malloc(classes * sizeof(float));
    float **scores = (float **) malloc(N * sizeof(float));

    //score
    for(int a =0; a < N; a++){
        float *oneScores = (float *) malloc(classes * sizeof(float));
        float maxScore = -FLT_MAX;
        for(int i = 0; i < classes; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                //NSLog(@"%f",*(*(h2 + a) + j));
                //NSLog(@"%f",w23[j][i]);

                dot += *(*(h2 + a) + j) * w23[j][i];
                
            }
            maxScore = MAX(dot, maxScore);
            *(oneScores + i) = dot + b3[1][i];
        }
        *(scores + a) = oneScores;
        *(max + a) = maxScore;
        free(oneScores);
    }
    
    //exp_score
    float **exp_scores = (float **) malloc(N * sizeof(float));
    float *denom = (float *) malloc(N * sizeof(float));

    for(int a = 0; a < N; a++){
        float *oneExp_scores = (float *) malloc(classes * sizeof(float));
        *(denom + a) = 0;
        for(int i = 0; i < classes; i++){
            *(oneExp_scores + i) = expf(*(*(scores + a) + i) - *(max + a));
            *(denom + a) += *(oneExp_scores + i);
        }
        *(exp_scores + a) = oneExp_scores;
        free(oneExp_scores);

    }
    
    //prob
    float **prob = (float **) malloc(N * sizeof(float));
    for(int a = 0; a < N; a++){
        float *oneProb = (float *) malloc(classes * sizeof(float));
        for(int i = 0; i < classes; i++){
            *(oneProb + i) = *(*(exp_scores + a) + i) / *(denom + a);
        }
        *(prob + a) = oneProb;
        free(oneProb);
    }

    
    //Backward pass
   
    float **dscores = (float **) malloc(N * sizeof(float));
    for(int a = 0; a < N; a++){
        float *oneDscores = (float *) malloc(classes * sizeof(float));
        for(int i = 0; i < classes; i++){
            if( i == (int)*(trainingLabel + a)){
                *(oneDscores + i) = *(*(prob + a) + i) - 1;
            }else{
                *(oneDscores + i) = *(*(prob + a) + i);
            }
        
        }
        *(dscores + a) = oneDscores;
        free(oneDscores);
    }
    
    
    //dw23
    float dw23[nh][classes];
    
    for(int i = 0; i < nh; i++){
        for(int a = 0; a < classes; a++){
            dot = 0;
            for(int j = 0; j < N; j++){
                dot += *(*(h2 + j) + i) * *(*(dscores + j)+a);
            }
            dw23[i][a] = dot / N + L * w23[i][a];
        }
    }
    
    //db3
    float db3[1][classes];
    for(int a= 0; a < classes; a++){
        db3[1][a] = 0;
        for(int i = 0; i < N; i++){
            db3[1][a] += *(*(dscores + i) + a);
        }
        db3[1][a] = db3[1][a]/N;
    }
    
    
    //dh2
    float dh2[N][nh];
    for(int a = 0; a < N; a++){
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < classes; j++){
                dot += *(*(dscores + a) + j) * w23[i][j];
            }
            dh2[a][i] = dot;
        }
    }
    
    for(int a = 0; a < N; a++){
        for(int i = 0; i < nh; i++){
            if(*(*(h2 + a)+i) <=0){
                dh2[a][i] = 0;
            }
        }
    }
    

    //dw12
    float dw12[nh][nh];
    for(int i = 0; i < nh; i++){
        for(int j = 0; j < nh; j++){
            dot = 0;
            for(int a = 0; a < N; a++){
                dot += *(*(h1 + a) + i) * dh2[a][j];
            }
            dw12[i][j] = dot / N + L * w12[i][j];
        }
    }
    
    
    //db2
    float db2[1][nh];
    for(int i = 0; i < nh; i++){
        db2[1][i] = 0;
        for(int a = 0; a< N; a++){
            db2[1][i] += dh2[a][i];
        }
        db2[1][i] /= N;
        
    }
    

    
    //dh1
    float dh1[N][nh];
    for(int a= 0; a < N; a++){
        for(int i = 0;i < nh; i++){
            dot = 0;
            for(int j = 0; j<nh; j++){
                dot += dh2[a][j] * w12[i][j];
            }
            dh1[a][i] = dot;
        }
    }
    
    for(int a = 0; a<N; a++){
        for(int i = 0; i < nh; i++){
            if(*(*(h1 + a) + i) <=0){
                dh1[a][i] = 0;
            }
        }
    }

    
    //dw01
    ////////////
    float dw01[D][nh];
    for(int i = 0; i < D; i++){
        for(int j = 0; j < nh; j++){
            dot = 0;
            for(int a = 0; a < N; a++){
                dot += *(*(trainingFeature + a) + i) * dh1[a][j];
            }
            dw01[i][j] = dot/N + L * w01[i][j];
        }
    }
    
    //db1
    float db1[1][nh];
    for(int i = 0; i < nh; i++){
        db1[1][i] = 0;
        for(int a = 0; a < N; a++){
            db1[1][i] += dh1[a][i];
        }
        db1[1][i] /= N;
    }
    
    
    //add dw01
    int ind= 0;
    for(int i = 0; i <nh; i++){
        for(int j = 0; j < D; j++){
            *(grad + ind) = dw01[i][j];
            ind++;
        }
    }
    
    //add db1
    for(int i = 0; i < nh; i++){
        *(grad + ind) =  db1[1][i];
        ind++;
        
    }
    
    //add dw12
    for(int i = 0; i < nh; i++){
        for(int j = 0; j < nh; j++){
            *(grad + ind) = dw12[i][j];
            ind++;
        }
    }
    
    //add db2
    for(int i = 0; i < nh; i++){
        *(grad + ind) = db2[1][i];
        ind++;
    }
    
    //add dw23
    for(int i = 0; i < nh; i++){
        for(int j = 0; i < classes; j++){
            *(grad + ind) = dw23[i][j];
            ind++;
        }
        
        
    }
    
    //add db3
    for(int i = 0; i < classes; i++){
        *(grad + ind) =db3[1][i];
        ind++;
    }
 
    
    free(prob);
    free(scores);
    free(h1);
    free(h2);
    free(ai);
    
    return grad;
}

*/


/**
 Calculate accuracy for multi(10) classes
 **/
- (float)calculateTrainAccuracyWithWeightNN:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int)DFeatureSize :(int)classes :(long)Ntest :(int)nh

{
    labelName = @"MNISTTestLabels";
    featureName = @"MNISTTestImages";
    fileType = @"dat";
 
    float *labelVector;
    labelVector = [self readTrainingLabelFile:labelName :fileType :Ntest];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile:featureName :fileType : DFeatureSize :Ntest];

    
    int correct = 0;
    int lesscorrect = 0;

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
    [self parsePara:w :lengthw01 :lengthw12 :lengthw23 :lengthb1 :lengthb2 :lengthb3 :w01 :b1 :w12 :b2 :w23 :b3];

    for(int a = 0; a < Ntest; a++){
        float *featureArray=(float *) malloc(DFeatureSize * sizeof(float));
        for(int k = 0; k < DFeatureSize; k++){
            *(featureArray + k) = *(*(featureVector + a) + k);
        }
        int label = (int)*(labelVector + a);

    
        //Forward pass
        float *h1 = (float *) malloc(nh * sizeof(float));
        float *h2 = (float *) malloc(nh * sizeof(float));
        
        //h1
        float dot = 0;
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < DFeatureSize; j++){
                dot += *(featureArray+ j) * [[w01 objectAtIndex:(i + j*nh)] floatValue];
            }
            float sum = dot + [[b1 objectAtIndex:i] floatValue];
            //*(h1 + i) = MAX(sum, 0.0);
            if(sum > 0){
                *(h1 + i) = sum;
            }else{
                *(h1 + i) = 0.0;
            }
        }
        
        //h2
        for(int i = 0; i < nh; i++){
            dot = 0;
            for(int j = 0; j < nh; j++){
                dot += *(h1 + j) * [[w12 objectAtIndex:(i + j*nh)]floatValue];
            }
            float sum = dot + [[b2 objectAtIndex:i] floatValue];
            //*(h2 + i) = MAX(sum, 0.0);
            if(sum > 0){
                *(h2 + i) = sum;
            }else{
                *(h2 + i) = 0.0;
            }
            
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
        
        if(bestGuess == label){
            correct++;
        }
        if(lessGuess == label){
            lesscorrect++;
        }
        
        free(h1);
        free(h2);
        free(scores);
        free(featureArray);
    }
    
    float accuracy = 100.0 * correct/Ntest;
    float lessaccuracy = 100.0 * lesscorrect/Ntest;
    
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


- (void)parsePara:(const float *)w :(int)lengthw01 :(int)lengthw12 :(int)lengthw23 :(int)lengthb1 :(int)lengthb2 :(int)lengthb3 :(NSMutableArray *)w01 :(NSMutableArray *)b1 :(NSMutableArray *)w12 :(NSMutableArray *)b2 :(NSMutableArray *)w23 :(NSMutableArray *)b3

{
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
    

    
}



@end
