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


#import "TrainingAlgo.h"
#import "UserDefine.h"
@import Accelerate;

@interface TrainingAlgo()

@property (nonatomic, readwrite) NSInteger trainingModelSize;
@property (nonatomic, readwrite) NSInteger featureSize;


@end

// Add Laplace noise
float * laplace (const float *loss, long D, float variance)
{
    float u;
    float b = 1;
    float radm;
    float lap;
    
    float *noise = (float *) malloc(D * sizeof(float));
    
    for(int i = 0; i < D; i++){
        u = *(loss + i);
        radm = (arc4random_uniform(100) / 100.0f) - 0.5;
        if(radm < 0){
            b = -1;
        }else{
            b = 1;
        }
        lap = u - variance * b * expf( 1 - 2 * fabsf(radm));
        *(noise + i) = lap;
    }
    
    return noise;
}

// Add Gaussian noise
float * gaussian (const float *loss, long D, float variance)
{
    float *noise = (float *) malloc(D * sizeof(float));
    float radm;
    float u;
    
    for(int i = 0; i < D; i++){
        u = *(loss + i);
        
        //Box-Muller Transformation:
        float x1 = arc4random_uniform(100) / 100.0f;
        float x2 = arc4random_uniform(100) / 100.0f;
        while(logf(x1) == INFINITY || logf(x1) == -INFINITY ){
            x1 = arc4random_uniform(100) / 100.0f;
        }

        radm = sqrtf(-2 * logf(x1)) * cosf(2* M_PI * x2);
        
        *(noise + i) = radm * variance + u;
    }
    
    return noise;
    
}

//This method is used for SVM Hinge Loss
float * computeLossSVM (const float *trainingFeature, const float *trainingLabel, const float *w, long D, float regConstant, int class)
{
    
    float h = 0;
    
    //Categorize
    float y = -1;
    if (*trainingLabel > 0)
        y = 1;
    
    //Compute dot product
    for(int i = 0; i < D; i++){
        h += *(trainingFeature + i) * *(w + i);
        
    }
    
    float *loss = (float *) malloc(D * sizeof(float));
    //Regularization variable
    float lambda = regConstant;
    
    //Compute gradients
    for(int i = 0; i < D; i++){
        
        float temp = *(trainingFeature + i) * y;
        float reg = 2 * *(w + i) * lambda;
        
        if(y * h >= 1)
            *(loss + i) = 0 + reg;
        else
            *(loss + i) = -1 * temp + reg;
        
    }

   
    
    return loss;
}


//This function is used for log regression
float * computeLossLog (const float *trainingFeature, const float *trainingLabel, const float *w, long D, float regConstant)
{
    //Dot product
    float h = 0;
    
    //Categorize
    float y = -1;
    if (*trainingLabel > 0)
        y = 1;
    
    //Compute dot product
    for(int i = 0; i < D; i++)
        h += *(trainingFeature + i) * *(w + i);
    
    float temp = exp( y * -1 * h);
    temp = y * temp / (1 + temp) * -1;
    
    float lambda = regConstant;
    
    float *loss = (float *) malloc(D * sizeof(float));
    
    vDSP_vsmul(trainingFeature, 1, &temp, loss, 1, D);
    
    //Regularization
    vDSP_vsma(w, 1, &lambda, loss, 1, loss, 1, D);

    
    
    return loss;
}

float * computeSoftMax (const float *trainingFeature, const float *trainingLabel, const float *w, long D, int classes, float regConstant)
{
    //If it is a binary class, change the variable to 1 in order to ensure the code below not to work
    //for multiple classes form.
    if(classes <= 2){
        classes = 1;
    }
   
    float *gradloss = (float *) malloc(D*classes * sizeof(float));
    
    
    float dot = 0;
    double denom = 0;
    double max=DBL_MIN;
    double *ai = (double *) malloc(classes * sizeof(double));
    
    //Store x dot w, and find the max dot product
    for(int i = 0; i < classes; i++){
        dot = 0;
        for(int j = 0; j < D; j++){
            dot += *(trainingFeature + j) * *(w + (D * i + j));
        }
        *(ai + i) = dot;
        max = MAX(dot, max);

    }
    
    //Compute the denominator of softmax function
    for(int i = 0; i < classes; i ++){
        denom += exp(*(ai + i) - max);
    }
   
    //Compute regularization
    float *regular = (float *) malloc(D*classes * sizeof(float));
    for (int i = 0; i < D*classes; i++){
        *(regular + i) = 2 * *(w + i) * regConstant;
       
    }
    
    //Compute gradients without adding regularization
    float prob = 0.0;
    int y = (int)*trainingLabel;
    if (y == 0 && classes == 2)
        y = -1;
    
    for(int i = 0; i < classes; i++){
        dot = 0;
        for(int j = 0; j < D; j++){
            dot += *(trainingFeature + j) * *(w + (j + D * i));

        }
        prob = expf(dot - max)/denom;
        int match = 0;
        if(i == y){
            match = 1;
        }
        for(int j = 0; j < D; j++){
            *(gradloss + (j + D * i)) = -1 * *(trainingFeature + j) * (match - prob);
        }
    }

    //add regularization
    for(int i = 0; i < D * classes; i++){
        *(gradloss + i) = *(gradloss + i) + *(regular + i);
    }
    
            
    return gradloss;
    
}

//Use one of loss funtions
float * computeLoss (const float *trainingFeature, const float *trainingLabel, const float *w, long D, int lossopt, float regConstant, int classes)
{
    
    //the choice of loss functions and noise functions depends on a user's definition in "UserDefine.m".
    if(lossopt == 1)
        return computeLossLog(trainingFeature, trainingLabel, w, D, regConstant);
    else if(lossopt == 2)
        return computeLossSVM(trainingFeature, trainingLabel, w, D, regConstant,classes);
    else
        return computeSoftMax(trainingFeature, trainingLabel, w, D, classes, regConstant);

    
}

//Use one of noise funtions
float * noiseLoss(float *loss, int noiseFunction, int length, float variance){
    if(noiseFunction == 1)
       return laplace(loss, length, variance);
    else if(noiseFunction == 2)
        return gaussian(loss, length, variance);
    else{
        return loss;
    }
    
}

@implementation TrainingAlgo


/**
 Train model
 **/
- (float *) trainModelWithWeight:(float *)w :(int)lossFunction :(int)noiseFunction :(int)classes :(int)sbatch :(float)regConstant :(float)variance :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize
{
    float *labelVector;
    labelVector = [self readTrainingLabelFile:labelName : fileType];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile:featureName : fileType : DFeatureSize];
    
    NSLog(@"The training set size: %ld", self.trainingModelSize);
    NSLog(@"The feature vector size: %ld", (long)self.featureSize);

    
    int length = (int)self.featureSize;
    if(classes > 2){
        length = length * classes;
    }
    
    //Array that is used to store randomized numbers which are indices of labelVector
    NSMutableArray *modelInd = [NSMutableArray array];
    for(int i = 0; i < self.trainingModelSize;i++){
        [modelInd addObject:@(i)];
    }
    
    float *loss = NULL;
    float *avgloss=(float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        *(avgloss + i) = 0;
    }
    float *noiseloss=(float *) malloc(length * sizeof(float));

    //compute gradients with batchSize
    if(w != NULL) {
        for(int i = 1; i <= sbatch; i++){
            long j =arc4random_uniform((UInt32)self.trainingModelSize);
            NSUInteger jind = [modelInd indexOfObject:@(j)];
            while(jind == NSNotFound){
                j =arc4random_uniform((UInt32)self.trainingModelSize);
                jind = [modelInd indexOfObject:@(j)];
            }
            [modelInd addObject:@(j)];
            loss = computeLoss(*(featureVector + j), labelVector + j, w, self.featureSize,lossFunction,regConstant,classes);
            noiseloss = noiseLoss(loss,noiseFunction, length, variance);
            
            for(int k = 0; k <length;k++){
                *(avgloss + k) = (*(noiseloss + k) + *(avgloss + k))/i;
            }
            loss = NULL;
        }
    }
    
    
   
    free(labelVector);
    
    for(int i = 0; i < self.trainingModelSize; i++) {
        free(*(featureVector + i));
    }
    
    free(featureVector);
    free(loss);
    free(avgloss);
    
    return noiseloss;
}

/**
 Calculate accuracy for binary test
 **/
- (float)calculateTrainAccuracyWithWeight:(float *)w : (NSString *)labelName : (NSString *)featureName : (NSString *)fileType : (int) DFeatureSize

{
    
    float *labelVector;
    labelVector = [self readTrainingLabelFile:labelName : fileType];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile:featureName : fileType : DFeatureSize];
    
    long truePositive = 0;
    
    for(int i = 0; i < self.trainingModelSize; i++) {
        float h = 0;
        for(int j = 0; j < self.featureSize; j++) {
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
    
    for(int i = 0; i < self.trainingModelSize; i++) {
        free(*(featureVector + i));
    }
    
    free(featureVector);
    
    return 1.0 * truePositive / self.trainingModelSize;
}

/**
 Read label file
 **/
- (float *) readTrainingLabelFile: (NSString *)labelSource : (NSString *) filetype {
    
    NSString *fileContent = [self readFileContentWithPath:labelSource type: filetype encoding:NSUTF8StringEncoding];
    
    NSArray *listContent = [fileContent componentsSeparatedByString:@"\n"];
    NSInteger labelSize = [listContent count];
    
    while([[listContent objectAtIndex:labelSize - 1] length] == 0)
        labelSize -= 1;
    
    self.trainingModelSize = labelSize;
    
    float *labelVector = (float *) malloc(self.trainingModelSize * sizeof(float));
    
    for (int i = 0; i < self.trainingModelSize; i++) {
        *(labelVector + i) = [[listContent objectAtIndex:i] floatValue];
    }
    
    return labelVector;
}



/**
 Read feature file
 **/
- (float **) readTrainingFeatureFile: (NSString *)FeatureSource : (NSString *) filetype : (int) DfeatureSize {
    
    
    NSString *fileContent = [self readFileContentWithPath:FeatureSource type:filetype encoding:NSUTF8StringEncoding];
    
    NSArray *listContent = [fileContent componentsSeparatedByString:@"\n"];
    NSInteger labelSize = [listContent count];
    
    //Eliminate empty lines at the end of files
    while([[listContent objectAtIndex:labelSize - 1] length] == 0)
        labelSize -= 1;
    
    if(self.trainingModelSize != labelSize)
        NSLog(@"Error: Label size and Feature size should be the same! ");
    
    if(self.trainingModelSize <= 0)
        return NULL;
    
    NSArray *features = [[listContent objectAtIndex:0] componentsSeparatedByString:@" "];
    self.featureSize = DfeatureSize;
    float **featureVectors = (float **) malloc(self.trainingModelSize * sizeof(float *));
    
    for (int i = 0; i < self.trainingModelSize; i++) {
        float *featureVector = (float *) malloc(self.featureSize * sizeof(float));
        features = [[listContent objectAtIndex:i] componentsSeparatedByString:@" "];
        for (int j = 0; j < self.featureSize; j++) {
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
