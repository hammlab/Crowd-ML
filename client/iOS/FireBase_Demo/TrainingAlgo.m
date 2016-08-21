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
#import "LogReg.h"

@import Accelerate;

@interface TrainingAlgo()

@property (nonatomic, readwrite) NSInteger trainingModelSize;
@property (nonatomic, readwrite) NSInteger featureSize;
@property (nonatomic, strong) LogReg *LogRegAlgo;
@property (nonatomic, strong) HingeLoss *HingeLossAlgo;
@property (nonatomic, strong) SoftMax *SoftMaxAlgo;
@property (nonatomic, strong) NeuralNetwork *NNAlgo;



@end

// Add Laplace noise
float * laplace (const float *grad, long D, double scale)
{
    double u;
    double sgn = 1;
    double radm;
    double lap;
    
    float *noise = (float *) malloc(D * sizeof(float));
    
    for(int i = 0; i < D; i++){
        u = *(grad + i);
        radm = (arc4random_uniform(INT_MAX) / (INT_MAX *1.0f)) - 0.5;
        if(radm < 0){
            sgn = -1;
        }else{
            sgn = 1;
        }
        lap = u - (scale/sqrt(2)) * sgn * exp( 1 - 2 * fabs(radm));
        *(noise + i) = lap;
    }
    
    return noise;
}

// Add Gaussian noise
float * gaussian (const float *grad, long D, double scale)
{
    float *noise = (float *) malloc(D * sizeof(float));
    double radm;
    double u;
    
    for(int i = 0; i < D; i++){
        u = *(grad + i);
        
        //Box-Muller Transformation:
        double x1 = arc4random_uniform(INT_MAX) / (INT_MAX *1.0f);
        double x2 = arc4random_uniform(INT_MAX) / (INT_MAX *1.0f);
        while(log(x1) == INFINITY || log(x1) == -INFINITY ){
            x1 = arc4random_uniform(INT_MAX) / (INT_MAX *1.0f);
        }
        
        radm = sqrt(-2 * log(x1)) * cos(2* M_PI * x2);
        
        *(noise + i) = radm * (scale/sqrt(2)) + u;
    }
    
    return noise;
    
}


//Use one of noise funtions
float * noiseLoss(float *loss, int noiseFunction, int length, double variance){
    if(noiseFunction == 1)
       return laplace(loss, length, variance);
    else if(noiseFunction == 2)
        return gaussian(loss, length, variance);
    else{
        float *noise = (float *) malloc(length * sizeof(float));
        for(int i = 0; i < length; i++){
            *(noise + i) = *(loss + i);
        }

        return noise;
    }
    
}

@implementation TrainingAlgo


/**
 Train model
 **/
- (float *) trainModelWithWeight:(float *)w :(int)lossFunction :(int)noiseFunction :(int)classes :(int)sbatch :(double)regConstant :(double)variance :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize :(int)getN :(float)L :(int)nh
{
    //initialize variables
    self.NNAlgo = [[NeuralNetwork alloc]init];
    self.LogRegAlgo = [[LogReg alloc]init];
    self.HingeLossAlgo = [[HingeLoss alloc]init];
    self.SoftMaxAlgo = [[SoftMax alloc]init];

    //Get labelsVector
    float *labelVector;
    labelVector = [self readTrainingLabelFile:labelName : fileType];
    
    //Get featuresVector
    float **featureVector;
    featureVector = [self readTrainingFeatureFile:featureName : fileType : DFeatureSize];
    self.featureSize = DFeatureSize;
    NSLog(@"The training set size: %ld", self.trainingModelSize);
    NSLog(@"The feature vector size: %ld", self.featureSize);
    

    int length = DFeatureSize;
    if(lossFunction == 3){
        length = length * classes;
    }else if(lossFunction == 4) {
        length = (length + 1) * nh + (nh + 1) * nh + (nh + 1) * classes;

    }
    
    //Array that is used to store randomized numbers which are indices of labelVector
    NSMutableArray *modelInd = [NSMutableArray array];
    
    float *grad = NULL;
    float *addGrad=(float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        *(addGrad + i) = 0;
    }
    float *avgGrad=(float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        *(avgGrad + i) = 0;
    }
    float *noiseGrad=NULL;

    //compute gradients
    if(w != NULL) {
        for(int i = 1; i <= sbatch; i++){
            
            long data =arc4random_uniform((UInt32)getN);
            NSUInteger jind = [modelInd indexOfObject:@(data)];
            while(jind != NSNotFound){
                data =arc4random_uniform((UInt32)getN);
                jind = [modelInd indexOfObject:@(data)];
            }
            [modelInd addObject:@(data)];
            if([modelInd count] == getN){
                [modelInd removeAllObjects];
            }
            
            [modelInd addObject:@(data)];      
            
            //Calculate gradients according to differnt loss functions
            if(lossFunction == 1){
                grad = [self.LogRegAlgo computeLossLog :*(featureVector + data):(labelVector + data) :w :self.featureSize :regConstant];
            }else if(lossFunction == 2){
                grad = [self.HingeLossAlgo computeLossSVM:*(featureVector + data) :(labelVector + data) :w :self.featureSize :regConstant :classes];
            }else if(lossFunction == 3){
                grad = [self.SoftMaxAlgo computeSoftMax:*(featureVector + data) :(labelVector + data) :w :self.featureSize :classes :regConstant];
            }else if(lossFunction == 4){
                grad = [self.NNAlgo computeNN:*(featureVector + data) :(labelVector + data) :w :self.featureSize :classes :regConstant :L :nh :sbatch];
            }
            
            
            for(int k = 0; k <length;k++){
                *(addGrad + k) = *(addGrad + k) + *(grad + k);

            }
            grad = NULL;
        }
    }

    //average gradients
    for(int k = 0; k <length;k++){
        *(avgGrad + k) = *(addGrad + k)/ sbatch;

    }
    NSLog(@"Gradient length: %d",length);
    
    //Add noise
    noiseGrad = noiseLoss(avgGrad, noiseFunction, length, variance);
    
    
    free(labelVector);
    
    for(int i = 0; i < self.trainingModelSize; i++) {
        free(*(featureVector + i));
    }
    
    free(featureVector);
    free(grad);
    free(addGrad);
    free(avgGrad);
    
    
    return noiseGrad;
}

/**
 Calculate accuracy
 **/
- (float)calculateTrainAccuracyWithWeight:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize :(int)classes :(int)lossFunction :(int)nh

{
    int Ntest = 1000;

    if(lossFunction == 1){
        return [self.LogRegAlgo calculateTrainAccuracyWithWeightBinary:w :labelName :featureName :fileType :DFeatureSize :classes :self.trainingModelSize :self.featureSize];
    }else if(lossFunction == 2){
        return [self.HingeLossAlgo calculateTrainAccuracyWithWeightBinary:w :labelName :featureName :fileType :DFeatureSize :classes :self.trainingModelSize :self.featureSize];
    }else if(lossFunction == 3){
        return [self.SoftMaxAlgo calculateTrainAccuracyWithWeightSoftMax:w :labelName :featureName :fileType :DFeatureSize :classes :Ntest];
    }else if(lossFunction == 4){
        return [self.NNAlgo calculateTrainAccuracyWithWeightNN:w :labelName :featureName :fileType :DFeatureSize :classes :Ntest :nh];
        
    }else{
        NSLog(@"Accuracy error!! Check your loss function");
        return 0.0;
    }
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
        //NSLog(@"Error: Label size and Feature size should be the same! ");
    
    if(self.trainingModelSize <= 0)
        return NULL;
    
    
    NSString *sep = @" ,";
    NSCharacterSet *set = [NSCharacterSet characterSetWithCharactersInString:sep];
    NSArray *features = [[listContent objectAtIndex:0] componentsSeparatedByCharactersInSet:set];
    self.featureSize = DfeatureSize;
    float **featureVectors = (float **) malloc(self.trainingModelSize * sizeof(float *));
    
    for (int i = 0; i < self.trainingModelSize; i++) {
        float *featureVector = (float *) malloc(self.featureSize * sizeof(float));
        features = [[listContent objectAtIndex:i] componentsSeparatedByCharactersInSet:set];
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
