//
//  trainingLogReg.m
//  FireBase_iOS_Client_Demo
//
//  Created by JINJIN SHAO and Yani Xie on 3/7/16.
//  Copyright © 2016 Crowd-ML Team. All rights reserved.
//

#import "TrainingAlgo.h"
#import "UserDefine.h"
@import Accelerate;

@interface TrainingAlgo()

@property (nonatomic, readwrite) NSInteger trainingModelSize;
@property (nonatomic, readwrite) NSInteger featureSize;
@property (nonatomic, strong) UserDefine *userParam;


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
        radm = arc4random_uniform((UInt32)1) - 0.5;
        if(radm < 0){
            b = -1;
        }
        lap = u - variance * b * radm * expf( 1 - 2 * fabsf(radm));
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
        
        float x1 = arc4random_uniform(100) / 100.0f;
        float x2 = arc4random_uniform(100) / 100.0f;

        radm = sqrtf(-2 * logf(x1)) * cosf(2* M_PI * x2);
        
        *(noise + i) = radm * variance + u;
    }
    
    return noise;
    
}

//This method is used for SVM Hinge Loss
float * computeLossSVM (const float *trainingFeature, const float *trainingLabel, const float *w, long D, float regConstant, int class)
{
    
    float h = 0;
    
    float y =(int)*trainingLabel ;
    if (y == 0 && class ==2)
        y = -1;
    
    for(int i = 0; i < D; i++){
        h += *(trainingFeature + i) * *(w + i);
        
    }
    
    float *loss = (float *) malloc(D * sizeof(float));
    //Regularization variable
    float lambda = regConstant;
    
    
    for(int i = 0; i < D; i++){
        float temp = *(trainingFeature + i) * y;
        float reg =2 * *(w + i) * lambda;
        
        if(*trainingLabel * h >= 1)
            *(loss + i) = 0 + reg;
        else
            *(loss + i) = -1 * temp + reg;
        
        //calculate and add regularization
        //*(loss + i) = MAX(0, 1 - *trainingLabel * h) + *(w + i) * lambda;
        
    }

   
    
    return loss;
}


//This function is used for log regression
float * computeLossLog (const float *trainingFeature, const float *trainingLabel, const float *w, long D)
{
    float h = 0;
    
    float y = -1;
    if (*trainingLabel > 0)
        y = 1;
    
    for(int i = 0; i < D; i++)
        h += *(trainingFeature + i) * *(w + i);
    
    float temp = exp( y * -1 * h);
    temp = y * temp / (1 + temp) * -1;
    
    float lambda = 0.000001;
    

    float *loss = (float *) malloc(D * sizeof(float));
    
    vDSP_vsmul(trainingFeature, 1, &temp, loss, 1, D);
    
    //Regularization
    vDSP_vsma(w, 1, &lambda, loss, 1, loss, 1, D);

    
    
    return loss;
}

float * computeSoftMax (const float *trainingFeature, const float *trainingLabel, const float *w, long D, int classes, float regConstant)
{
    //If it is a binary class, change the variable to 1 in order to ensure the code below not to work
    //for multiple classes.
    if(classes <= 2){
        classes = 1;
    }
   
    float *gradloss = (float *) malloc(D*classes * sizeof(float));
    
    
    float dot = 0;
    float denom = 0;
    
    for(int i = 0; i < classes; i++){
        dot = 0;
        for(int j = 0; j < D; j++)
            dot += *(trainingFeature + j) * *(w + (D * i + j));

        denom += expf(dot);

    }
    
    float *regular = (float *) malloc(D*classes * sizeof(float));
    for (int i = 0; i < D*classes; i++){
        *(regular + i) = 2 * *(w + i) * regConstant;
       
    }
    
    float prob;
    int y = (int)*trainingLabel;
    if (y == 0 && classes == 2)
        y = -1;
    
    for(int i = 0; i < classes; i++){
        dot = 0;
        for(int j = 0; j < D; j++){
            dot += *(trainingFeature + j) * *(w + (j + D * i));
        }
        prob = expf(dot)/denom;
        int match = 0;
        if(i == y){
            match = 1;
        }
        for(int j = 0; j < D; j++){
            *(gradloss + (j + D * i)) = -1 * *(trainingFeature + j) * (match - prob);
        }
    }
    
    for(int i = 0; i < D * classes; i++){
        *(gradloss + i) = *(gradloss + i) + *(regular + i);
    }
    
            
    return gradloss;
    
}

//Use one of loss funtions
float * computeLoss (const float *trainingFeature, const float *trainingLabel, const float *w, long D, int lossopt, float regConstant, int classes)
{
    
    //the choice of loss functions and noise functions depends on a user's definition in "UserDefine.m".
    float *loss;
    if(lossopt == 1)
        loss = computeLossLog(trainingFeature, trainingLabel, w, D);
    else if(lossopt == 2)
        loss = computeLossSVM(trainingFeature, trainingLabel, w, D, regConstant,classes);
    else
        loss = computeSoftMax(trainingFeature, trainingLabel, w, D, classes, regConstant);


    return loss;
    
}


@implementation TrainingAlgo

- (void) trainModelWithSelfTest {
    float *labelVector;
    labelVector = [self readTrainingLabelFile];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile];
    
    NSLog(@"The training set size: %ld", self.trainingModelSize);
    NSLog(@"The feature vector size: %ld", self.featureSize);
    
    self.userParam = [[UserDefine alloc] init];
    
    int lossFunction = (int)[self.userParam lossOpt];
    int noiseFunction = (int)[self.userParam noiseOpt];

    NSInteger Niter = 5;
    NSInteger t = 0;
    float *w;
    w = (float *)malloc(self.featureSize * sizeof(float));
    
    for(int i = 0; i < self.featureSize; i++) {
        *(w + i) = 0;
    }
    
    float *loss;
    float rate;
    float c = 1e-6;
    float regConstant = [self.userParam RegConstant];
    float variance = [self.userParam variance];
    int classes = [self.userParam Classes];

    for (int i = 0; i < Niter; i++) {
        for (int j = 0; j < self.trainingModelSize; j++) {
            t += 1;
            
            loss = computeLoss(*(featureVector + j), labelVector + j, w, self.featureSize,lossFunction, regConstant, classes);
            rate = c / t * -1;
            
            vDSP_vsma(loss, 1, &rate, w, 1, w, 1, self.featureSize);
            
            free(loss);
        }
    }
    
    NSLog(@"Training Done!");
    NSLog(@"Calculating training error ... ");
    
    //Self Test
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
    
    NSLog(@"Training Error: %f", 1.0 * truePositive / self.trainingModelSize);
    
}

- (float *)trainModelWithWeight: (float *)w
{
    float *labelVector;
    labelVector = [self readTrainingLabelFile];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile];
    
    NSLog(@"The training set size: %ld", self.trainingModelSize);
    NSLog(@"The feature vector size: %ld", (long)self.featureSize);

    
    self.userParam = [[UserDefine alloc] init];
    int lossFunction =(int)[self.userParam lossOpt];
    int noiseFunction = (int)[self.userParam noiseOpt];
    int classes = [self.userParam Classes];
    
    
    int sbatch = [self.userParam batchSize];
    float regConstant = [self.userParam RegConstant];
    float variance = [self.userParam variance];
    
    int length = (int)self.featureSize;
    if(classes > 2){
        length = length * classes;
    }
   
    
    NSMutableArray *modelInd = [NSMutableArray array];
    for(int i = 0; i < self.trainingModelSize;i++){
        [modelInd addObject:@(i)];
    }
    
    float *loss = NULL;
    float *avgloss=(float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        *(avgloss + i) = 0;
    }
    
    //Average loss.
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
            
            for(int k = 0; k <length;k++){
                *(avgloss + k) = (*(loss + k) + *(avgloss + k))/i;
            }
            loss = NULL;
        }
    }
    
    float *noiseloss=(float *) malloc(length * sizeof(float));
    if(noiseFunction == 1)
        noiseloss = laplace(avgloss, self.featureSize, variance);
    else if(noiseFunction == 2)
        noiseloss = gaussian(avgloss, self.featureSize, variance);
    else{
        for(int i = 0; i < length; i++) {
            *(noiseloss + i) = *(avgloss + i);
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

- (float)calculateTrainAccuracyWithWeight:(float *)w
{
    self.userParam = [[UserDefine alloc] init];
    
    float *labelVector;
    labelVector = [self readTrainingLabelFile];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile];
    
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

- (float *) readTrainingLabelFile {
    self.userParam = [[UserDefine alloc] init];
    
    NSString *fileContent = [self readFileContentWithPath:[self.userParam labelSourceName] type:[self.userParam sourceType] encoding:NSUTF8StringEncoding];
    
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

- (float **) readTrainingFeatureFile {
    self.userParam = [[UserDefine alloc] init];
    
    NSString *fileContent = [self readFileContentWithPath:[self.userParam featureSourceName] type:[self.userParam sourceType] encoding:NSUTF8StringEncoding];
    
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
    //self.featureSize = [features count];
    self.featureSize = [self.userParam featureSize];
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
