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

//This method is used for SVM Hinge Loss
float * computeLossSVM (const float *trainingFeature, const float *trainingLabel, const float *w, long D, double regConstant, int class)
{
    
    double h = 0;
    
    //Categorize
    double y = -1;
    if (*trainingLabel > 0)
        y = 1;
    
    //Compute dot product
    for(int i = 0; i < D; i++){
        h += *(trainingFeature + i) * *(w + i);
        
    }
    
    float *loss = (float *) malloc(D * sizeof(float));
    //Regularization variable
    double lambda = regConstant;
    
    //Compute gradients
    for(int i = 0; i < D; i++){
        
        double temp = *(trainingFeature + i) * y;
        double reg = 2 * *(w + i) * lambda;
        
        if(y * h >= 1)
            *(loss + i) = 0 + reg;
        else
            *(loss + i) = -1 * temp + reg;
        
    }

   
    
    return loss;
}


//This function is used for log regression
float * computeLossLog (const float *trainingFeature, const float *trainingLabel, const float *w, long D, double regConstant)
{
    //Dot product
    double h = 0;
    
    //Categorize
    double y = -1;
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

float * computeSoftMax (const float *trainingFeature, const float *trainingLabel, const float *w, long D, int classes, double regConstant)
{
    //If it is a binary class, change the variable to 1 in order to ensure the code below not to work
    //for multiple classes form.
    if(classes <= 2){
        classes = 1;
    }
    
    float *gradloss = (float *) malloc(D*classes * sizeof(float));
    for(int i = 0; i < D*classes; i++){
        *(gradloss + i) = 0.0;
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
            *(gradloss + (j + (D * i))) = -1 * *(trainingFeature + j) * (match - prob) + 2 * *(w + (j + (D * i))) * regConstant;
        }
    }
    
    
    free(ai);
    return gradloss;
}


float * computeNN (const float *trainingFeature, const float *trainingLabel, const float *w, long D, int classes, double regConstant, float L, int nh){
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


    float *dscores = (float *) malloc(classes * sizeof(float));
    for(int i = 0; i < classes; i++){
        if( i == (int)*trainingLabel){
            *(dscores + i) = *(prob + i) - 1;
        }else{
            *(dscores + i) = *(prob + i);
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
    }
    
     //add db1
    for(int i = 0; i < lengthb1; i++){
        *(gradloss + ind) = [[db1 objectAtIndex: i] floatValue];
    }
    
    //add dw12
    for(int i = 0; i < lengthw12; i++){
        *(gradloss + ind) = [[dw12 objectAtIndex: i] floatValue];
    }

    //add db2
    for(int i = 0; i < lengthb2; i++){
        
        *(gradloss + ind) = [[db2 objectAtIndex: i] floatValue];

    }

    //add dw23
    for(int i = 0; i < lengthw23; i++){
        *(gradloss + ind) = [[dw23 objectAtIndex: i] floatValue];
    }
    
    //add db3
    for(int i = 0; i < lengthb3; i++){
        *(gradloss + ind) = [[db3 objectAtIndex: i] floatValue];

    }

    free(prob);
    free(scores);
    free(h1);
    free(h2);
    free(ai);

    return gradloss;
}

//Use one of loss funtions
float * computeLoss (const float *trainingFeature, const float *trainingLabel, const float *w, long D, int lossopt, double regConstant, int classes, float L, int nh)
{
    
    //the choice of loss functions and noise functions depends on a user's definition in "UserDefine.m".
    if(lossopt == 1)
        return computeLossLog(trainingFeature, trainingLabel, w, D, regConstant);
    else if(lossopt == 2)
        return computeLossSVM(trainingFeature, trainingLabel, w, D, regConstant,classes);
    else if(lossopt == 3)
        return computeSoftMax(trainingFeature, trainingLabel, w, D, classes, regConstant);
    else
        return computeNN(trainingFeature, trainingLabel, w, D, classes, regConstant, L, nh);


    
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
    float *labelVector;
    labelVector = [self readTrainingLabelFile:labelName : fileType];
    
    float **featureVector;
    featureVector = [self readTrainingFeatureFile:featureName : fileType : DFeatureSize];
    self.featureSize = DFeatureSize;
    NSLog(@"The training set size: %ld", self.trainingModelSize);
    NSLog(@"The feature vector size: %ld", self.featureSize);
    

    int length = (int)self.featureSize;
    if(lossFunction == 3){
        length = length * classes;
    }else if(lossFunction == 4) {
        length = (length + 1) * nh + (nh + 1) * nh + (nh + 1) * classes;

    }
    
    //Array that is used to store randomized numbers which are indices of labelVector
    NSMutableArray *modelInd = [NSMutableArray array];
    
    float *loss = NULL;
    float *addGrad=(float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        *(addGrad + i) = 0;
    }
    float *avgloss=(float *) malloc(length * sizeof(float));
    for(int i = 0; i < length; i++) {
        *(avgloss + i) = 0;
    }
    float *noiseloss=NULL;

    //compute gradients with batchSize
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
            
            loss = computeLoss(*(featureVector + data), labelVector + data, w, self.featureSize,lossFunction,regConstant,classes, L, nh);
         
            for(int k = 0; k <length;k++){
                *(addGrad + k) = *(addGrad + k) + *(loss + k);
            }
            loss = NULL;
        }
    }

    for(int k = 0; k <length;k++){
        *(avgloss + k) = *(addGrad + k)/ sbatch;

    }
    noiseloss = noiseLoss(avgloss, noiseFunction, length, variance);
    
    
    free(labelVector);
    
    for(int i = 0; i < self.trainingModelSize; i++) {
        free(*(featureVector + i));
    }
    
    free(featureVector);
    free(loss);
    free(addGrad);
    free(avgloss);
    
    
    return noiseloss;
}

/**
 Calculate accuracy for binary test
 **/
- (float)calculateTrainAccuracyWithWeight:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize :(int)classes

{
    
    long truePositive = 0;
    
    if(classes == 2){
        float *labelVector;
        labelVector = [self readTrainingLabelFile:labelName : fileType];
        
        float **featureVector;
        featureVector = [self readTrainingFeatureFile:featureName : fileType : DFeatureSize];
    
        for(int i = 0; i < self.trainingModelSize; i++) {
            double h = 0;
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
        
    }else{
        
        float *labelVector;
        labelName = @"MNISTTestLabels";
        labelVector = [self readTrainingLabelFile:labelName : fileType];
        
        float **featureVector;
        labelName = @"MNISTTestImages";
        featureVector = [self readTrainingFeatureFile:featureName : fileType : DFeatureSize];
        
        for(int i = 0; i < 1000; i++){
            float *ai=(float *) malloc(classes * sizeof(float));
        
            for(int h = 0; h < classes; h++){
                double dot = 0;
                for(int j = 0; j < self.featureSize; j++){
                    dot += *(*(featureVector + h) + j) * *(w + (j + h * self.featureSize));
                }
                *(ai + h) = dot;
            }
            int bestGuess = 0;
            for(int h = 0; h < classes; h++){
                if(*(ai + h) > *(ai + bestGuess)){
                    bestGuess = h;
                }
            }
            int label = (int)*(labelVector + i);
            if(bestGuess == label){
                truePositive++;
            }
        }
        
        free(labelVector);
        
        for(int i = 0; i < self.trainingModelSize; i++) {
            free(*(featureVector + i));
        }
        
        free(featureVector);
        
        
        return 100*truePositive/1000;
        
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
