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

#import "UserDefine.h"
#import <Foundation/Foundation.h>
@import Firebase;

@interface UserDefine()

/**
 *  The Firebase reference this userDefine instance uses.
 */

@property (nonatomic, readwrite) int D_featureSize;
@property (nonatomic, readwrite) int K_class;
@property (nonatomic, readwrite) float L_reg;
@property (nonatomic, readwrite) int N_test;
@property (nonatomic, readwrite) int clientBatchS;
@property (nonatomic, readwrite) NSString *featureSource;
@property (nonatomic, readwrite) NSString *labelSource;
@property (nonatomic, readwrite) NSString *lossFunction;
@property (nonatomic, readwrite) NSString *noiseType;
@property (nonatomic, readwrite) float noiseVariance;
@property (nonatomic, readwrite) int paramIterator;
@property (nonatomic, readwrite) int nhNumber;
@property (nonatomic, readwrite) int naught;
@property (nonatomic, readwrite) int localUpdate;
@property (nonatomic, readwrite) int maxIter;



@end

@implementation UserDefine

/**
 Initialize params
 **/
- (void)Initialize:(FIRDatabaseReference *) paramRef {
    
    //D
    FIRDatabaseReference *DRef = [paramRef child:@"D"];
    [DRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: D is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *featureSize = snapshot.value;
            self.D_featureSize = [featureSize intValue];
        }
    }];
    
    //K
    FIRDatabaseReference *KRef = [paramRef child:@"K"];
    [KRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: K is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *class = snapshot.value;
            self.K_class = [class intValue];
        }
    }];
    
    //L
    FIRDatabaseReference *LRef = [paramRef child:@"L"];
    [LRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: L is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *regularization = snapshot.value;
            self.L_reg = [regularization floatValue];
        }
    }];
    
    //N
    FIRDatabaseReference *NRef = [paramRef child:@"N"];
    [NRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: N is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *testSample = snapshot.value;
            self.N_test = [testSample intValue];
        }
    }];
    
    //clientBatchSize
    FIRDatabaseReference *clientBatchSizeRef = [paramRef child:@"clientBatchSize"];
    [clientBatchSizeRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: clientBatchSize is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *clientBatch = snapshot.value;
            self.clientBatchS = [clientBatch intValue];
        }
    }];
    
    
    //featureSource
    FIRDatabaseReference *featureSourceRef = [paramRef child:@"featureSource"];
    [featureSourceRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: featureSource is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSString *featureFile = snapshot.value;
            self.featureSource = featureFile;
        }
    }];
    
   
    
    //labelSource
    FIRDatabaseReference *labelSourceRef = [paramRef child:@"labelSource"];
    [labelSourceRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: labelSource is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSString *labelFile = snapshot.value;
            self.labelSource = labelFile;            
        }
        
    }];
    
    
    
    //lossFunction
    FIRDatabaseReference *lossFunctionRef = [paramRef child:@"lossFunction"];
    [lossFunctionRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: lossFunction is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSString *lossopt = snapshot.value;
            self.lossFunction = lossopt;
        }
    }];
    
    
    //noiseDistribution
    FIRDatabaseReference *noiseDistributionRef = [paramRef child:@"noiseDistribution"];
    [noiseDistributionRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: noiseDistribution is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSString *noiseopt = snapshot.value;
            self.noiseType = noiseopt;
        }
    }];
    
    //noiseScale
    FIRDatabaseReference *noiseScaleRef = [paramRef child:@"noiseScale"];
    [noiseScaleRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: noiseScale is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *noiseVar = snapshot.value;
            self.noiseVariance = [noiseVar intValue];
        }
    }];
    
    
    //paramIter
    FIRDatabaseReference *paramIterRef = [paramRef child:@"paramIter"];
    [paramIterRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: paramIter is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *paramIterator = snapshot.value;
            self.paramIterator = [paramIterator intValue];
        }
    }];
    
    //nh
    FIRDatabaseReference *nhRef = [paramRef child:@"nh"];
    [nhRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: nh is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *nh = snapshot.value;
            self.nhNumber = [nh intValue];
        }
    }];
    
    //c (naught rate)
    FIRDatabaseReference *cRef = [paramRef child:@"c"];
    [cRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: nh is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *c = snapshot.value;
            self.naught = [c intValue];
        }
    }];
    
    //localUpdateNum
    FIRDatabaseReference *localUNRef = [paramRef child:@"localUpdateNum"];
    [localUNRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: nh is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *localUN = snapshot.value;
            self.localUpdate = [localUN intValue];
        }
    }];
    
    //maxIter
    FIRDatabaseReference *maxIterRef = [paramRef child:@"iteration"];
    [maxIterRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]){
            NSLog(@"Warning: iteration is null object. ");
            exit(EXIT_SUCCESS);
        }else{
            NSNumber *maxiter = snapshot.value;
            self.maxIter = [maxiter intValue];
        }
    }];
}

/**
 Define nh for NN
 */
- (int)nh{
    
    return self.nhNumber;
}

/**
 Define feature size
 */
- (int)D{
    return self.D_featureSize;
}


/**
 Define the number of classes
 */
- (int)K{
    
    return self.K_class;
}


/**
 Defind regularization variable: lambda
 */
- (float)L{
    return self.L_reg;
}


/**
 Define the size of testSample
 */
- (int)N{
    return self.N_test;
}


/**
 Define clientBatchsize
 */
- (int)clientBatchSize{
    return self.clientBatchS;
}


/**
 Define noise scale
 */
- (float)noiseScale{
    return self.noiseVariance;
}

/**
 Define paramIter
 */
- (int)paramIter{
    
    return self.paramIterator;
}

/**
 Define naughRate
 */
- (int)naughtRate{
    
    return self.naught;
}


- (int)iteration{
    
    return self.maxIter;
}


/**
 Define localUpdateNum
 */
- (int)localUpdateNum{
    
    return self.localUpdate;
}


/**
 Define the file name of feature file
 */

-(NSString *) featureSourceName{
    
    NSRange searchDot = [self.featureSource rangeOfString:@"."];
    if(searchDot.location == NSNotFound){
        NSLog(@"Please enter a correct feature filename");
        exit(EXIT_SUCCESS);
    }else{
        NSLog(@"feature file name: %@",[self.featureSource substringToIndex:searchDot.location]);
        return [self.featureSource substringToIndex:searchDot.location];
    }
    
}

/**
 Define the file name of label file
 */
-(NSString *) labelSourceName{

    
    NSRange searchDot = [self.labelSource rangeOfString:@"."];
    if(searchDot.location == NSNotFound){
        NSLog(@"Please enter a correct label filename");
        exit(EXIT_SUCCESS);
    }else{
        NSLog(@"label file name: %@",[self.labelSource substringToIndex:searchDot.location]);
        return [self.labelSource substringToIndex:searchDot.location];
    }
    
}

/**
 Define the type of feature/label file
 */
-(NSString *) sourceType{
    
    
    NSRange searchDot = [self.featureSource rangeOfString:@"."];
    if(searchDot.location == NSNotFound){
        NSLog(@"Please enter a correct data filename");
        exit(EXIT_SUCCESS);
    }else{
        NSLog(@"file type: %@",[self.featureSource substringFromIndex:searchDot.location]);
        return [self.featureSource substringFromIndex:searchDot.location];
    }
    

}

/**
 Choose one of loss function
 */
- (NSInteger) lossOpt{
    
    if([self.lossFunction isEqualToString:@"LogReg"]){
        // opt 1 will call "computeLossLog" method
        NSLog(@"Loss function: LogReg");
        return 1;
    }else if([self.lossFunction isEqualToString:@"Hinge"]){
        //opt 2 will call "computeLossSVM" method (hinge loss)
        NSLog(@"Loss function: Hinge loss");
        return 2;
    }else if([self.lossFunction isEqualToString: @"Softmax"]){
        //opt 3 will call "computeSoftMax" method
        //Note: it can work for more than 2 classes;
        NSLog(@"Loss function: Softmax");

        return 3;
    }else if([self.lossFunction isEqualToString: @"SoftmaxNN"]){
        //opt 3 will call "computeSoftMax" method
        //Note: it can work for more than 2 classes;
        NSLog(@"Loss function: SoftmaxNN");
        return 4;
    }else{
        NSLog(@"Please define the correct loss function.");
        exit(EXIT_SUCCESS);
    }
   
}

/**
 Choose one of noise function
 */
- (NSInteger) noiseOpt{
    
    if([self.noiseType isEqualToString:@"Laplace"]){
        // opt 1 will call "laplace(loss, D)" method
        return 1;
    }else if([self.noiseType isEqualToString:@"Gaussian"]){
        //opt 2 will call "gaussian(loss, D)" method
        return 2;
    }else if([self.noiseType isEqualToString:@"NoNoise"]){
        //opt 3: no noise
        return 3;
    }else{
        NSLog(@"Please define the correct noise function.");
        exit(EXIT_SUCCESS);
    }
   
}

@end