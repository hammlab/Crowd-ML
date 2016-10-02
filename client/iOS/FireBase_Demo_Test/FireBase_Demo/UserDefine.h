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

#import <Foundation/Foundation.h>
@import Firebase;

@interface UserDefine : NSObject
/**
 *  The Firebase reference this userDefine instance uses.
 */

@property (nonatomic, readonly) int D_featureSize;
@property (nonatomic, readonly) int K_class;
@property (nonatomic, readonly) float L_reg;
@property (nonatomic, readonly) int N_test;
@property (nonatomic, readonly) int clientBatchS;
@property (nonatomic, readonly) NSString *featureSource;
@property (nonatomic, readonly) NSString *labelSource;
@property (nonatomic, readonly) NSString *lossFunction;
@property (nonatomic, readonly) NSString *noiseType;
@property (nonatomic, readonly) float noiseVariance;
@property (nonatomic, readonly) int paramIterator;
@property (nonatomic, readonly) int nhNumber;
@property (nonatomic, readonly) int naught;
@property (nonatomic, readonly) int localUpdate;
@property (nonatomic, readonly) int maxIter;


-(void)Initialize:(FIRDatabaseReference*) paramRef;

/**
 Define nh for NN
 */
- (int)nh;

/**
 Define feature size
 */
- (int)D;


/**
 Define the number of classes
 */
- (int)K;

/**
 Defind regularization variable: lambda
 */
- (float)L;

/**
 Define the size of testSample
 */
- (int)N;


/**
 Define clientBatchsize
 */
- (int)clientBatchSize;

/**
 Define noise scale
 */
- (float)noiseScale;

/**
 Define paramIter
 */
- (int)paramIter;

/**
 Define naughRate
 */
- (int)naughtRate;

/**
 Define localUpdateNum
 */
- (int)localUpdateNum;


- (int)iteration;

/**
 Define the file name of feature file
 */

-(NSString *) featureSourceName;
/**
 Define the file name of label file
 */
-(NSString *) labelSourceName;
/**
 Define the type of feature/label file
 */
-(NSString *) sourceType;
/**
 Choose one of loss function
 */
- (NSInteger) lossOpt;

/**
 Choose one of noise function
 */
- (NSInteger) noiseOpt;



@end