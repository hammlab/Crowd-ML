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
#import "LogReg.h"
#import "HingeLoss.h"
#import "SoftMax.h"
#import "NeuralNetwork.h"

@interface TrainingAlgo : NSObject

/**
 * The number of data points in this training model.
 */
@property (nonatomic, readonly) NSInteger trainingModelSize;

/**
 * The size of feature vectors in this model.
 */
@property (nonatomic, readonly) NSInteger featureSize;

@property (nonatomic, readonly) LogReg *LogRegAlgo;


- (float *) trainModelWithWeight:(float *)w :(int)lossFunction :(int)noiseFunction :(int)class :(int)sbatch :(double)regConstant :(double)variance :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize :(int)getN :(float)L :(int)nh;


/**
 Calculate accuracy for binary/Multi class
 **/
- (float)calculateTrainAccuracyWithWeight:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize :(int)classes :(int)lossFunction :(int)nh;

@end
