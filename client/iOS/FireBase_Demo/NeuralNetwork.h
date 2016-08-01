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

@interface NeuralNetwork : NSObject


- (float *) computeNN :(const float *)trainingFeature :(const float *)trainingLabel :(const float *)w :(long)D :(int)classes :(double) regConstant :(float) L :(int)nh :(int)clientBatchSize;

/**
 Calculate accuracy for multi(10) class
 **/
- (float)calculateTrainAccuracyWithWeightNN:(float *)w :(NSString *)labelName :(NSString *)featureName :(NSString *)fileType :(int) DFeatureSize :(int)classes :(long)Ntest :(long)featureSize :(int)nh;

/**
 Read label file
 **/
- (float *) readTrainingLabelFile: (NSString *)labelSource : (NSString *) filetype :(long)trainingModelSize;

/**
 Read feature file
 **/
- (float **) readTrainingFeatureFile: (NSString *)FeatureSource : (NSString *) filetype : (int) DfeatureSize :(long)trainingModelSize;


- (NSString *) readFileContentWithPath: (NSString *)filePath
                                  type: (NSString *)fileType
                              encoding: (NSStringEncoding)encoding;




@end
