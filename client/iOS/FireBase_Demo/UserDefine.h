//
//  UserDefine.h
//  FireBase_Demo
//
//  Created by yani xie on 6/3/16.
//  Copyright © 2016 Crowd-ML Team. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface UserDefine : NSObject

/**
 Define how many classes
 */
- (int)Classes;

/**
 Define feature size
 */
- (int)featureSize;

/**
 Defind regularization variable: lambda = 0.000001
 */
-(float)RegConstant;

/**
 Define the file name of feature file
 */

-(NSString*) featureSourceName;

/**
 Define the file name of label file
 */
-(NSString*) labelSourceName;

/**
 Define the type of feature/label file
 */
-(NSString*) sourceType;

/**
 Define batch size
 */
-(int)batchSize;


/**
 Defind variance(sigma) for noise function
 */
-(float)variance;


/**
 Choose one of loss function
 */
- (NSInteger) lossOpt;

/**
 Choose one of noise function
 */
- (NSInteger) noiseOpt;

@end