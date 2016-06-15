//
//  trainingLogReg.h
//  FireBase_iOS_Client_Demo
//
//  Created by JINJIN SHAO and Yani Xie on 3/7/16.
//  Copyright © 2016 Crowd-ML Team. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface TrainingAlgo : NSObject

/**
 * The number of data points in this training model.
 */
@property (nonatomic, readonly) NSInteger trainingModelSize;

/**
 * The size of feature vectors in this model.
 */
@property (nonatomic, readonly) NSInteger featureSize;

- (void) trainModelWithSelfTest;
- (float *) trainModelWithWeight: (float *)w;
- (float) calculateTrainAccuracyWithWeight: (float *)w;

@end
