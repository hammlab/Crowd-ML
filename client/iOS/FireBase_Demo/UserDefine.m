//
//  UserDefine.m
//  FireBase_Demo
//
//  Created by yani xie on 6/3/16.
//  Copyright © 2016 Crowd-ML Team. All rights reserved.
//

#import "UserDefine.h"
#import <Foundation/Foundation.h>
@interface UserDefine()
@end

@implementation UserDefine


/**
 Define how many classes
 */
- (int)Classes{
    return 2;
}

/**
 Define feature size
 */
- (int)featureSize{
    return 785;
}


/**
 Defind regularization variable: lambda
 */
-(float)RegConstant{
    float lambda = 0.000001;
    return lambda;
}

/**
 Define the file name of feature file
 */

-(NSString *) featureSourceName{
    NSString *name = @"trainingFeature";
    //NSString *name = @"MNISTtrainingimages";
    return name;
}

/**
 Define the file name of label file
 */
-(NSString *) labelSourceName{
    NSString *name = @"trainingLabel";
    //NSString *name = @"MNISTtraininglabels";
    return name;
}

/**
 Define the type of feature/label file
 */
-(NSString *) sourceType{
    NSString *type = @"dat";
    return type;
}

/**
 Define batch size
 */
-(int)batchSize{
    return 10;
}


/**
 Defind variance(sigma) for noise function
 */
-(float)variance{
    return 1.0;
}


/**
 Choose one of loss function
 */
- (NSInteger) lossOpt{
    // opt 1 will call "computeLossLog" method
    //NSInteger opt = 1;
    
    //opt 2 will call "computeLossSVM" method (hinge loss)
    NSInteger opt = 2;
    
    //opt 3 will call "computeSoftMax" method
    //Note: it can work for more than 2 classes;
    //NSInteger opt = 3;
    
    return opt;
}

/**
 Choose one of noise function
 */
- (NSInteger) noiseOpt{
    // opt 1 will call "laplace(loss, D)" method
    //NSInteger opt = 1;
    
    //opt 2 will call "gaussian(loss, D)" method
    //NSInteger opt = 2;
    
    //opt 3: no noise
    NSInteger opt = 3;
    
    
    return opt;
}
@end