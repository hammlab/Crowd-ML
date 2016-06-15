//
//  ViewController.h
//  FireBase_iOS_Client_Demo
//
//  Created by JINJIN SHAO and Yani Xie on 3/7/16.
//  Copyright © 2016 Crowd-ML Team. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController

@property (nonatomic, strong) IBOutlet UIButton *trainButton;
@property (nonatomic, strong) IBOutlet UILabel *trainErrorLabel;
@property (nonatomic, strong) IBOutlet UITextField *trainTimesField;

- (IBAction) trainingOnceButtonTapped:(UIButton *)sender;
- (IBAction) trainingButtonTapped:(UIButton *)sender;
- (IBAction) calculateTrainErrorTapped:(UIButton *)sender;

@end

