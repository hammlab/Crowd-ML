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

#import <UIKit/UIKit.h>


@interface ViewController : UIViewController

@property (nonatomic, strong) IBOutlet UIButton *trainButton;
@property (nonatomic, strong) IBOutlet UILabel *trainErrorLabel;
@property (nonatomic, strong) IBOutlet UITextField *trainTimesField;

- (IBAction) SetUpButtonTapped:(UIButton *)sender;
- (IBAction) trainingButtonTapped:(UIButton *)sender;
- (IBAction) calculateTrainErrorTapped:(UIButton *)sender;

@end

