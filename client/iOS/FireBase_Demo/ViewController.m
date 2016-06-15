//
//  ViewController.m
//  FireBase_iOS_Client_Demo
//
//  Created by JINJIN SHAO and Yani Xie on 3/7/16.
//  Copyright © 2016 Crowd-ML Team. All rights reserved.
//

#import "ViewController.h"
#import "TrainingAlgo.h"
#import "UserDefine.h"
@import Firebase;


@interface ViewController ()

/**
 *  The Firebase reference this viewcontroller instance uses.
 */
@property (nonatomic, strong) FIRDatabaseReference *rootRef;

/**
 *  The user name of that user who has already signed in.
 *  For locating and accessing the place in firebase to upload GradLoss.
 */
@property (nonatomic, strong) NSString *logginUID;

/**
 *  The training model this viewcontroller instance links to.
 */
@property (nonatomic, strong) TrainingAlgo *trainModel;

/**
 *  The User defined parameters this viewcontroller instance links to.
 */
@property (nonatomic, strong) UserDefine *userParam;


/**
 *  The size of feature vectors in the model.
 */
@property (nonatomic) long trainFeatureSize;

/**
 *  The disabled button.
 *  After tap the training button, before this viewcontroller upload GradLoss, disabled training button.
 */
@property (nonatomic, strong) UIButton *disabledButton;

/**
 *  Count the training times when do multiple training.
 */
@property (nonatomic) NSInteger countTrainingTime;

/**
 *  Check if just train once.
 */
@property (nonatomic) BOOL isTrainOnce;

/**
 *  Check if login has problem.
 */
@property (nonatomic) BOOL LoginError;


/**
 *  Record the weightIter under "user"
 */
@property (nonatomic) NSString *wIterU;

/**
 *  Record the iteration under "weight"
 */
@property (nonatomic) NSString *wIterTW;


@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.rootRef = [[FIRDatabase database] reference];
    self.trainModel = [[TrainingAlgo alloc] init];
    self.userParam = [[UserDefine alloc] init];
    self.countTrainingTime = 0;
    self.isTrainOnce = true;
    
    
    
    // Do any additional setup after loading the view, typically from a nib.
    
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    
    NSLog(@"View Appeared");
    
    [self login];
}

- (void) login{
    UIAlertController *LoginController = [UIAlertController
                                          alertControllerWithTitle:@"Login"
                                          message:@"Enter your email and password"
                                          preferredStyle:UIAlertControllerStyleAlert];
    
    
    UIAlertAction *okAction = [UIAlertAction
                               actionWithTitle:NSLocalizedString(@"OK", @"OK action")
                               style:UIAlertActionStyleDefault
                               handler:^(UIAlertAction *action)
                               {
                                   UITextField *login = LoginController.textFields.firstObject;
                                   UITextField *password = LoginController.textFields.lastObject;
                                   
                                   NSString *userNames = login.text;
                                   NSString *setPassword = password.text;
                                   
                                   
                                   [[FIRAuth auth] signInWithEmail:userNames password:setPassword completion:^(FIRUser *authData, NSError *error) {
                                       
                                       if (error) {
                                           NSLog(@"Error: %@", [error debugDescription]);
                                           [self createUserInFireBaseWithUserName:userNames Password:setPassword];
                                           
                                       } else {
                                           self.logginUID = authData.uid;
                                       }
                                       
                                       
                                   }];
                                   
                                   
                               }];
    
    /*
    UITextField *login = LoginController.textFields.firstObject;
    UITextField *password = LoginController.textFields.lastObject;
    if(login.text.length > 5 && password.text.length > 5){
            okAction.enabled = YES;
    }else{
        okAction.enabled = NO;
    }
     */
    [LoginController addTextFieldWithConfigurationHandler:^(UITextField *textField)
     {
         textField.placeholder = NSLocalizedString(@"Email", @"Login");
         
     }];
    
    [LoginController addTextFieldWithConfigurationHandler:^(UITextField *textField)
     {
         textField.placeholder = NSLocalizedString(@"Password", @"Password");
         textField.secureTextEntry = YES;
         [textField addTarget:self
                       action:@selector(alertTextFieldDidChange:)
             forControlEvents:UIControlEventEditingChanged];
     }];

    
    [LoginController addAction:okAction];
    [self presentViewController:LoginController animated:YES completion:nil];
    

}

- (void)alertTextFieldDidChange:(UITextField *)sender
{
    UIAlertController *alertController = (UIAlertController *)self.presentedViewController;
    if (alertController)
    {
        UITextField *email = alertController.textFields.firstObject;
        UIAlertAction *okAction = alertController.actions.lastObject;
        okAction.enabled = email.text.length > 5 ;
    }
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void) createUserInFireBaseWithUserName: (NSString *) userName
                                 Password: (NSString *) password
{

    [[FIRAuth auth] createUserWithEmail:userName password:password completion:^(FIRUser *result, NSError *error) {
        
        if(error)
        {
            NSLog(@"Error: %@", [error debugDescription]);
            UIAlertController *errorControl = [UIAlertController
                                                alertControllerWithTitle:@"Error"
                                                message:[error debugDescription]
                                                preferredStyle:UIAlertControllerStyleAlert];
            UIAlertAction *acceptAction = [UIAlertAction actionWithTitle:@"Start again" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
            }];
            
            
            [errorControl addAction:acceptAction];
            [self presentViewController:errorControl animated:YES completion:nil];
            
            self.LoginError = true;
        }
        else
        {
            self.logginUID = result.uid;
            NSLog(@"Successfully Create User Account with uid: %@ ", result.uid);
            
            UIAlertController *createNewUser = [UIAlertController
                                                alertControllerWithTitle:@"Welcome"
                                                message:@"A new account is created for you!"
                                                preferredStyle:UIAlertControllerStyleAlert];
            UIAlertAction *acceptAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
            }];
            
            
            [createNewUser addAction:acceptAction];
            [self presentViewController:createNewUser animated:YES completion:nil];
            
        }
    }];
    if(self.LoginError){
        [self login];
    }
    
}

/**
 *  Download the current weight from firebase. Calculate the error rate based on this weight.
 */
- (IBAction)calculateTrainErrorTapped:(UIButton *)sender
{
    [self loadWeightFromFireBaseAndCalculateTrainError];
    
}

- (void)loadWeightFromFireBaseAndCalculateTrainError {
    FIRDatabaseReference *weightRef1 = [self.rootRef child:@"trainingWeights"];
    FIRDatabaseReference *weightRef = [weightRef1 child:@"weights"];

    
    [weightRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        
        float *w;
        
        if(snapshot.value == [NSNull null]) {
            
            NSLog(@"Warning: Weight is null object. ");
            
            w = NULL;
            
            [self.trainErrorLabel setText:@"Weight is empty in FireBase. "];
            
        } else {
            
            NSArray *wArray = snapshot.value;
            int wLen = (int)[wArray count];
            NSLog(@"Get a new Weight obejct of length: %d", wLen);
            
            w = (float *) malloc(wLen * sizeof(float));
            
            for(int i = 0; i < wLen; i++) {
                *(w + i) = [[wArray objectAtIndex:i] floatValue];
            }
            
            float accuracy = [self.trainModel calculateTrainAccuracyWithWeight:w];
            
            [self.trainErrorLabel setText:[NSString stringWithFormat:@"%.3f", accuracy]];
            
            free(w);
        }
    }];
}

/**
 *  Set up user's information in firebase database.
 */
- (IBAction)trainingOnceButtonTapped:(UIButton *)sender
{
    [[[[self.rootRef child:@"users"]  child:self.logginUID] child:@"gradientProcessed"] observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        
        
        if(snapshot.value == [NSNull null]) {
            /**
             *  set up account's info.
             */
            
            NSDictionary *gradlossDict = [NSDictionary dictionary];
            for(int i = 0; i < self.trainFeatureSize; i++) {
                [gradlossDict setValue: [NSNumber numberWithFloat:0.0] forKey:[NSString stringWithFormat:@"%d", i]];
            }
            
            
            FIRDatabaseReference *gradlossRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradients"];
            
            
            NSNumber *infoDict  = [NSNumber numberWithBool:YES];
            
            FIRDatabaseReference *infoRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradientProcessed"];
            
            FIRDatabaseReference *gradIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradIter"];
            NSString *graditer = @"1";
            
            [gradIterRef setValue:graditer];
            [gradlossRef setValue:gradlossDict];
            [infoRef setValue:infoDict];
            [self UpdateWeightIter];
            
            
            UIAlertController *checkedAlert = [UIAlertController alertControllerWithTitle:@"Message" message:@"No user info exists in Firebase. New user info set up. " preferredStyle:UIAlertControllerStyleAlert];
            
            UIAlertAction *deaultAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
            }];
            
            [checkedAlert addAction:deaultAction];
            [self presentViewController:checkedAlert animated:YES completion:nil];
            
        }else{
            //If a user's info is setup, prompt out a message
            UIAlertController *checkedAlert = [UIAlertController alertControllerWithTitle:@"Error" message:@"A repository is already set up. " preferredStyle:UIAlertControllerStyleAlert];
            
            UIAlertAction *deaultAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
            }];
            
            [checkedAlert addAction:deaultAction];
            [self presentViewController:checkedAlert animated:YES completion:nil];
            [self UpdateWeightIter];
        }
    }];
    
    //Diable setup button
    [sender setEnabled:false];
    self.disabledButton = sender;
    
    [self.trainErrorLabel setText:@"Hello"];
}

- (void) loadReadByServerFromFirebase
{
   
    [[[[self.rootRef child:@"users"]  child:self.logginUID] child:@"gradientProcessed"] observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        
        
        if(snapshot.value == [NSNull null]) {
            /**
             *  set up account's info.
             */
            
            NSDictionary *gradlossDict = [NSDictionary dictionary];
            for(int i = 0; i < self.trainFeatureSize; i++) {
                [gradlossDict setValue: [NSNumber numberWithFloat:0.0] forKey:[NSString stringWithFormat:@"%d", i]];
            }
            
            FIRDatabaseReference *gradlossRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradients"];
            
            
            NSNumber *infoDict  = [NSNumber numberWithBool:YES];
            
            FIRDatabaseReference *infoRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradientProcessed"];
            
            FIRDatabaseReference *gradIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradIter"];
            NSString *graditer = @"1";
            
            [gradIterRef setValue:graditer];
            [gradlossRef setValue:gradlossDict];
            [infoRef setValue:infoDict];
            [self UpdateWeightIter];

            

            UIAlertController *checkedAlert = [UIAlertController alertControllerWithTitle:@"Warning" message:@"No user info exists in Firebase. New user info set up. Please tap this button again and begin to train your model." preferredStyle:UIAlertControllerStyleAlert];
            
            UIAlertAction *deaultAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
            }];
            
            [checkedAlert addAction:deaultAction];
            [self presentViewController:checkedAlert animated:YES completion:nil];

            
        } else {
            [self GetWeightIterUnderUserAndTrainingWeights];
            NSNumber *response = snapshot.value;
            
            /**
            if(![self.wIterU isEqualToString:self.wIterTW] && !self.isTrainAgain){
                NSLog(@"s1");
                [self UpdateWeightIter];
                [self loadReadByServerFromFirebase];
                
            }else if(self.isTrainAgain){
                NSLog(@"s2");
                self.isTrainAgain = false;
                [self UpdateWeightIter];
                [self loadWeightFromFireBaseAndTrain];
                
            }else */
            if([response integerValue]== 0){
                
                if(![self.wIterU isEqualToString:self.wIterTW]){
                   
                    [self UpdateWeightIter];
                    [self loadReadByServerFromFirebase];
                }
                //Multiple training
                if(self.countTrainingTime > 0 && !self.isTrainOnce) {
                    
                    [NSThread sleepForTimeInterval:.3];
                    [self tryTrainingAgain];
                }else {
                    UIAlertController *checkedAlert = [UIAlertController alertControllerWithTitle:@"Warning" message:@"Old gradients has not been checked out. " preferredStyle:UIAlertControllerStyleAlert];
                    
                    UIAlertAction *deaultAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
                    }];
                    
                    [checkedAlert addAction:deaultAction];
                    [self presentViewController:checkedAlert animated:YES completion:nil];
                }

            }else{
                
                [self loadWeightFromFireBaseAndTrain];

            }

            
            /**
            if((int)response == 0)
            {
                //Multiple training
                if(self.countTrainingTime > 0 && !self.isTrainOnce) {
                    //[NSThread sleepForTimeInterval:.3];
                    
                    [self UpdateWeightIter];
                    [self tryTrainingAgain];
                }else {
                    UIAlertController *checkedAlert = [UIAlertController alertControllerWithTitle:@"Warning" message:@"Old gradients has not been checked out. " preferredStyle:UIAlertControllerStyleAlert];
                    
                    UIAlertAction *deaultAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
                    }];
                    
                    [checkedAlert addAction:deaultAction];
                    [self presentViewController:checkedAlert animated:YES completion:nil];
                }
            }else if([self.wIterU isEqualToString:self.wIterTW]){
                [self UpdateWeightIter];
            }
            else{
                [self loadWeightFromFireBaseAndTrain];
            }
             */
        
        }
    }];
   [self UpdateWeightIter];
}

- (void) tryTrainingAgain
{
    [self UpdateWeightIter];
    [self loadReadByServerFromFirebase];
}



- (void)loadWeightFromFireBaseAndTrain
{
    
    FIRDatabaseReference *weightRef1 = [self.rootRef child:@"trainingWeights"];
    FIRDatabaseReference *weightRef = [weightRef1 child:@"weights"];

    
    NSLog(@"Start Block");
    
    [weightRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        
        float *w;
        
        if(snapshot.value == [NSNull null]) {
            
            NSLog(@"Warning: Weight is null object. ");
            
            w = NULL;
            
        } else {
            
            //Get current weights on firebase database.
            NSArray *wArray = snapshot.value;
            int wLen = (int)[wArray count];
            NSLog(@"Get a new Weight obejct of length: %d", wLen);
            
            w = (float *) malloc(wLen * sizeof(float));
            
            for(int i = 0; i < wLen; i++) {
                *(w + i) = [[wArray objectAtIndex:i] floatValue];
            }
            
        }
        
        NSLog(@"Complete load Weight. ");
        
        
        
        [self trainModelAndUploadGradLossWithWeight:w];
        
        free(w);
    }];
    [self UpdateWeightIter];
  
}

- (void)trainModelAndUploadGradLossWithWeight: (float *) w {
    
    float *gradloss;
    gradloss = [self.trainModel trainModelWithWeight:w];
    self.trainFeatureSize = self.trainModel.featureSize;
    if([self.userParam Classes] > 2){
        self.trainFeatureSize = self.trainModel.featureSize * [self.userParam Classes];
    }
    
    NSLog(@"Get gradient");
   
   
    if(w == NULL) {
        [self uploadNewWeightToFireBase];
    }else {
        [self GetWeightIterUnderUserAndTrainingWeights];
        if(![self.wIterU isEqualToString: self.wIterTW]){
            [self UpdateWeightIter];
        }
        [self uploadGradLossToFireBase:gradloss];
    }
   
    
    free(gradloss);
    
    self.countTrainingTime -= 1;
    
    //Multiple training
    if(self.countTrainingTime > 0) {
        [self loadReadByServerFromFirebase];
    }
    [self UpdateWeightIter];
    
}

- (void) uploadGradLossToFireBase: (float *)gradloss
{
  
    //Update gradloss
    NSDictionary *gradlossDict = [[NSMutableDictionary alloc] initWithCapacity:self.trainFeatureSize];
    
    for(int i = 0; i < self.trainFeatureSize; i++) {
        [gradlossDict setValue: [NSNumber numberWithFloat: *(gradloss + i)] forKey:[NSString stringWithFormat:@"%d", i]];
    }
    
    FIRDatabaseReference *gradlossRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradients"];
    
    //Change readyByServer to false.
    NSNumber *infoDict  = [NSNumber numberWithBool:NO];
    
    FIRDatabaseReference *infoRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradientProcessed"];
    
    FIRDatabaseReference *gradIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradIter"];
    
    
    NSLog(@"Uploading gradients..");
    [gradlossRef setValue:gradlossDict];
    
    [gradIterRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        
        
        if(snapshot.value == [NSNull null]) {
            
            NSLog(@"Warning: gradIter is null object. ");
            
        } else {
            
            //Update gradIter on firebase database.
            NSString *iternum = snapshot.value;
            int graditrnum = [iternum intValue] + 1;
            NSString *graditr = [NSString stringWithFormat:@"%i", graditrnum];
            [gradIterRef setValue:graditr];
        }
        
        NSLog(@"Complete update gradIter. ");
        
    }];

    NSLog(@"Uploaded gradients");
    
    [infoRef setValue:infoDict];
    
    [self.disabledButton setEnabled:true];
    [self UpdateWeightIter];
        
}

- (void) uploadNewWeightToFireBase
{
    //Set new weight to 0
    NSDictionary *wDict = [[NSMutableDictionary alloc] initWithCapacity:self.trainFeatureSize];
    
    for(int i = 0; i < self.trainFeatureSize; i++) {
        [wDict setValue: [NSNumber numberWithFloat:0.0] forKey:[NSString stringWithFormat:@"%d", i]];
    }
    
    FIRDatabaseReference *weightRef = [[self.rootRef child:@"trainingWeights"] child:@"weights"];
    
    [weightRef setValue:wDict];
    
    [self.disabledButton setEnabled:true];
    
}

/**
 *  Train multiple times.
 */
- (IBAction)trainingButtonTapped:(UIButton *)sender
{
    
    NSString *tempTimes = self.trainTimesField.text;
    
    
    int totalTrainTimes = 0;
    if([tempTimes length] > 0)
    {
        totalTrainTimes = (int)[tempTimes integerValue];
        
        self.countTrainingTime = totalTrainTimes;
        self.isTrainOnce= false;
        
        [self UpdateWeightIter];
        //Begin to train multiple times.
        [self loadReadByServerFromFirebase];
        
        [sender setEnabled:true];
    }

    
}

-(void) UpdateWeightIter
{
    //copy "iteration" under "Weight" to "weightIter" under "Users"
    FIRDatabaseReference *weightIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"weightIter"];
    
    FIRDatabaseReference *weightRef1 = [self.rootRef child:@"trainingWeights"];
    FIRDatabaseReference *weightRef = [weightRef1 child:@"iteration"];
    
    [weightRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        if(snapshot.value == [NSNull null]) {
            
            NSLog(@"Warning: iteration is null object. ");
        } else {
            
            //Get current weightIter on firebase database.
            NSString *weightnum = snapshot.value;
            [weightIterRef setValue: weightnum];
        }
    }];
}

-(void) GetWeightIterUnderUserAndTrainingWeights
{
    //copy "iteration" under "Weight" to "weightIter" under "Users"
    //Note: when it finishes a round of training, "weightIter" is 1 less than "iteration". When you continue
    // to do another round of training, it will update it.
    FIRDatabaseReference *weightIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"weightIter"];
    [weightIterRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        self.wIterU = [NSString stringWithFormat:@"%@",snapshot.value];
    }];
    
    FIRDatabaseReference *weightRef1 = [self.rootRef child:@"trainingWeights"];
    FIRDatabaseReference *weightRef = [weightRef1 child:@"iteration"];
    
    [weightRef observeSingleEventOfType:FIRDataEventTypeValue withBlock:^(FIRDataSnapshot *snapshot) {
        self.wIterTW = [NSString stringWithFormat:@"%@", snapshot.value];
    }];
}



@end
