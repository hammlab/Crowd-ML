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
 *  The Firebase reference this viewcontroller instance uses.
 */
@property (nonatomic, strong) FIRDatabaseReference *paramRef;

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
 *  Record the weightIter under "user"
 */
@property (nonatomic) NSString *wIterU;

/**
 *  Record the iteration under "weight"
 */
@property (nonatomic) NSString *wIterTW;

/**
 *  For local update.
 */
@property (nonatomic) BOOL localTraining;


/**
 *  For local update, counting the number of times of training
 */
@property (nonatomic) int numOfTraining;


@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.rootRef = [[FIRDatabase database] reference];
    self.paramRef = [self.rootRef child:@"parameters"];
    self.trainModel = [[TrainingAlgo alloc] init];
    self.userParam = [[UserDefine alloc] init];
    [self.userParam Initialize:self.paramRef];
    self.countTrainingTime = 0;
    self.isTrainOnce = true;
    self.localTraining = true;
    
    
    
    // Do any additional setup after loading the view, typically from a nib.
    
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    
    NSLog(@"View Appeared");
    
    [self login];
    
}

- (void) login{
    UIAlertController *LoginController = [UIAlertController
                                          alertControllerWithTitle:@"Login/Sign up"
                                          message:@"Enter your email and password"
                                          preferredStyle:UIAlertControllerStyleAlert];
    
    [LoginController addTextFieldWithConfigurationHandler:^(UITextField *textField)
     {
         textField.placeholder = NSLocalizedString(@"Email", @"Login");
         [textField addTarget:self
                       action:@selector(alertTextFieldDidChange:)
             forControlEvents:UIControlEventEditingChanged];
         
     }];
    
    [LoginController addTextFieldWithConfigurationHandler:^(UITextField *textField)
     {
         textField.placeholder = NSLocalizedString(@"Password", @"Password");
         textField.secureTextEntry = YES;
         [textField addTarget:self
                       action:@selector(alertTextFieldDidChange:)
             forControlEvents:UIControlEventEditingChanged];
     }];
    
    
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
    
    okAction.enabled = NO;
    
    [LoginController addAction:okAction];
    [self presentViewController:LoginController animated:YES completion:nil];
    
    
    
}

//Check valid login information
- (void)alertTextFieldDidChange:(UITextField *)sender
{
    UIAlertController *alertController = (UIAlertController *)self.presentedViewController;
    if (alertController)
    {
        UITextField *email = alertController.textFields.firstObject;
        UITextField *password = alertController.textFields.lastObject;
        UIAlertAction *okAction = alertController.actions.lastObject;
        okAction.enabled = password.text.length > 5 && email.text.length > 5;
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
    
    if(self.logginUID == NULL){
        [self login];
    }
    
}
/*
  Manually clear entries under "users".
 */
- (IBAction)calculateTrainErrorTapped:(UIButton *)sender
{
    
    FIRDatabaseReference *userInfo = [self.rootRef child:@"users"];
    [userInfo removeValue];
    [self.trainErrorLabel setText:@"User infomation is cleared."];
    
    /**
     *  Download the current weight from firebase. Calculate the error rate based on this weight.
     */
    //[self loadWeightFromFireBaseAndCalculateTrainError];


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
            
            
            NSString *labelName = [self.userParam labelSourceName];
            NSString *featureName = [self.userParam featureSourceName];
            NSString *fileType = [self.userParam sourceType];
            int DFeatureSize = [self.userParam D];
            int classes = [self.userParam K];
            int lossFunction = (int)[self.userParam lossOpt];
            int nh = [self.userParam nh];
            
            double accuracy = [self.trainModel calculateTrainAccuracyWithWeight:w :labelName :featureName :fileType :DFeatureSize :classes :lossFunction :nh];
            
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
            
            FIRDatabaseReference *paraIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"paramIter"];
            NSNumber *paramNum = [NSNumber numberWithInt:[self.userParam paramIter] ];
            
            [gradIterRef setValue:graditer];
            [gradlossRef setValue:gradlossDict];
            [infoRef setValue:infoDict];
            [paraIterRef setValue: paramNum];
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
            
            NSString *graditer = @"1";FIRDatabaseReference *paraIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"paramIter"];
            NSNumber *paramNum = [NSNumber numberWithInt:[self.userParam paramIter] ];
            
            
            [gradIterRef setValue:graditer];
            [gradlossRef setValue:gradlossDict];
            [infoRef setValue:infoDict];
            [paraIterRef setValue:paramNum];
            [self UpdateWeightIter];
            
            
            UIAlertController *checkedAlert = [UIAlertController alertControllerWithTitle:@"Warning" message:@"No user info exists in Firebase. New user info set up. Please tap this button again and begin to train your model." preferredStyle:UIAlertControllerStyleAlert];
            
            UIAlertAction *deaultAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
            }];
            
            [checkedAlert addAction:deaultAction];
            [self presentViewController:checkedAlert animated:YES completion:nil];
            
            
        } else {
            [self GetWeightIterUnderUserAndTrainingWeights];
            NSNumber *response = snapshot.value;
            
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
        
        
        if(self.localTraining){
            [self localTrainModelWithWeight:w];
        }else{
            [self trainModelAndUploadGradLossWithWeight:w];
        }
        
        free(w);
    }];
    [self UpdateWeightIter];
    
}

- (void)trainModelAndUploadGradLossWithWeight: (float *) w {
    
    float *gradloss = NULL;
    
    int lossType = (int)[self.userParam lossOpt];
    int noiseType = (int)[self.userParam noiseOpt];
    int class = [self.userParam K];
    int batchsize= [self.userParam clientBatchSize];
    double regConstant = [self.userParam L];
    double variance = [self.userParam noiseScale];
    NSString *labelName = [self.userParam labelSourceName];
    NSString *featureName = [self.userParam featureSourceName];
    NSString *fileType = [self.userParam sourceType];
    int DFeatureSize = [self.userParam D];
    int N = [self.userParam N];
    float L = [self.userParam L];
    int nh = [self.userParam nh];
    int localUpdateNum = 0;
    int naughtRate = 10;
    
    int length = DFeatureSize;
    if(lossType == 3){
        length = DFeatureSize * class;
    }else if(lossType == 4){
        length = (DFeatureSize + 1) * nh + (nh + 1) * nh + (nh + 1) * class;
    }
    
    if(localUpdateNum <= 0){
        gradloss = [self.trainModel trainModelWithWeight:w :lossType :noiseType :class :batchsize :regConstant :variance :labelName :featureName :fileType :DFeatureSize :N :L :nh];

    }else{
        for(int i = 0; i < localUpdateNum; i++){
            gradloss = [self.trainModel trainModelWithWeight:w :lossType :noiseType :class :batchsize :regConstant :variance :labelName :featureName :fileType :DFeatureSize :N :L :nh];
            
            for(int k = 0; k < length; k++){
                *(w + k) = *(w + k) - naughtRate/sqrtf(i*localUpdateNum) * *(gradloss + k);
            }
            gradloss = NULL;
        }
    }
    
    self.trainFeatureSize = self.trainModel.featureSize;
    if(lossType == 3){
        self.trainFeatureSize = self.trainModel.featureSize * class;
    }else if(lossType == 4){
        self.trainFeatureSize = (self.trainModel.featureSize + 1) * nh + (nh + 1) * nh + (nh + 1) * class;
    }
    
    NSLog(@"Get gradient");
    
    
    if(w == NULL) {
        [self uploadNewWeightToFireBase];
    }else if(localUpdateNum <= 0) {
        [self GetWeightIterUnderUserAndTrainingWeights];
        if(![self.wIterU isEqualToString: self.wIterTW]){
            [self UpdateWeightIter];
        }
        [self uploadGradLossToFireBase:gradloss];
    }else if(localUpdateNum > 0){
        
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

- (void) uploadLocalWeightToFireBase: (float *)w
{
    
    //Update gradloss
    NSDictionary *wDict = [[NSMutableDictionary alloc] initWithCapacity:self.trainFeatureSize];
    for(int i = 0; i < self.trainFeatureSize; i++) {
        
        [wDict setValue: [NSNumber numberWithFloat: *(w + i)] forKey:[NSString stringWithFormat:@"%d", i]];
    }
    
    FIRDatabaseReference *wRef = [[self.rootRef child:@"trainingWeights"] child:@"weights"];
    
    //Change readyByServer to false.
    NSNumber *infoDict  = [NSNumber numberWithBool:NO];
    
    FIRDatabaseReference *infoRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradientProcessed"];
    
    FIRDatabaseReference *gradIterRef = [[[self.rootRef child:@"users" ] child:self.logginUID] child:@"gradIter"];
    
    
    NSLog(@"Uploading local weights..");
    [wRef setValue:wDict];
    
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
    
    NSLog(@"Uploaded local weights.");
    
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
    
    if(self.localTraining){
        [self BeginLocalTrain:(int)[tempTimes integerValue]];
        
    }else{
    
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


-(void) BeginLocalTrain :(int)numOfTraining
{
    //set up param and load initial weight
    self.numOfTraining = numOfTraining;
    [self loadWeightFromFireBaseAndTrain];
}

-(void)localTrainModelWithWeight:(float*) w{
    //set up param
    int lossType = (int)[self.userParam lossOpt];
    int noiseType = (int)[self.userParam noiseOpt];
    int class = [self.userParam K];
    int batchsize= [self.userParam clientBatchSize];
    double regConstant = [self.userParam L];
    double noiseScale = [self.userParam noiseScale];
    NSString *labelName = [self.userParam labelSourceName];
    NSString *featureName = [self.userParam featureSourceName];
    NSString *fileType = [self.userParam sourceType];
    int DFeatureSize = [self.userParam D];
    int N = [self.userParam N];
    float L = [self.userParam L];
    int nh = 80;
    int naughtRate = 10;
    int localUpdateNum = 0;
    
    int length = DFeatureSize;
    if(lossType == 3){
        length = DFeatureSize * class;
    }else if(lossType == 4){
        length = (DFeatureSize + 1) * nh + (nh + 1) * nh + (nh + 1) * class;
    }
    
    float *gradloss = NULL;
    
    for(int i = 1; i <= self.numOfTraining; i++){
        if(localUpdateNum<=0){
             gradloss = [self.trainModel trainModelWithWeight:w :lossType :noiseType :class :batchsize :regConstant :noiseScale :labelName :featureName :fileType :DFeatureSize :N :L :nh];
            NSLog(@"Complete calculate graidents locally.");

        }else{
            for(int cnt = 0; cnt < localUpdateNum; cnt++){
                gradloss = [self.trainModel trainModelWithWeight:w :lossType :noiseType :class :batchsize :regConstant :noiseScale :labelName :featureName :fileType :DFeatureSize :N :L :nh];
                NSLog(@"Complete calculate graidents locally.");
                

                for(int k = 0; k < length; k++){
                    *(w + k) = *(w + k) - naughtRate/sqrtf(i*localUpdateNum) * *(gradloss + k);
                }
                NSLog(@"Complete update weights locally for localUpdateNum >0.");
            }
        }
       
        if(localUpdateNum <= 0){
            for(int k = 0; k < length; k++){
                *(w + k) = *(w + k) - naughtRate/sqrtf(i)* *(gradloss + k);
                //NSLog(@"%d: %f, grad: %f, sqrt: %f",k, *(w+k), *(gradloss + k),sqrtf(i));
            }
            NSLog(@"Complete update weights locally.");

        }
        NSLog(@"Iteration: %d",i);
        [self.trainModel calculateTrainAccuracyWithWeight:w :labelName :featureName :fileType :DFeatureSize :class :lossType :nh];
        
        //Reset gradients
        gradloss = NULL;
    }

    free(gradloss);

}




@end