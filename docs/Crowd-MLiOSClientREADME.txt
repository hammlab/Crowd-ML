//
//  iOS Client README
//
//  Created by JINJIN SHAO on 5/3/16.
//  Copyright © 2016 Crowd-ML Team. All rights reserved.
//

Crowd-ML iOS Client README


What the code does:

        The code here can be mainly divided into two parts. The first part, including TrainingLogReg.h and TrainingLogReg.m, implements a computational model that classifies digit 0 and digit 1 by using gradient loss function. The second part, including ViewController.h and ViewController.m, implements a module that connects Firebase repositories, downloads current weights, calculates loss based on current weights and the model in the first part, and upload loss to Firebase repositories. 


 < TrainingLogReg >
This part of code is mainly about implementing a classifier that works on digit 0 and digit 1. The whole work flow is demonstrated in the funtion //trainModelWithSelfTest, which is only used for testing. This part of code has two functions that might be used outside this class. The first one is //trainModelWithWeight: (float *) w. It reads weights as (float *)w and then computes a new loss in terms of w. Note that it randomly picks a sample from the training dataset without replacement. The second function is calculateTrainAccuracyWithWeight: (float *)w. It read weights as (float *)w and then, based on this w, goes through the training dataset to compute the accuracy of its predictions. In this part, there are two functions for reading files. One is for TrainingFeature.dat and the other is for TrainLabel.dat. At last, there is a utility function for reading contents from files. 


< ViewController >
In this part, the code communicates with Firebase and trains a classifier remotely. Once this view appears, from userNames, an array of strings including three users who have already registered, it selects one user and lets him sign in. Later, all loss values and other information are saved in this user’s directory in Firebase. There is a utility function //createUserInFirebaseWithUserName: (NSString *) userName Password: (NSString *) password for creating new users.  The code here has three different workflows.
The first one, called by function //calculateTrainErrorTapped, downloads the current weights in Firebase, and then calculates the accuracy of predictions based on the downloaded weights.
The second one, called by function //trainOnceButtonTapped, firstly disables the button and requests an observation event to check “readByServer” variable in Firebase. If there is no such variable, which means this user’s directory has not been set up, the code sets up everything for this user and then alerts about this situation. If “readByServer” is false, which indicates that the last loss value has not been processed by the server, since this workflow sets the global variable isTrainOnce at true, the code pops up an alert that this training fails. If “readByServer” is true, which indicates that the server is ready for the next loss value, the code downloads the current weights from Firebase and utilizes the training model to compute a new loss. If the current weights is empty, the code does nothing but uploads an initial weights to Firebase. 
The third one, called by function //trainingButtonTapped, communicates with Firebase and trains this classifier multiple times. The code does the same thing, that described in the second workflow, one by one. The only difference is that the code sets the global variable “isTrainOnce” at false. Thus the code here would not stop checking “readByServer” to complete one training until the code finishes enough trainings indicated by “totalTrainTimes”. 


Requirement:
        How to set up firebase


1. Create an account.
2. Create an new app in dashboard and copy the standard Firebase URL that is the root reference to this Firebase repository.


 How to set up Xcode.
1. The code needs one library Firebase, which is already in the project uploaded to Dropbox.
2. If you want to set up again, use CocoaPods to configure this dependency. See details, please go to the website: https://www.firebase.com/docs/ios/guide/setup.html


Getting Start: 
1. Download the code from dropbox. Also download the temporary server code written in Python.
2. Run the app. Then go to the server code directory where there are two folders, /bin and /template. In this directory, type “python bin/app.py”. Note that it won’t work if you do “cd bin; python app.py”. 
3. Now you can feel free to use this app. Note that if you only start the app without turning on the server, you cannot train your model and even crash the app. This is because when you train the model multiple times, the code in app won’t stop until server gives response.