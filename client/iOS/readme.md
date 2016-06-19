#### Requirements

###### How to set up firebase
  
1. Create an account.
2. Create a new project 
3. Go into your new app, click "Add Firebase to your iOS app", and follow the instruction
4. Go into your new app, and select "Auth".
5. "Set up sign in method" --> "Email/password" --> "enable" and save

  Then you can create a new users here: --> refresh the webpage --> add users
  
  Or create a new user when you use the app.



##### How to set up Xcode
 
1. The code needs Firebase library, which is already in the project. You shoud already install it in above step 3.
2. Date files currently in "FireBase_Demo" folder are "trainingFeature.dat" and "trainingLabel.dat". Download "MNISTTrainLabels.dat" and "MNISTTrainImages.dat.zip"(unzip it before using) from "data" folder, and put them in "FireBase_Demo" folder, if necessary.


##### How to use the app
1. First, the app will prompt a Login/Signup pop up. If you are a new user, you can create a new account by entering your email(more than 5 characters) and password(more than 5 characters). If you are a registered user, simply enter your email and password. If you enter a wrong password, the pop up will show up again and please re-enter your account information.

  Note: if you have troube logining, please see a detailed error message in Xcode output console.
  
2. "Setup" (this step will set up a user's information in firebase database) --> enter an integer under "Train times" --> "Start Training" --> wait until it's done. You can see the current accuracy in terminal and messages in output console in Xcode --> "Start Training" (if you want to train it again, or "Get Train Accuracy" if your are training a binary model)


#### Getting started 

1. Download the whole "client/iOS" folder from Github. Also download all files in "server" folder.
2. Run the app. Then run the server (Check a readme file of the server to see how to run it). 
3. Now you can feel free to use this app. Note that if you only start the app without turning on the server, you cannot train your model, because "gradientProcessed" cannot be set to true by server so that the client will not send a new gradient.
