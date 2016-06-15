Crowd-ML ReadMe


	Abstract

	The source code contained in this library is for an Android app which allows users to securely update data on a Firebase database for machine learning purposes, without ever directly sending the user’s own data. By logging in or creating an account via an email/password combination, every user is assigned a unique user ID to store gradient values and a checkout boolean. The app will take the weight vector from Firebase, and use that alongside a generated input vector to create a gradient. That gradient is then sent back to the user’s Firebase branch, which is used by a central server to update the weight vector. The app can also sequentially send numerous gradient vectors, which are kept from interfering from each other using a checkout system involving Firebase event listeners and the user’s boolean value. 
	The Server code is a JavaScript file which when running will listen for any updates in the user values of the Firebase database. When a change is made or new user added, the new gradient value is read and compiled into a minibatch stack stored inside the server, and the user's checkout value is adjusted to let the client know the value has been read. Once that minibatch stack has reached the desired size (currently set to 10), the gradient values are averaged and applied to the weight vector. Each weight update is based on the learning rate, which is adjusted every iteration.


	Requirements


There are 3 requirements to run the client code:


1. Firebase- Firebase is an online database which allows communication between servers and apps. In order for the features of the app to work, it needs a firebase account to read and send data to and from. The existing code uses the database at https://gd-prototype.firebaseio.com but this can be substituted for another with a similar tree structure. In order to create a new database, you simply need to create an account at www.firebase.com and select “create new app”
2. Java SDK - This is necessary for Android studio to run properly and can be downloaded from http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
3. Android Studio- The source code runs on Android Studio. The IDE and bundled SDK tools can be downloaded from http://developer.android.com/sdk/index.html. Once Android Studio is installed, you can use it to open the GD-Prototype folder and access and edit the source code.

There are 2 requirements to run the server code:

1. Download and install Node which can be done here: https://nodejs.org/en/
2. Setup Node Firebase API with the commands
$npm install firebase --save
$npm install firebase-token-generator --save
$npm install -g firebase-tools
        

        Getting Started

For client code:

1. Download GD-Prototype from github repository
2. Open in Android Studio
3. Set Build Variant to release
4. Select Build>Build APK
5. Download APK onto Android device
6. Run

For server code:
1. Download gd-server.js 
2. run command
$node gd-server.js


	Load Testing

	Load testing of the server/firebase was done using a python script to emulate a large number of clients simultaneously. While due to Firebase's REST API, none of these are ever truly "signed in," they are still able to send authenticated requests to Firebase, which serves the same purpose. 

	Requirements

1. Install Python's Firebase library
$pip install firebase
2. Install Python's firebase-token-generator library
$pip install firebase-token-generator

	Getting Started

1. Run testing Node server the same as listed above
$node test-server.js
2. Run multiclient Python code
$python multiClient.py <number of clients you wish to emulate (max 100)>
3. If every weight from 1 to the number chosen is equal to 2, the test has run successfully
