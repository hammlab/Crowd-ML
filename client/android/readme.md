#####Add Firebase to your app

* Your app will not build until this is finished

1. Download Android Studio 1.5 or higher
2. Install Android SDK Manager
3. Download Android Studio project and retrieve the package name
4. Paste the training data feature and label files (found in the data folder) and paste them into /Crowd-ML/client/android/Crowd-ML/app/src/main/assets/ 
5. From the Firebase console at console.firebase.google.com, go to your Firebase project and click 'Add Firebase to your Android app' and follow the steps
6. Copy the downloaded google-services.json file into your app's module folder, typically app/

#####Add the SDK 

Most dependencies are universal between Android projects and are included in the download, however if you wish to add more features, you may need to add additional Firebase dependencies. To do this, in your module Gradle file (in the app/build.gradle) file, add the apply plugin line at the bottom of the file to enable the Gradle plugin:

 ```
    apply plugin: 'com.android.application'
    
    android {
    // ...
    }
    
    dependencies {
    // ...
    compile 'com.google.firebase:firebase-core:9.0.2'
    //<Additional dependencies go here>
    }
 ```

A more detailed instruction and list of additional SDKs can be found here you may wish to use:

https://firebase.google.com/docs/android/setup


####Configuring Android app

When the app begins, it will download the parameters from the firebase database, which have been uploaded by the server. These consist of: 

* regularization constant
* noise variance
* noise distribution
* number of classes
* loss function
* label and feature source files (these must be located in the assets folder)
* featureSize
* number of training samples
* gradient batch size.
