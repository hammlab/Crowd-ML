##### About python client
The pythonClient.py implements all basic functionalities of a client-side app similar to iOS and android implementations.
WARNING: the client uses the secret key from the firebase db, and should NOT be distributed in public.
The client is best used as a test for developing purposes.  

The client can also run locally (i.e., off-line) without firebase by setting
```
localTraining = True
```
inside the script. 
Local training is much faster than using firebase, and is useful for testing.

##### Requirements
To use this script, you need to install the following libraries:

1. firebase-python <http://ozgur.github.io/python-firebase/>
2. firebase-token-generator-python <https://github.com/firebase/firebase-token-generator-python>

To install, run
```
pip install python-firebase
pip install firebase-token-generator
```
For more information on installation, please refer to the websites above.

