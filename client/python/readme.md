##### About python client
The pythonClient.py implements all basic functionalities of a client-side app similar to iOS and android implementations.
WARNING: the client uses the secret key from the firebase db, and should NOT be distributed in public.
The client is best used as a test for developing purposes.  
<!---
The pythonClient_local.py is similar to pythonClient.py, but it runs locally (i.e., off-line) without firebase for. This is for testing purposes.
--->
To use this script, you need to install the following libraries:

1. firebase-python <http://ozgur.github.io/python-firebase/>
2. firebase-token-generator-python <https://github.com/firebase/firebase-token-generator-python>

To install, run
```
pip install python-firebase
pip install firebase-token-generator
```
For more information on installation, please refer to the websites above.

