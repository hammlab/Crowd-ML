#### Running multiple virtual devices on a single machine
We used a python script to emulate a large number of clients simultaneously. 
While due to Firebase's REST API, none of these are ever truly "signed in," 
they are still able to send authenticated requests to Firebase, which serves the same purpose. 

#####	Requirements

1. Install Python's Firebase library
```
$pip install firebase
```
2. Install Python's firebase-token-generator library
```
$pip install firebase-token-generator
```

##### Testing

1. Run testing Node server 
```
$node test-server.js
````
2. Run multiclient Python code
```
$python multiClient.py <number of clients you wish to emulate (max 100)>
```
3. If every weight from 1 to the number chosen is equal to 2, the test has run successfully
