import sys
import numpy as np
from firebase import firebase
from firebase_token_generator import create_token

ref = firebase.FirebaseApplication('https://gd-prototype.firebaseio.com', None)
users = firebase.FirebaseApplication('https://gd-prototype.firebaseio.com/users', None)
secret = 'MbbCsPSfwasu98Ho4jCS5BdeJDplqjHzWLky4c46'

D = 785; #Feature size
nTrain = 12665; #number of training samples


uid = "99a37f55-4837-42a6-9188-7801e8cc0005"
print("test1")
auth_payload = { "uid": "system", "serverPrivilege": True}
token = create_token(secret, auth_payload)
user = '/users/'+'1'+'/'

result = ref.get('/users' , uid, params = {"auth":token})
#print(result)
ready = True;

uid = 1
while uid < 101:
	ref.put('/users/', str(uid), result, params = {"auth":token})
	uid+= 1;
	



test = input("enter any key to end ")
