import sys
import numpy as np
from firebase import firebase
from firebase_token_generator import create_token

ref = firebase.FirebaseApplication('https://gd-prototype.firebaseio.com', None)
users = firebase.FirebaseApplication('https://gd-prototype.firebaseio.com/users', None)
secret = 'MbbCsPSfwasu98Ho4jCS5BdeJDplqjHzWLky4c46'

D = 785; #Feature size
nTrain = 12665; #number of training samples
ready = False;


uid = sys.argv[1]

#uid = "99a37f55-4837-42a6-9188-7801e8cc0005"
print("uid: "+uid)
auth_payload = {"uid": uid}
token = create_token(secret, auth_payload)
user = '/users/'+uid+'/'

result = ref.get(user+'gradients' , None, params = {"auth":token})
ready = True;
iterNum = int(sys.argv[2])
print("iterNum: "+sys.argv[2])
i = 0;
while i < iterNum:
	grad = np.zeros(D) #simplified gradient for testing
	grad[int(sys.argv[1])] = 1.0;
		
	checkout = ref.get(user+'checkout' , None, params = {"auth":token})
	if(checkout == False):
		gradJson = grad.tolist()
		ref.put(user, 'gradients', gradJson, params = {"auth":token}) 
		ref.put(user ,'checkout', True, params = {"auth":token})
		i+= 1
	



print(uid+" finished")

