### Crowd-ML (Machine Learning)
#### is a framework for crowd-sourced machine learning with privacy mechanisms.
---
Smart and connected devices are becoming increasingly pervasive in daily life,
including smartphones, wearables, smart home appliance, smart meters, connected cars, surveillance cameras, 
and environmental sensors. 
Can we benefit from analysis of massive data generated from such devices, without storing all data centrally or breaching  users' privacy?
More specifically, can machine learning be outsourced to a crowd of connected devices, with a guarantee of differential privacy? 

![Crowd-ML concept figure](schematic1-5.jpg "Crowd-ML concept")

The library allows devices (Android, iOS, and python clients) to learn a common classifier/regression model with differential privacy, by solving the distributed ERM problem: min_w f(w) = 1/M sum_{i=1}^M f_i(w), where f_i(w) = 1/n sum_j l(h_w(x_{ij}), y_{ij}).
The library implements private distributed synchronous risk minimization based on [**Hamm'15**], using [Google Firebase](https://firebase.google.com/) as a simple and robust syncrhonization method.  This idea was featured in [Gigaom] (https://gigaom.com/2015/01/22/researchers-show-a-machine-learning-network-for-connected-devices/).

Choosing the type and amount of noise to guarantee differential privacy is left to the library user; the type and the amount 
depend on model assumptions. Please see [Chaudhuri'11], [Rajkumar'12], [Song'13], [Bassily'14], [Hamm'16].
If noise is not used, this library can also serve as a crowd-based, parallel/distributed optimization framework [Tsitsiklis'84], [Agarwal'11], [Dekel'11]. 


### Features
---
#### Implemented client types
Android (JAVA), iOS (Objective C), linux (python)

#### Implemented server types
Node.js

#### Available options

* Loss function / classifiers: binary logistic regression, softmax, binary SVM with hindge loss 
* Noise:  type {Laplace, Gaussian}, scale s
* Learning rate: constant, 1/t, 1/sqrt{t}, AdaGrad, RMSprop
* Client- and server-side minibatch

#### Applications
---
Currently, the system is demonstrated with the MNIST dataset (http://yann.lecun.com/exdb/mnist/),
for 10-class and binary (0-vs-1) classification problems. 
Ideally, the most relevant types of data whose privacy is important are those generated
from smartphones and IoT devices. More examples will be added in the near future. 


### How to use the Crowd-ML library
---
#### 1. Set up firebase account.
See [firebase/readme.md](firebase/readme.md) for more instructions.

#### 2. Download and build client apps
Currently, the client uses pre-stored local data such as MNIST.
The users of this library should replace it with custom data sensing or collecting routines. Specific instructions are available for:

- iOS devices, see [client/iOS/readme.md](client/iOS/readme.md).
- Android devices, see [client/android/readme.md](client/android/readme.md).
- Python clients, see [client/python/readme.md](client/python/readme.md).

#### 3. Download and change server-side app.
See [server/readme.md](server/readme.md) for more instructions.

#### 4. Distribute client apps to users and start the server-side app.


### Acknowledgements
---
* Jihun Hamm (PI, OSU [homepage](https://web.cse.ohio-state.edu/~hammj/))
* Tyler Zeller (Undergraduate, OSU)
* David Soller (Undergraduate, OSU)
* Sayam Ganguly (Graduate, OSU)
* Jackson Luken (Undergraduate, OSU)
* Yani Xie (Undergraduate, OSU)
  
This research was supported in part by Google Faculty Research Award 2015 and Google Internet of Things Technology Research Award 2016. 


### References
* [Hamm'17]: J. Hamm, J. Luken, Y. Xie, "Crowd-ML: A Library for Privacy-preserving Machine Learning on Smart Devices," IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017 [pdf](docs/icassp17_2_jh.pdf)
* [Hamm'15]: J. Hamm, A. Champion, G. Chen, M. Belkin, and D.Xuan, 
"Crowd-ML: A privacy-preserving learning framework for a crowd of smart devices." In Proceedings of the 35th IEEE
International Conference on Distributed Computing Systems (ICDCS). IEEE, 2015. [pdf](docs/icdcs15_jh_final.pdf)
* [Chaudhuri'11]: K. Chaudhuri, C. Monteleoni, and A. D. Sarwate, "Differentially private empirical risk minimization," JMLR, vol. 12, 2011, pp. 1069–1109
* [Rajkumar'12]: A. Rajkumar, and S. Agarwal. "A differentially private stochastic
gradient descent algorithm for multiparty classification." In AISTATS, 2012, pp. 933–941
* [Song'13]: S. Song, K. Chaudhuri, and A. D. Sarwate, "Stochastic gradient descent with differentially private updates," in Proc. IEEE GlobalSIP, 2013.
* [Bassily'14]: R. Bassily, A. Smith, and A. Thakurta, A. "Private empirical risk minimization: Efficient algorithms and tight error bounds." In Foundations of Computer Science (FOCS), 2014, pp. 464-473
* [Hamm'16]: J. Hamm, P. Cao, and M. Belkin, "Learning privately from multiparty data," in Proc. ICML, 2016
* [Tsitsiklis'84]: J.N. Tsitsiklis, D.P. Bertsekas, and M. Athans, "Distributed asynchronous deterministic and stochastic gradient optimization algorithms." in American Control Conference, 1984, pp. 484-489 
* [Agarwal'11]: A. Agarwal and J. C. Duchi, "Distributed delayed stochastic optimiza-
tion." in Proc. NIPS, 2011, pp. 873–881.
* [Dekel'11]: O. Dekel, R. Gilad-Bachrach, O. Shamir, and L. Xiao, "Optimal distributed online prediction," in Proc. ICML, 2011.

### License
---
Released under the Apache License 2.0.  See the [LICENSE.txt](LICENSE.txt) file for further details.

