
### Todo
We'll make it public by June 19th  

2. Add header in every file
5. Server app more robust? (wrong data size, nan, etc)
7. How does firebase verify clients? Later.
8. Client
  * Initially asks for id and passwd. Server URL and .JSON is embedded.
  * If paramIter changes, it reset the process.  
9. Server
  * Has full control over the clients?
  * Command: pause, resume, restart, expire
  * Useful for a new session with different parameters, e.g., with different loss types.  

10. firebase:crowdml/params: paramIter, D, K, L, N, batchSize, lossType=('Hinge','Softmax','LogReg'), noiseType=(Gaussian|Laplace|NoNoise), noiseScale, trainingFeatureFile, trainingLabelFile
11. firebased:crowdml/users/uid/paramIter: 

12. No constant files for clients.
