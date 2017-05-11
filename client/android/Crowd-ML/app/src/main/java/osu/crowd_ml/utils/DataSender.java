package osu.crowd_ml.utils;

/*
    Copyright 2017 Crowd-ML team


    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License
*/

import android.content.Context;
import android.util.Log;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.List;

import osu.crowd_ml.Parameters;
import osu.crowd_ml.TrainingWeights;
import osu.crowd_ml.UserData;
import osu.crowd_ml.loss_functions.LossFunction;

public final class DataSender {

    // Firebase references
    private final static FirebaseDatabase database = FirebaseDatabase.getInstance();
    private final static DatabaseReference ref = database.getReference();
    private final static DatabaseReference weightsRef = ref.child("trainingWeights");
    private final static DatabaseReference parametersRef = ref.child("parameters");
    private DatabaseReference userRef;

    // Database Listeners
    private ValueEventListener userListener;
    private ValueEventListener paramListener;
    private ValueEventListener weightListener;

    // Handling WiFi connectivity
    private boolean wifiDisconnect = false;

    // Bebugging
    private boolean VERBOSE_DEBUG = true;

    // Parameters
    private Parameters params;
    private int paramIter;
    private LossFunction loss;
    private int localUpdateNum;
    private volatile int t;
    private List<Double> weights;

    // Work calculations
    private Thread workThread;
    private String UID;
    private DataComputer dataWorker;
    private TrainingWeights weightVals;
    private UserData userCheck;
    private int gradientIteration;
    private boolean init;

    public DataSender(String UID, Context context) {
        // Get database references.
        userRef = ref.child("users").child(UID);
        this.UID = UID;

        dataWorker = new DataComputer(context);
        params = new Parameters();
    }

    public void setWifiDisconnect(boolean state) {
        wifiDisconnect = state;
    }

    private void setParameters(){
        paramIter = params.getParamIter();
        loss = params.getLossFunction();
        localUpdateNum = params.getLocalUpdateNum();
    }

    public void addFirebaseListeners(){
        if (VERBOSE_DEBUG)
            Log.d("addFirebaseListeners", "Adding listeners.");

        // Step 1. Add parameters listener.
        paramListener = parametersRef.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
                if (VERBOSE_DEBUG)
                    Log.d("onDataChange", "Got parameters");

                onParameterDataChange(dataSnapshot);
            }

            @Override public void onCancelled(DatabaseError error) {
                // Parameter listener error
                Log.d("BackgroundDataSend", "Parameter listener error");
            }
        });

        // Step 2. Add weight listener.
        weightListener = weightsRef.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
                if (VERBOSE_DEBUG)
                    Log.d("onDataChange", "Got weights");

                stopWorkThread();

                weightVals = dataSnapshot.getValue(TrainingWeights.class);
                weights = weightVals.getWeights().get(0);
                t = weightVals.getIteration();
                dataWorker.setWeights(weightVals);
            }

            @Override public void onCancelled(DatabaseError error) {
                // Weight event listener error
                Log.d("BackgroundDataSend", "Weights listener error");
            }
        });
    }

    public void removeFirebaseListeners(){
        Log.d("removeFirebaseListeners", "Removing listeners.");

        // Step 1. Check if listeners are null, and if not remove them as listeners.
        if (paramListener != null)
            parametersRef.removeEventListener(paramListener);

        if (weightListener != null)
            weightsRef.removeEventListener(weightListener);

        if (userListener != null)
            userRef.removeEventListener(userListener);

        // Step 2. Set to null.
        paramListener  = null;
        userListener   = null;
        weightListener = null;
    }


    private void onParameterDataChange(DataSnapshot dataSnapshot){
        // Step 1. Get all parameters from the data snapshot
        params = dataSnapshot.getValue(Parameters.class);

        // Step 2. Set parameters.
        setParameters();
        dataWorker.setParameters(params);

        // Calling this method after clearing the order list (the above line) will initialize
        // it to a list of size N consisting of int values in the range [0, N) in random order.
        dataWorker.maintainSampleOrder();

        // Step 3. ??? TODO: why do we add this here? This adds a new listener every time the parameters are updated
        // TODO: I believe this prevents trying to send gradients before parameters have been set for the client.
        addUserListener();
    }

    private void addUserListener(){

        // Step 1. Check if there is already a user listener and remove if so.
        if (userListener != null) {
            userRef.removeEventListener(userListener);
            userListener = null;
        }

        // Step 2. Check if a wifi disconnect caused this listener to be reinitialized.
        if (!wifiDisconnect) {
            // Wifi did not disconnect. This means the application has been launched for the first
            // time.
            Log.d("addUserListener", "Wifi was not disconnected.");
            init = false;

            // TODO(tylermzeller): Gradient iteration needs to be 0 when a new experiment starts.
            // How do we let the client know a new experiment has started?
            gradientIteration = 0;
        }

        // If there was a wifi disconnect, we need to reset the disconnect var.
        wifiDisconnect = false;

        // Step 3. Add new user listener.
        userListener = userRef.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
                if (VERBOSE_DEBUG)
                    Log.d("onDataChange", "Got user values.");

                // Step 4. Get updated user values.
                userCheck = dataSnapshot.getValue(UserData.class);

                // Step 5. If the user hasn't been initialized yet, do it now.
                if(!init){
                    if (VERBOSE_DEBUG)
                        Log.d("userValues", "Init user");
                    init = true;
                    initUser();
                }

                Log.d("userValues", userCheck.getGradientProcessed() + " " + userCheck.getGradIter() + " " + gradientIteration);

                // Step 6. Check if we can compute the gradient.
                if (userCheck.getGradientProcessed() && userCheck.getGradIter() == gradientIteration) {

                    // Step 7. Check the localUpdateNum for the type of processing the client should do.
                    if (localUpdateNum == 0) {
                        // Step 8. Compute a single step of SGD.
                        Thread gradientThread = new Thread() {
                            @Override
                            public void run() {
                                List<Double> gradients = dataWorker.getNoisyGradients();
                                Log.d("sendGradientWeights", "Attempting to send.");
                                sendComputedValues(gradients);
                            }
                        };
                        startWorkThread(gradientThread);
                    } else if (localUpdateNum > 0) {
                        // Step 8. Compute localUpdateNum steps of batchGD.
                        Thread weightThread = new Thread() {
                            @Override
                            public void run() {
                                List<Double> weights = dataWorker.getWeights();
                                Log.d("sendComputedValues", "Attempting to send.");
                                sendComputedValues(weights);
                            }
                        };
                        startWorkThread(weightThread);
                    }
                }
            }

            @Override public void onCancelled(DatabaseError firebaseError) {
                // Error
            }
        });
    }

    //allows for newly created users to initialize values
    private void initUser() {
        // Step 1. Get the current weight vector.
        //List<Double> initGrad = weightVals.getWeights().get(0);

        // Step 2. Send the DB the user's initial values.
        sendUserValues(weights, false, gradientIteration, t, paramIter);
    }

    /**
     * Compute the gradient of the weights and send back to the server.
     */
    private void sendComputedValues(List<Double> values){
        // Check if wifi is connected to send the gradient.
        if (!Thread.currentThread().isInterrupted()) {
            if (VERBOSE_DEBUG)
                Log.d("sendComputedValues", "Sending values to server.");

            // TODO (david soller) : This wasn't changing between Gradient & Weights error?
            boolean gradientProcessed = false;

            // Send the gradient to the server.
            sendUserValues(values, gradientProcessed, ++gradientIteration, t, paramIter);
        } else {
            if (VERBOSE_DEBUG)
                Log.d("sendComputedValues", "Can't send values to server.");
        }
    }

    private void sendUserValues(List<Double> gradientsOrWeights, boolean gradientProcessed, int gradIter, int weightIter, int paramIter){
        userRef.setValue(
                new UserData(gradientsOrWeights, gradientProcessed, gradIter, weightIter, paramIter)
        );
    }

    /**
     * When the client runs with localUpdateNum=0, the client only computes the gradient of the
     * weights and sends the gradients back.
     */
    private void startWorkThread(Thread thread) {
        // Step 1. Check if the worker thread is non-null and running.
        if (workThread != null && workThread.isAlive()) {

            // Wait for the thread to finish.
            try {
                workThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // Step 2. Reset the worker thread.
        workThread = thread;

        // Step 3. Start the thread.
        workThread.start();
    }

    public void stopWorkThread() {
        // Step 1. Check if the worker thread is non-null and running.
        if (workThread != null && workThread.isAlive()) {

            // Step 2. Interrupt the thread.
            workThread.interrupt();

            // Step 3. Wait for the thread to die.
            try {
                workThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                Log.d("stopWorkThread", "Work thread ended.");
            }
        }
    }
}
