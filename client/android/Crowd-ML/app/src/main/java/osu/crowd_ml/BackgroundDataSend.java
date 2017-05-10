package osu.crowd_ml;

/*
    Copyright 2016 Crowd-ML team


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

import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.os.PowerManager;
import android.support.annotation.Nullable;
import android.support.v4.app.NotificationCompat;
import android.telephony.TelephonyManager;
import android.util.Log;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import osu.crowd_ml.loss_functions.LossFunction;
import osu.crowd_ml.noise_distributions.Distribution;
import osu.crowd_ml.utils.NetworkUtils;

public class BackgroundDataSend extends Service {

    // TODO(tylermzeller) this is never used. Consider removing.
    //private static final int DEFAULT_BATCH_SIZE = 1;

    final static FirebaseDatabase database = FirebaseDatabase.getInstance();
    final static DatabaseReference ref = database.getReference();
    final static DatabaseReference weightsRef = ref.child("trainingWeights");
    final static DatabaseReference parametersRef = ref.child("parameters");
    DatabaseReference userRef;

    // Handling WiFi connectivity
    private Thread wifiThread;
    private Thread workThread;
    private Handler wifiHandler;
    private volatile boolean isWifiConnected = false;
    private boolean wifiDisconnect = false;

    // Wakelock
    private PowerManager.WakeLock wakeLock;

    private String UID;
    private List<Integer> order;
    private TrainingWeights weightVals;
    private Parameters params;
    private UserData userCheck;
    private int gradientIteration;
    private int dataCount = 0;
    private boolean init;

    // Database Listeners
    private ValueEventListener userListener;
    private ValueEventListener paramListener;
    private ValueEventListener weightListener;

    // Bebugging
    private boolean VERBOSE_DEBUG = true;

    private int paramIter;
    private Distribution dist;
    private int K;
    private LossFunction loss;
    private String labelSource;
    private String featureSource;
    private int D;
    private int N;
    private int batchSize;
    private double noiseScale;
    private double L;
    private int nh;
    private int localUpdateNum;
    private double c;
    private double eps;
    private String descentAlg;
    private int maxIter;
    private volatile int t;
    private List<Double> learningRate;
    private List<Double> weights;

    private int length;

    private boolean network;
    private BroadcastReceiver receiver;

    @Nullable @Override public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();

        // Step 1. Extract necessary information
        UID = MultiprocessPreferences.getDefaultSharedPreferences(this).getString("uid", "");

        // Step 2. Get database references.
        userRef = ref.child("users").child(UID);

        // Step 3. Initialize necessary data.
        weightVals = new TrainingWeights();
        userCheck = new UserData();
        params = new Parameters();

        // Step 4. Create a listener to handle wifi connectivity.
        network = isDataConnected();
        receiver = new BroadcastReceiver() {
            public void onReceive(Context context, Intent intent) {
                network = isDataConnected();
                handleWifiChange();
            }
        };
        registerReceiver(receiver, new IntentFilter("android.net.conn.CONNECTIVITY_CHANGE"));

        // Step 5. Acquire a lock on the CPU for computation during sleep.
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyWakelockTag");
        wakeLock.acquire();

        // Step 6. Begin this service as a foreground service.
        Intent notificationIntent = new Intent(this, Login.class);

        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0,
                notificationIntent, 0);

        Notification notification = new NotificationCompat.Builder(this)
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentTitle("Background Service Running")
                .setContentText("Processing data")
                .setContentIntent(pendingIntent).build();

        /*
         * NOTE: A foreground service is used to decouple the service from the application. When a
         * user exits from the application view (the Login activity), using a foreground service
         * prevents this service from restarting. The number supplied below is arbitrary but must be
         * > 0.
         * */
        startForeground(1337, notification);

    }

    private boolean isDataConnected() {
        try {
            ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
            NetworkInfo info = cm.getActiveNetworkInfo();

            //TODO: See if making sure it's not a metered connection would be better? Consult: https://developer.android.com/reference/android/net/ConnectivityManager.html#isActiveNetworkMetered
            if (info.getType() == ConnectivityManager.TYPE_WIFI) {
                return cm.getActiveNetworkInfo().isConnectedOrConnecting();
            } else {
                return false;
            }
        } catch (Exception e) {
            return false;
        }
    }

    private void handleWifiChange() {
        if (network) {
            // Step 3. Check if wifi was previously disconnected.
            if (!isWifiConnected) {
                if (VERBOSE_DEBUG) {
                    Log.d("handleMessage", "Handling wifi connect.");
                }

                isWifiConnected = true;
                addFirebaseListeners();
            }
        } else {
            // Step 3. Check if wifi was previously connected.
            if (isWifiConnected) {
                stopWorkThread();
                if (VERBOSE_DEBUG) {
                    Log.d("handleMessage", "Handling wifi disconnect.");
                }

                wifiDisconnect = true;
                isWifiConnected = false;
                removeFirebaseListeners();
            }
        }
    }

    private void addFirebaseListeners(){
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
            }

            @Override public void onCancelled(DatabaseError error) {
                // Weight event listener error
                Log.d("BackgroundDataSend", "Weights listener error");
            }
        });
    }

    private void removeFirebaseListeners(){
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

    @Override public void onDestroy() {

        Log.d("onDestroy", "Unregister Wifi receiver.");
        // Step 1. End the wifi receiver.
        unregisterReceiver(receiver);

        Log.d("onDestroy", "Stopping the worker thread.");
        // Step 2. End the worker thread, if running.
        stopWorkThread();

        Log.d("onDestroy", "Removing Listeners.");
        // Step 3. Remove listeners.
        removeFirebaseListeners();

        Log.d("onDestroy", "Stopping foreground service.");
        // Step 4. Remove this service from the foreground.
        stopForeground(true);

        Log.d("onDestroy", "Releasing wakelock.");
        // Step 5. Release the wakelock.
        wakeLock.release();

        // Step 6. Stop the service.
        stopSelf();

        // Step 7. Let Android destroy the rest.
        super.onDestroy();
    }

    private void onParameterDataChange(DataSnapshot dataSnapshot){
        // Step 1. Get all parameters from the data snapshot
        params = dataSnapshot.getValue(Parameters.class);

        // Step 2. Set parameters.
        setParameters();

        if(loss.lossType().equals("binary") && K > 2){
            // Error: Binary classifier used on non-binary data
            dataCount = -1;
        }

        // Must call setLength() before these lines
        if (descentAlg.equals("adagrad") || descentAlg.equals("rmsProp"))
            learningRate = new ArrayList<>(Collections.nCopies(length, 0.0d));

        // Must clear the sample order list of previous contents.
        order = new ArrayList<>();

        // Calling this method after clearing the order list (the above line) will initialize
        // it to a list of size N consisting of int values in the range [0, N) in random order.
        maintainSampleOrder();

        // Step 3. ??? TODO: why do we add this here? This adds a new listener every time the parameters are updated
        // TODO: I believe this prevents trying to send gradients before parameters have been set for the client.
        addUserListener();
    }

    private void setParameters(){
        paramIter = params.getParamIter();
        dist = params.getNoiseDistribution();
        K = params.getK();
        loss = params.getLossFunction();
        labelSource = params.getLabelSource();
        featureSource = params.getFeatureSource();
        D = params.getD();
        N = params.getN();
        batchSize = params.getClientBatchSize();
        noiseScale = params.getNoiseScale();
        L = params.getL();
        nh = params.getNH();
        localUpdateNum = params.getLocalUpdateNum();
        c = params.getC();
        eps = params.getEps();
        descentAlg = params.getDescentAlg();
        maxIter = params.getMaxIter();

        // Call to setup the length
        loss.setLength(params);
        length = loss.getLength();

        // Added to stop infinite send loop
        dataCount = maxIter;
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
                        startGradientThread();
                    } else if (localUpdateNum > 0) {
                        // Step 8. Compute localUpdateNum steps of batchGD.
                        startWeightThread();
                    }
                }
            }

            @Override public void onCancelled(DatabaseError firebaseError) {
                // Error
            }
        });
    }

    private void stopWorkThread(){
        // Step 1. Check if the worker thread is non-null and running.
        if (workThread != null && workThread.isAlive()){

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

    /**
     * When the client runs with localUpdateNum=0, the client only computes the gradient of the
     * weights and sends the gradients back.
     */
    private void startGradientThread(){
        // Step 1. Check if the worker thread is non-null and running.
        if (workThread != null && workThread.isAlive()){

            // Wait for the thread to finish.
            try {
                workThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // Step 2. Reset the worker thread.
        workThread = new Thread() {
            @Override
            public void run() {
                sendGradient();
            }
        };

        // Step 3. Start the thread.
        workThread.start();
    }

    private void startWeightThread(){
        // Step 1. Check if the worker thread is non-null and running.
        if (workThread != null && workThread.isAlive()){

            // Wait for the thread to stop.
            try {
                workThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // Step 2. Reset the worker thread.
        workThread = new Thread() {
            @Override
            public void run() {
                sendWeight();
            }
        };

        // Step 3. Start the thread.
        workThread.start();
    }

    //allows for newly created users to initialize values
    private void initUser() {
        // Step 1. Get the current weight vector.
        //List<Double> initGrad = weightVals.getWeights().get(0);

        // Step 2. Send the DB the user's initial values.
        sendUserValues(weights, false, gradientIteration, t, paramIter);
    }

    /**
     * Maintains the sample order list.
     *
     * The sample order list is queried for random indices of training samples without replacement
     * (until all values are removed, that is).
     *
     * If order is null or empty, the list will be filled with int values in the range [0, N), then
     * shuffled.
     */
    private void maintainSampleOrder(){

        // Step 1. Ensure the order list is initialized.
        if (order == null){
            order = new ArrayList<>();
        }

        // Step 2. If the order list is empty, fill with values in the range [0, N).
        if (order.isEmpty()){
            for (int i = 0; i < N; i++) //create sequential list of input sample #s
                order.add(i);

            // Step 3. Randomize order.
            Collections.shuffle(order);
        }
    }

    /**
     * Compute the gradient of the weights and send back to the server.
     */
    private void sendGradient(){
        // Cache old parameters, in case we need to rollback changes
        List<Integer> oldOrder = new ArrayList<>(order);

        // Compute the gradient with random noise added.
        List<Double> noisyGrad = computeNoisyGrad();

        // Check if wifi is connected to send the gradient.
        if (!Thread.currentThread().isInterrupted()) {
            Log.d("sendGradient", "Sending gradient.");

            // Send the gradient to the server.
            sendUserValues(noisyGrad, false, ++gradientIteration, t, paramIter);

            // Decrease iteration
            //dataCount--;
        } else {
            order = new ArrayList<>(oldOrder);
        }
    }

    private void sendWeight(){
        // Cache old parameters, in case we need to rollback changes
        int oldT = t;
        List<Integer> oldOrder = new ArrayList<>(order);

        // Get current weights.
        //List<Double> weights = weightVals.getWeights().get(0);
        // Calc new weights using local update num.
        for (int i = 0; i < localUpdateNum; i++) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
            weights = internalWeightCalc();
            //t++;
            if (VERBOSE_DEBUG)
                Log.d("sendWeight", "local iter: " + i + 1);
        }

        // Check if wifi is connected to send the gradient.
        if (!Thread.currentThread().isInterrupted()) {
            if (VERBOSE_DEBUG)
                Log.d("sendGradient", "Sending gradient.");

            // Send the gradient to the server.
            sendUserValues(weights, false, ++gradientIteration, t, paramIter);

            // Decrease iteration.
            //dataCount--;
        } else { // If we can't send the gradient, rollback the old params
            if (VERBOSE_DEBUG)
                Log.d("sendGradient", "Can't send gradient.");
            t = oldT;
            order = new ArrayList<>(oldOrder);
        }
    }

    private void sendUserValues(List<Double> gradientsOrWeights, boolean gradientProcessed, int gradIter, int weightIter, int paramIter){
        userRef.setValue(
            new UserData(gradientsOrWeights, gradientProcessed, gradIter, weightIter, paramIter)
        );
    }

    private List<Integer> gatherBatchSamples(){
        List<Integer> batchSamples = new ArrayList<>();

        int i = 0; // counter
        Random r = new Random(); // rng

        // Loop batchSize times
        while(i < batchSize) {
            // Calling this method here ensures that the order list is never empty. When the order
            // list becomes empty, a new epoch of training occurs as the list is repopulated with
            // random int values in the range [0, N).
            maintainSampleOrder();

            // get a random index in the range [0, |order|) to query the order list.
            int q = r.nextInt(order.size());

            // Remove the value at index q and add it to the current batch of samples.
            batchSamples.add(order.remove(q));
            i++; // counter
        }
        return batchSamples;
    }

    private List<Double> computeAverageGrad(List<double[]> X, List<Integer> Y, List<Double> weights){
        // Init average gradient vector
        List<Double> avgGrad = new ArrayList<>(Collections.nCopies(length, 0.0d));

        // For each sample, compute the gradient averaged over the whole batch.
        double[] x;
        List<Double> grad;
        for(int i = 0; i < batchSize; i++){
            // Periodically check if this thread has been interrupted. See the javadocs on
            // threading for best practices.
            if (Thread.currentThread().isInterrupted()){
                break;
            }
            x = X.get(i); // current sample feature
            int y = Y.get(i); // current label

            // Compute the gradient.
            grad = loss.gradient(weights, x, y, D, K, L, nh);

            // Add the current normalized gradient to the avg gradient vector.
            for(int j = 0; j < length; j++) {
                avgGrad.set(j, (avgGrad.get(j) + grad.get(j)) / batchSize);
            }

//            for(int j = 0; j < length; j++) {
//                avgGrad.set(j, avgGrad.get(j) / batchSize);
//            }

        }

        return avgGrad;
    }

    private List<Double> computeNoisyGrad(){
        // Init training sample batch
        List<Integer> batchSamples = gatherBatchSamples();

        // TODO(tylermzeller) this is a bottleneck on physical devices. Buffered file I/O seems to
        // invoke the GC often.
        // Get training sample features.
        List<double[]> xBatch = readSamples(batchSamples);

        // Get training sample labels.
        List<Integer> yBatch = readLabels(batchSamples);

        // Compute average gradient vector
        List<Double> avgGrad = computeAverageGrad(xBatch, yBatch, weights);

        // Init empty noisy gradient vector
        List<Double> noisyGrad = new ArrayList<>(length);

        // Add random noise probed from the client's noise distribution.
        for (double avg : avgGrad) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }
            // TODO(davidsoller): This changes the size of the ArrayList after we initialized it with a certain size
            // Change to .add(index, element) as per https://developer.android.com/reference/java/util/ArrayList.html
            noisyGrad.add(dist.noise(avg, noiseScale));
        }

        return noisyGrad;
    }

    private List<Double> internalWeightCalc(){
        // Compute the gradient with random noise added
        List<Double> noisyGrad = computeNoisyGrad();

        // Periodically check if this thread has been interrupted. See the javadocs on
        // threading for best practices.
        if (Thread.currentThread().isInterrupted()){
            return noisyGrad;
        }

        // Return the updated weights
        return InternalServer.calcWeight(weights, noisyGrad, learningRate, t, descentAlg, c, eps);
    }

    public List<double[]> readSamples(List<Integer> sampleBatch){
        List<double[]> xBatch = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(featureSource)));
            String line;
            int counter = 0;
            double[] sampleFeatures = new double[D];
            String[] features;
            int max = Collections.max(sampleBatch);
            while ((line = br.readLine()) != null && counter <= max){
                // Periodically check if this thread has been interrupted. See the javadocs on
                // threading for best practices.
                if (Thread.currentThread().isInterrupted()) {
                    break;
                }

                if(sampleBatch.contains(counter)){

                    features = line.split(",| ");

                    // TODO: why is this list necessary?
                    //List<String> featureList = new ArrayList<>(Arrays.asList(features));
                    //featureList.removeAll(Arrays.asList(""));
                    for(int i = 0; i < D; i++) {
                        sampleFeatures[i] = Double.parseDouble(features[i]);
                        //sampleFeatures[i] = Double.parseDouble(features[i]);
                    }
                    xBatch.add(sampleFeatures);
                }
                counter++;
            }
        } catch (IOException e) {
            // TODO(tylermzeller) probably a better way to handle this.
            e.printStackTrace();
            dataCount = -1;
        }
        return xBatch;
    }

    public List<Integer> readLabels(List<Integer> sampleBatch){
        List<Integer> yBatch = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(labelSource)));
            String line;
            int counter = 0;
            while ((line = br.readLine()) != null && counter <= Collections.max(sampleBatch)){
                // Periodically check if this thread has been interrupted. See the javadocs on
                // threading for best practices.
                if (Thread.currentThread().isInterrupted()) {
                    break;
                }

                if(sampleBatch.contains(counter)){
                    line = line.trim();
                    int sampleLabel = Integer.parseInt(line);
                    if(sampleLabel == 0 && loss.lossType().equals("binary")){
                        sampleLabel = -1;
                    }
                    yBatch.add(sampleLabel);
                }
                counter++;
            }
        } catch (IOException e) {
            // TODO(tylermzeller) probably a better way to handle this.
            e.printStackTrace();
            dataCount = -1;
        }
        return yBatch;
    }
}
