package osu.crowd_ml;

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

import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.os.PowerManager;
import android.support.annotation.Nullable;
import android.support.v4.app.NotificationCompat;
import android.util.Log;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.lang.ref.WeakReference;
import java.util.List;

import osu.crowd_ml.BuildConfig;
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
    //private List<Integer> order;
    private TrainingWeights weightVals;
    private Parameters params;
    private UserData userCheck;
    private int gradientIteration;

    // Database Listeners
    private ValueEventListener userListener;
    private ValueEventListener paramListener;
    private ValueEventListener weightListener;

    // Training
    private Trainer trainer;

    private int paramIter;
    private int localUpdateNum;
    private volatile int t;
    private double[] weights;

    private volatile boolean weightsUpdated = false;
    private volatile boolean paramsUpdated  = false;

    private static class WifiHandler extends Handler {
        private final WeakReference<BackgroundDataSend> mService;

        WifiHandler(BackgroundDataSend service) {
            mService = new WeakReference<>(service);
        }
        @Override
        public void handleMessage(Message msg) {
            BackgroundDataSend service = mService.get();
            if (service != null) {
                service.handleMessage(msg);
            }
        }
    }

    public void handleMessage(Message msg){
        if (msg.what == 0){ // Now connected
            if (BuildConfig.DEBUG)
                Log.d("handleMessage", "Handling wifi connect.");

            isWifiConnected = true;
            addFirebaseListeners();
        } else if (msg.what == 1){
            stopWorkThread();
            if (BuildConfig.DEBUG)
                Log.d("handleMessage", "Handling wifi disconnect.");

            wifiDisconnect = true;
            isWifiConnected = false;
            removeFirebaseListeners();
        }
    }

    @Nullable @Override public IBinder onBind(Intent intent) {
        return null;
    }

    @Override public void onCreate() {
        super.onCreate();
        if (BuildConfig.DEBUG)
            Log.d("onCreate", "Create Service");

        // Step 1. Extract necessary information
        UID = MultiprocessPreferences.getDefaultSharedPreferences(this).getString("uid", "");

        // Step 2. Get database references.
        userRef = ref.child("users").child(UID);

        // Step 3. Initialize necessary data.
        weightVals = new TrainingWeights();
        userCheck = new UserData();
        params = new Parameters();

        // Step 4. Create a worker to handle wifi connectivity.
        wifiHandler = new WifiHandler(this);

        // Step 5. Acquire a lock on the CPU for computation during sleep.
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyWakelockTag");
        wakeLock.acquire();

        // Setup the ML training libraries
        trainer = TensorFlowTrainer.getInstance();

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

    // Start command is called whenever focus is given back to the app (like when the user clicks
    // the notification for the foreground service.
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        super.onStartCommand(intent, flags, startId);
        if (BuildConfig.DEBUG)
            Log.d("onStartCommand", "Start Service");

        // Make sure there isn't already a wifi thread working.
        if (wifiThread == null) {
            wifiThread = new Thread() {
                @Override
                public void run() {
                    try {
                        if (BuildConfig.DEBUG)
                            Log.d("wifiThread", "Detecting Wifi.");

                        // Step 1. Run thread until interrupted.
                        while (!isInterrupted()) {

                            // Step 2. Check if a wifi connection is detected AND if the user can access the internet.
                            if (NetworkUtils.isWifiConnected(BackgroundDataSend.this) && NetworkUtils.isOnline()) {
                                // Step 3. Check if wifi was previously disconnected.
                                if (!isWifiConnected) {
                                    wifiHandler.sendEmptyMessage(0);
                                }
                            } else {
                                // Step 3. Check if wifi was previously connected.
                                if (isWifiConnected) {
                                    wifiHandler.sendEmptyMessage(1);
                                }
                            }

                            // Step 4. Sleep 1 second before checking wifi status again.
                            Thread.sleep(1000);
                        }
                    } catch (Exception e) {
                        Log.e("wifiThread", "Interrupt.");
                    }
                }
            };

            wifiThread.start();
        }

        return START_STICKY;
    }

    private void addFirebaseListeners(){
        if (BuildConfig.DEBUG)
            Log.d("addFirebaseListeners", "Adding listeners.");

        // Step 1. Add parameters listener.
        paramListener = parametersRef.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
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
                onWeightDataChange(dataSnapshot);
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

        Log.d("onDestroy", "Stopping the worker thread.");
        // Step 1. End the worker thread, if running.
        stopWorkThread();

        Log.d("onDestroy", "Destroying the trainer.");
        trainer.destroy();

        Log.d("onDestroy", "Interrupting the wifi");
        // Step 2. End the wifi thread.
        stopWifiThread();

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

    /**
     * Communication with the server must be initiated with an {@code initUser()} call. The call to
     * this method relies on the retrieval of the most update-to-date parameters and weights from
     * the firebase listeners. It is unknown what order these listeners will dispatch updates, so
     * the check for updates from both listeners happens in both on*DataChange methods.
     */
    private void onParameterDataChange(DataSnapshot dataSnapshot) {
        if (BuildConfig.DEBUG)
            Log.d("onDataChange", "Got Params");
        // Step 1. Get all parameters from the data snapshot
        params = dataSnapshot.getValue(Parameters.class);

        // Step 2. Set parameters.
        setParameters();
        //setLength();
        if (!wifiDisconnect){
            gradientIteration = 0;
        }


        if(params.getLossFunction().lossType().equals("binary") && params.getK() > 2) {
            //TODO(tylermzeller): handle this error
            // Error: Binary classifier used on non-binary data
            //dataCount = -1;
        }

        // Parameters are now updated.
        // We don't want to set paramsUpdated to true if there was a wifi disconnection. A wifi
        // disconnection will reset the listeners which will trigger them with a default
        // response, not a response triggered by the server.
        paramsUpdated = !wifiDisconnect;



        // Step 3. ??? TODO: why do we add this here? This adds a new listener every time the parameters are updated
        // TODO: I believe this prevents trying to send gradients before parameters have been set for the client.
        addUserListener();
        if (paramsUpdated && weightsUpdated) {
            initUser();
            weightsUpdated = false;
        }

        // Always reset wifiDisconnect
        wifiDisconnect = false;
    }

    private void onWeightDataChange(DataSnapshot dataSnapshot) {
        if (BuildConfig.DEBUG)
            Log.d("onDataChange", "Got weights");

        // Stop the work thread if running.
        stopWorkThread();

        weightVals = dataSnapshot.getValue(TrainingWeights.class);
        List<Float> w = weightVals.getWeights();
        Log.d("onWeights", w.get(0) + "");
        weights = new double[w.size()];
        for (int i = 0; i < w.size(); i++) weights[i] = w.get(i);
        t = weightVals.getIteration();

        // Weight parameters are now updated
        weightsUpdated = true;


        Log.d("onDataChange", t + "");
        if (paramsUpdated) {
            initUser();
            paramsUpdated = false;
        }
    }

    private void setParameters() {
        paramIter = params.getParamIter();
        localUpdateNum = params.getLocalUpdateNum();
    }

    private void addUserListener(){

        // Step 1. Check if there is already a user listener and remove if so.
        if (userListener != null) {
            userRef.removeEventListener(userListener);
            userListener = null;
        }

        // Step 3. Add new user listener.
        userListener = userRef.addValueEventListener(new ValueEventListener() {
            @Override public void onDataChange(DataSnapshot dataSnapshot) {
                if (BuildConfig.DEBUG)
                    Log.d("onDataChange", "Got user values.");

                // Step 4. Get updated user values.
                userCheck = dataSnapshot.getValue(UserData.class);

                // Step 6. Check if we can compute the gradient.
                if (userCheck.getGradientProcessed() && userCheck.getGradIter() == gradientIteration) {
                    if (BuildConfig.DEBUG)
                        Log.d("onDataChange", "Can Start Work Thread");
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

    private void stopWifiThread(){
        // Step 1. Check if the worker thread is non-null and running.
        if (wifiThread != null && wifiThread.isAlive()){

            // Step 2. Interrupt the thread.
            wifiThread.interrupt();

            // Step 3. Wait for the thread to die.
            try {
                wifiThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                Log.d("stopWifiThread", "Wifi thread ended.");
            }
        }
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
        if (BuildConfig.DEBUG)
            Log.d("startGradientThread", "Starting Work Thread");
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
        if (BuildConfig.DEBUG)
            Log.d("startWeightThread", "Starting Work Thread");
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
        Log.d("initUser", "Param iter: " + paramIter + " Weight iter: " + t);
        // Send the server this user's initial values.
        List<Double> w = new ArrayList<>();
        for (double num : weights) w.add(num);
        sendUserValues(w, false, gradientIteration, t, paramIter);
    }

    /**
     * Compute the gradient of the weights and send back to the server.
     */
    private void sendGradient() {
        if (BuildConfig.DEBUG)
            Log.d("sendGradient", "Start Send Gradient");
        // Compute the gradient with random noise added.
        trainer .setIter(t)
                .setParams(params)
                .setWeights(weights);

        if (BuildConfig.DEBUG)
            Log.d("sendGradient", "Starting Trainer");
        List<Double> noisyGrad = trainer.getNoisyGrad();

        // Check if wifi is connected to send the gradient.
        if (!Thread.currentThread().isInterrupted()) {
            Log.d("sendGradient", "Sending gradient.");

            // Send the gradient to the server.
            sendUserValues(noisyGrad, false, ++gradientIteration, t, paramIter);
        } else if (BuildConfig.DEBUG) {
            Log.d("sendGradient", "Can't send gradient.");
        }
    }

    private void sendWeight(){
        if (BuildConfig.DEBUG)
            Log.d("sendWeight", "Start Send Weight");

        // Calc new weights
        trainer .setIter(t)
                .setParams(params)
                .setWeights(weights);

        if (BuildConfig.DEBUG)
            Log.d("sendWeight", "Starting Trainer");
        weights = trainer.train(localUpdateNum);
        List<Double> w = new ArrayList<>();
        for (double num : weights) w.add(num);

        // Check if wifi is connected to send the gradient.
        if (!Thread.currentThread().isInterrupted()) {
            if (BuildConfig.DEBUG)
                Log.d("sendWeight", "Sending weights.");
                Log.d("sendWeight", w.get(0) + "");

            // Send the gradient to the server.
            sendUserValues(w, false, ++gradientIteration, t, paramIter);
        } else if (BuildConfig.DEBUG) {
            Log.d("sendWeight", "Can't send weights.");
        }
    }

    private void sendUserValues(List<Double> gradientsOrWeights, boolean gradientProcessed, int gradIter, int weightIter, int paramIter){
        userRef.setValue(
            new UserData(gradientsOrWeights, gradientProcessed, gradIter, weightIter, paramIter)
        );
    }
}
