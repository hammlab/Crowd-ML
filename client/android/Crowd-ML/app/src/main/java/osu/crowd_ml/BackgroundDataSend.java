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
import android.os.IBinder;
import android.os.PowerManager;
import android.support.annotation.Nullable;
import android.support.v4.app.NotificationCompat;
import android.util.Log;

import osu.crowd_ml.utils.DataSender;


/**
 * Handles the service management.
 * Managing the work threads and the firebase connection.
 */
public class BackgroundDataSend extends Service {

    // Handling WiFi connectivity
    private boolean network;
    private BroadcastReceiver receiver;
    private volatile boolean isWifiConnected = false;
    private boolean wifiDisconnect = false;

    // Wakelock
    private PowerManager.WakeLock wakeLock;

    // Data interaction
    private DataSender dataSender;
    private String UID;

    // Bebugging
    private boolean VERBOSE_DEBUG = true;


    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();

        // TODO(david soller) : fix these comments
        // Step 1. Extract necessary information
        UID = MultiprocessPreferences.getDefaultSharedPreferences(this).getString("uid", "");

        // Step 3. Initialize necessary data.
        dataSender = new DataSender(UID, this);

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
                dataSender.setIsWifiConnected(isWifiConnected);
                dataSender.addFirebaseListeners();
            }
        } else {
            // Step 3. Check if wifi was previously connected.
            if (isWifiConnected) {
                dataSender.stopWorkThread();
                if (VERBOSE_DEBUG) {
                    Log.d("handleMessage", "Handling wifi disconnect.");
                }

                wifiDisconnect = true;
                isWifiConnected = false;
                dataSender.setWifiDisconnect(wifiDisconnect);
                dataSender.setIsWifiConnected(isWifiConnected);
                dataSender.removeFirebaseListeners();
            }
        }
    }

    @Override
    public void onDestroy() {

        Log.d("onDestroy", "Unregister Wifi receiver.");
        // Step 1. End the wifi receiver.
        unregisterReceiver(receiver);

        Log.d("onDestroy", "Stopping the worker thread.");
        // Step 2. End the worker thread, if running.
        dataSender.stopWorkThread();

        Log.d("onDestroy", "Removing Listeners.");
        // Step 3. Remove listeners.
        dataSender.removeFirebaseListeners();

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
}
