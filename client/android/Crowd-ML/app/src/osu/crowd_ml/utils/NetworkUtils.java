package osu.crowd_ml.utils;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

import java.io.IOException;

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

public final class NetworkUtils {

    // Prevent instantiation
    private NetworkUtils(){}

    private static final int TRIES = 3;

    public static boolean isWifiOn = false;

    public static boolean isWifiConnected(Context context) {
        ConnectivityManager connectivityManager = (ConnectivityManager)
                context.getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo networkInfo = null;
        if (connectivityManager != null) {
            networkInfo = connectivityManager.getActiveNetworkInfo();
        }

        return networkInfo != null && networkInfo.getState() == NetworkInfo.State.CONNECTED &&
                networkInfo.getType() == ConnectivityManager.TYPE_WIFI;
        //return true;
    }
    public static boolean isOnline() throws IOException, InterruptedException{
        try {
            boolean isOnline = false;
            int i = 0;
            Runtime runtime = Runtime.getRuntime();
            while (!isOnline && i < TRIES){
                Process ipProcess = runtime.exec("/system/bin/ping -c 1 8.8.8.8");
                int exitValue = ipProcess.waitFor();
                isOnline = (exitValue == 0);
                i++;
            }
            return isOnline;
        } catch (IOException | InterruptedException e) { throw e; }
    }

}
