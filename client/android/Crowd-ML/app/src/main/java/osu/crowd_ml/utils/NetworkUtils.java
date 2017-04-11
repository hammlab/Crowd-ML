package osu.crowd_ml.utils;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

import java.io.IOException;
import java.io.InterruptedIOException;

/**
 * Created by tylerzeller on 1/29/17.
 */

public final class NetworkUtils {

    // Prevent instantiation
    private NetworkUtils(){}

    private static final int TRIES = 3;

    public static boolean isWifiOn = false;

    public static boolean isWifiConnected(Context context) {
//        ConnectivityManager connectivityManager = (ConnectivityManager)
//                context.getSystemService(Context.CONNECTIVITY_SERVICE);
//        NetworkInfo networkInfo = null;
//        if (connectivityManager != null) {
//            networkInfo = connectivityManager.getActiveNetworkInfo();
//        }
//
//        return networkInfo != null && networkInfo.getState() == NetworkInfo.State.CONNECTED && networkInfo.getType() == ConnectivityManager.TYPE_WIFI;
        return true;
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
