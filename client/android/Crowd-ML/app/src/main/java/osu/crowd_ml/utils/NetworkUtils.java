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
        return isWifiOn;
    }
    public static boolean isOnline() throws IOException, InterruptedException{

        Runtime runtime = Runtime.getRuntime();
        try {

            Process ipProcess = runtime.exec("/system/bin/ping -c 1 8.8.8.8");
            int     exitValue = ipProcess.waitFor();
            return (exitValue == 0);

        } catch (IOException | InterruptedException e) { throw e; }
    }

}
