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

    public static boolean isWifiOnForEmulator = false;
}
