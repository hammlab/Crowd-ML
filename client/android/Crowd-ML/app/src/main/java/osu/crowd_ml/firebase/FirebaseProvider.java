package osu.crowd_ml.firebase;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.support.annotation.Nullable;

import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;

/**
 * Created by tylerzeller on 3/6/17.
 */

public class FirebaseProvider extends ContentProvider {

    private static boolean created = false;

    @Override
    public boolean onCreate() {
        if (created) {
            // Workaround for https://code.google.com/p/android/issues/detail?id=172655
            return false;
        }

        Context context = getContext();
        FirebaseApp.initializeApp(context, FirebaseOptions.fromResource(context));
        created = true;

        // Return false to mimic behavior of FirebaseInitProvider.
        // It should keep the pseudo ContentProvider from being a real one.
        return false;
    }

    @Nullable
    @Override
    public Cursor query(Uri uri, String[] strings, String s, String[] strings1, String s1) {
        return null;
    }

    @Nullable
    @Override
    public String getType(Uri uri) {
        return null;
    }

    @Nullable
    @Override
    public Uri insert(Uri uri, ContentValues contentValues) {
        return null;
    }

    @Override
    public int delete(Uri uri, String s, String[] strings) {
        return 0;
    }

    @Override
    public int update(Uri uri, ContentValues contentValues, String s, String[] strings) {
        return 0;
    }
}
