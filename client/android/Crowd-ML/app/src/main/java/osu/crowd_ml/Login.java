package osu.crowd_ml;

import android.annotation.SuppressLint;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import osu.crowd_ml.utils.NetworkUtils;
import osu.crowd_ml.utils.StringUtils;

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

public class Login extends AppCompatActivity implements ILoginView, View.OnClickListener {

    private static final int DEFAULT_BATCH_SIZE = 1;

    TextView mSignInStatus = null;
    TextView mServiceStatus = null;
    ToggleButton mWifiToggle = null;
    ProgressBar mProgress = null;
    ToggleButton mToggle = null;

    ILoginPresenter mPresenter = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        mPresenter = new SetupPresenter(this);
        mPresenter.onCreate();
    }

    @Override protected void onStart() {
        super.onStart();

        // Step 1. Get all views from the layout
        mSignInStatus = (TextView) findViewById(R.id.message);
        mServiceStatus = (TextView) findViewById(R.id.service_status);
        mWifiToggle = (ToggleButton) findViewById(R.id.wifi_toggle);

        mProgress = (ProgressBar) findViewById(R.id.progress);

        mToggle = (ToggleButton) findViewById(R.id.toggle_button);

        // Step 2. Register listeners to views, if necessary
        mToggle.setOnClickListener(this);
        mWifiToggle.setOnClickListener(this);

        // Step 3. Invoke the presenter's onStart callback
        mPresenter.onStart();
    }

    @Override public void onStop() {
        // Step 1. Perform the default stop actions
        super.onStop();

        // Step 2. Invoke the presenter's onStop callback
        mPresenter.onStop();
    }

    @Override public void onDestroy(){
        // Step 1. Destroy presenter first to prevent presenter from calling methods on a null view.
        mPresenter.onDestroy();

        // Step 2. Perform default destroy actions
        super.onDestroy();
    }

    @Override public void showCreatingUser() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // Step 1. Hide unecessary views.
                setViewsInvisible(mServiceStatus, mToggle, mWifiToggle);

                // Step 2. Update text on statuses.
                mSignInStatus.setText("Creating user.");

                // Step 3. Show necessary views.
                setViewsVisible(mProgress, mSignInStatus);
            }
        });
    }

    @Override public void showUserSigningIn() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // Step 1. Hide unecessary views.
                setViewsInvisible(mServiceStatus, mToggle, mWifiToggle);

                // Step 2. Update text on statuses.
                mSignInStatus.setText("Signing user in.");

                // Step 3. Show necessary views.
                setViewsVisible(mProgress, mSignInStatus);
            }
        });
    }

    @Override public void showUserSigninSuccess() {
        runOnUiThread(new Runnable() {
            @SuppressLint("DefaultLocale")
            @Override
            public void run() {
                setViewsInvisible(mProgress);

                // Step 1. Update text on statuses.
                mSignInStatus.setText("All Signed in!");

                // Step 2. Start the service.
                startService(new Intent(Login.this, TestService.class));

                // Step 3. Verify that the service is running.
                if (isServiceRunning(TestService.class)){
                    mServiceStatus.setText("Service is running. Check button to stop.");
                    mToggle.setChecked(true);
                } else {
                    mServiceStatus.setText("Service is not running. Check button to start.");
                    mToggle.setChecked(false);
                }

                // Step 4. For emulator device only! Check whether the wifi is toggled or not.
                if (NetworkUtils.isWifiOn){
                    mWifiToggle.setChecked(true);
                } else {
                    mWifiToggle.setChecked(false);
                }

                // Step 5. Show necessary views.
                setViewsVisible(mServiceStatus, mToggle, mWifiToggle);
            }
        });
    }

    @Override
    public void showErrorCreatingUser() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // Step 1.
                setViewsInvisible(mProgress, mServiceStatus, mToggle, mWifiToggle);

                // Step 2.
                mSignInStatus.setText("Could not create user.");

                // Step 3.
                setViewsVisible(mSignInStatus);
            }
        });
    }

    @Override
    public void showErrorSigningin() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // Step 1.
                setViewsInvisible(mProgress, mServiceStatus, mToggle, mWifiToggle);

                // Step 2.
                mSignInStatus.setText("Could not sign user in.");

                // Step 3.
                setViewsVisible(mSignInStatus);
            }
        });
    }

    @Override
    public void showUserOffline() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // Step 1.
                setViewsInvisible(mProgress, mServiceStatus, mToggle, mWifiToggle);

                // Step 2.
                mSignInStatus.setText("User went offline.");

                // Step 3.
                setViewsVisible(mSignInStatus);
                //TODO: Add manual sign in button
            }
        });
    }

    @Override public Context getContext(){
        return this;
    }

    private boolean isServiceRunning(Class<?> serviceClass) {
        ActivityManager manager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        for (ActivityManager.RunningServiceInfo service : manager.getRunningServices(Integer.MAX_VALUE)) {
            if (serviceClass.getName().equals(service.service.getClassName())) {
                return true;
            }
        }
        return false;
    }

    @SuppressLint("CommitPrefEdits")
    @Override
    public void onClick(View view) {
        ToggleButton tb = (ToggleButton) view;

        if (tb == mToggle){
            Log.d("ServiceToggle", "toggled");
            if(tb.isChecked()){

                // Step 1. Start the service.
                startService(new Intent(Login.this, TestService.class));
                Log.d("Starting test service", "lol");

                // Step 2. Verify the service is running.
                if (isServiceRunning(TestService.class)){
                    mServiceStatus.setText("Service is running. Check button to stop.");
                } else {
                    // Error
                    mServiceStatus.setText("There was an error starting the service.");
                    mToggle.setChecked(false);
                }
            } else {

                // Step 1. Stop the service
                stopService(new Intent(Login.this, TestService.class));
                Log.d("stopping test service", "lol");
                // Step 2. Verify the service is stopped.
                if (!isServiceRunning(TestService.class)){
                    mServiceStatus.setText("Service is not running. Check button to stop.");
                } else {
                    // Error
                    mServiceStatus.setText("There was an error stopping the service.");
                    mToggle.setChecked(true);
                }
            }
        } else if (tb == mWifiToggle){
            Log.d("WifiToggle", "toggled");
            if(tb.isChecked()){
                NetworkUtils.isWifiOn = true;
            } else {
                NetworkUtils.isWifiOn = false;
            }
        }

    }

    @Override
    public void addUserInfoToPreferences(Bundle userInfo){
        // Step 1. Get shared preferences and preferences editor.
//        MultiprocessPreferences.Editor editor = MultiprocessPreferences.getDefaultSharedPreferences(this).edit();
//
//        // Step 2. Extract necessary user info from bundle.
//        editor.putString("email", userInfo.getString("email"));
//        editor.putString("password", userInfo.getString("password"));
//        editor.putString("uid", userInfo.getString("uid"));
//
//        // Step 3. Apply the additions to shared preferences.
//        editor.apply();
    }

    private void setViewsInvisible (View... views){
        for (View view : views){
            if (view != null){
                view.setVisibility(View.INVISIBLE);
            }
        }
    }

    private void setViewsVisible (View... views){
        for (View view : views){
            if (view != null){
                view.setVisibility(View.VISIBLE);
            }
        }
    }
}
