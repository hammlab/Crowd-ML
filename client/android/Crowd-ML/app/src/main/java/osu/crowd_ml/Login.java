package osu.crowd_ml;

import android.content.Context;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

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

public class Login extends AppCompatActivity implements ILoginView {

    TextView mMessage = null;
    ProgressBar mProgress = null;
    ILoginPresenter mPresenter = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.content_login);
        mPresenter = new SetupPresenter(this);
        mPresenter.onCreate();
        //Firebase.setAndroidContext(this);
    }

    @Override protected void onStart() {
        super.onStart();

        mMessage = (TextView) findViewById(R.id.message);
        mProgress = (ProgressBar) findViewById(R.id.progress);
        mPresenter.onStart();
    }

    @Override public void onStop() {
        super.onStop();
        mPresenter.onStop();
    }

    @Override public void onDestroy(){
        // Destroy in this order to prevent presenter from calling methods on a null view
        mPresenter.onDestroy();
        super.onDestroy();
    }

    @Override public void showCreatingUser() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(Login.this, "Creating user. Please wait.", Toast.LENGTH_LONG).show();
                mProgress.setVisibility(View.VISIBLE);
            }
        });
    }

    @Override public void showUserSigningIn() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(Login.this, "Signing user in. Please wait.", Toast.LENGTH_LONG).show();
                mProgress.setVisibility(View.VISIBLE);
            }
        });
    }

    @Override public void showUserSigninSuccess() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(Login.this, "User signed in.", Toast.LENGTH_LONG).show();
                mProgress.setVisibility(View.INVISIBLE);
            }
        });
    }

    @Override
    public void showErrorCreatingUser() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(Login.this, "Could not create user.", Toast.LENGTH_LONG).show();
                mProgress.setVisibility(View.INVISIBLE);
            }
        });
    }

    @Override
    public void showErrorSigningin() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(Login.this, "Could not sign user in.", Toast.LENGTH_LONG).show();
                mProgress.setVisibility(View.INVISIBLE);
            }
        });
    }

    @Override
    public void showUserOffline() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(Login.this, "User went offline.", Toast.LENGTH_LONG).show();
                mProgress.setVisibility(View.INVISIBLE);
            }
        });
    }

    @Override public Context getContext(){
        return this;
    }
}
