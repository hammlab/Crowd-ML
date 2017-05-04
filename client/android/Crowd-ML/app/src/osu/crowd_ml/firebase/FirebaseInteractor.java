package osu.crowd_ml.firebase;

import android.content.Context;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.util.Log;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.UUID;

import osu.crowd_ml.R;

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

public class FirebaseInteractor implements IFirebaseInteractor {

    private static FirebaseInteractor instance = new FirebaseInteractor();

    public static FirebaseInteractor getInstance() {
        return instance;
    }

    private FirebaseAuth mAuth;
    private FirebaseAuth.AuthStateListener mAuthListener;
    private String mEmail = null;
    private String mPassword = null;
    private String mUid = null;

    private HashSet<AuthStateListener> mListeners;

    // Called once when instance is first requested
    private FirebaseInteractor() {
        // Step 1. Initialize all variables
        mListeners = new HashSet<>();
        mAuth = FirebaseAuth.getInstance();

        // Step 2. Register a single, distributed firebase auth listener
        mAuthListener = new FirebaseAuth.AuthStateListener() {
            @Override public void onAuthStateChanged(@NonNull FirebaseAuth firebaseAuth) {

                // Step 1. Get the current firebase user
                FirebaseUser user = firebaseAuth.getCurrentUser();

                // Step 2. Check if the resulting user is signed in or not
                if (user != null) {
                    // User is signed in
                    mUid = user.getUid();
                    Bundle userInfo = new Bundle();
                    userInfo.putString("email", mEmail);
                    userInfo.putString("password", mPassword);
                    userInfo.putString("uid", mUid);
                    alertSignIn(userInfo);
                } else {
                    // User is signed out
                    alertSignOut();
                }
            }
        };

        mAuth.addAuthStateListener(mAuthListener);
    }

    // Helper method. Alert all registered listeners of a sign in.
    private void alertSignIn(Bundle userInfo){
        for (AuthStateListener listener : mListeners){
            listener.onSignIn(userInfo);
        }
    }

    // Helper method. Alert all registered listeners of a sign out.
    private void alertSignOut(){
        for (AuthStateListener listener : mListeners){
            listener.onSignOut();
        }
    }

    // Register a new listener.
    public void addAuthStateListener(AuthStateListener listener){
        assert mListeners != null;

        if (!mListeners.contains(listener)){
            mListeners.add(listener);
            Log.d("FirebaseInteractor", "Listener added.");
        } else {
            Log.d("FirebaseInteractor", "Listener already exists.");
        }
        Log.d("FirebaseInteractor", mListeners.size() + "");
    }

    // Remove an existing listener.
    public void removeAuthStateListener(AuthStateListener listener){
        assert mListeners != null;

        if (mListeners.contains(listener)){
            mListeners.remove(listener);
            Log.d("FirebaseInteractor", "Listener removed.");
        } else {
            Log.d("FirebaseInteractor", "Listener does not exist.");
        }
        Log.d("FirebaseInteractor", mListeners.size() + "");
    }

    public void destroyFirebaseListener(){
        if (mAuthListener != null) {
            mAuth.removeAuthStateListener(mAuthListener);
        }
        mListeners.clear();
    }

    @Override
    public void createUserAccount(final OnCreateUserListener listener) {
        // Step 1. check if user exists or not
        boolean exists;
        try {
            exists = userExists();
        } catch (IOException e){
            e.printStackTrace();
            // Error opening settings file, so notify listener and exit
            listener.onCreateUserFailure();
            return;
        }

        if (!exists){
            // TODO: change this garb
            // Step 2. Generate unique username and password for this user
            mEmail = UUID.randomUUID().toString().replaceAll("-", "") + "@gmail.com";
            mPassword = "password";

            // TODO: addOnFailureListener() instead?
            // Step 3. Register user with Firebase
            mAuth.createUserWithEmailAndPassword(mEmail, mPassword)
            .addOnCompleteListener(new OnCompleteListener<AuthResult>() {
                @Override
                public void onComplete(@NonNull Task<AuthResult> task) {
                    // Step 4. Check whether creation was successful
                    if (!task.isSuccessful()) {

                        // Step 5. If not successful, notify presenter
                        listener.onCreateUserFailure();
                    } else {

                        // Step 5. If successful, save the data, catching any errors
                        boolean error = saveLogin();

                        // Step 6. If save was successful or not, notify presenter accordingly
                        if (error){
                            listener.onCreateUserFailure();
                        } else {
                            listener.onCreateUserSuccess();
                        }
                    }

                }
            });
        } else {
            // Step 2. If user already exists, nothing to create
            listener.onUserAlreadyExists();
        }

    }

    @Override
    public void signInUser(final OnSigninUserListener listener) {
        String email = null;
        String pass = null;
        boolean error = false;

        try {
            email = getEmail();
            pass = getPassword();
        } catch (IOException e) {
            e.printStackTrace();
            error = true;
        }

        if (error || email == null || pass == null){
            listener.onSigninUserFailure();
            return;
        }

        mAuth.signInWithEmailAndPassword(email, pass)
        .addOnCompleteListener(new OnCompleteListener<AuthResult>() {
            @Override
            public void onComplete(@NonNull Task<AuthResult> task) {
                Log.d("FirebaseInteractor", "signInWithEmail:onComplete:" + task.isSuccessful());

                // TODO: invoke BackgroundDataSend when wifi is enabled
//                        if (task.isSuccessful()) {
//                            Intent intent = new Intent(Login.this, DataSend.class);
//                            intent.putExtra("EMAIL", email);
//                            intent.putExtra("PASSWORD", password);
//                            intent.putExtra("UID", uid);
//                            startActivity(intent);
//                        }

                if (task.isSuccessful()) {
                    listener.onSigninUserSuccess();
                } else {
                    listener.onSigninUserFailure();
                }

            }
        });
    }

    private boolean userExists() throws IOException {
        for (String filename : CrowdMLApplication.getAppContext().fileList()){
            if (filename.equals(CrowdMLApplication.getAppContext().getString(R.string.user_settings_file))){
                return true;
            }
        }
        return false;
    }

    private boolean saveLogin() {
        String FILENAME = CrowdMLApplication.getAppContext().getString(R.string.user_settings_file);
        String string = mEmail + "\n" + mPassword;

        FileOutputStream fos = null;
        try {
            fos = CrowdMLApplication.getAppContext().openFileOutput(FILENAME, Context.MODE_PRIVATE);
            fos.write(string.getBytes());
            fos.close();
        } catch (IOException e){
            e.printStackTrace();
            return true;
        }
        return false;
    }

    private String getEmail() throws IOException {
        if (mEmail == null){
            String FILENAME = CrowdMLApplication.getAppContext().getString(R.string.user_settings_file);
            FileInputStream fis = CrowdMLApplication.getAppContext().openFileInput(FILENAME);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis));
            mEmail = br.readLine();
            mPassword = br.readLine();
            br.close();
        }

        return mEmail;
    }

    private String getPassword() throws IOException {
        if (mPassword == null){
            String FILENAME = CrowdMLApplication.getAppContext().getString(R.string.user_settings_file);
            FileInputStream fis = CrowdMLApplication.getAppContext().openFileInput(FILENAME);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis));
            mEmail = br.readLine();
            mPassword = br.readLine();
            br.close();
        }

        return mPassword;
    }
}
