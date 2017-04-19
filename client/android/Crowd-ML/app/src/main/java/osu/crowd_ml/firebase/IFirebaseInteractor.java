package osu.crowd_ml.firebase;

import android.os.Bundle;

/**
 * Created by tylerzeller on 11/28/16.
 */

public interface IFirebaseInteractor {

    interface OnCreateUserListener {

        void onCreateUserSuccess();

        void onCreateUserFailure();

        void onUserAlreadyExists();
    }

    interface OnSigninUserListener {
        void onSigninUserSuccess();

        void onSigninUserFailure();
    }

    interface AuthStateListener {
        void onSignIn(Bundle userInfo);

        void onSignOut();
    }

    void createUserAccount(OnCreateUserListener listener);

    void signInUser(OnSigninUserListener listener);

}
