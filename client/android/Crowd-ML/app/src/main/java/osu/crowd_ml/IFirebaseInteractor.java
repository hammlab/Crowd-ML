package osu.crowd_ml;

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
        void onSignIn();

        void onSignOut();
    }

    void createUserAccount(OnCreateUserListener listener);

    void signInUser(OnSigninUserListener listener);


}
