package osu.crowd_ml;

import android.content.Context;

public interface ILoginView {

    void showCreatingUser();

    void showUserSigningIn();

    void showUserSigninSuccess();

    void showErrorCreatingUser();

    void showErrorSigningin();

    void showUserOffline();

    Context getContext();

}
