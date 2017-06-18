package osu.crowd_ml;

import android.content.Context;
import android.os.Bundle;

public interface ILoginView {

    void showCreatingUser();

    void showUserSigningIn();

    void showUserSigninSuccess();

    void showErrorCreatingUser();

    void showErrorSigningin();

    void showUserOffline();

    void addUserInfoToPreferences(Bundle userInfo);

    Context getContext();

}
