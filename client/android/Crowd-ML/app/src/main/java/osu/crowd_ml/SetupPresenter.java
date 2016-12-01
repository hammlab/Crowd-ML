package osu.crowd_ml;

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

public class SetupPresenter implements ILoginPresenter, IFirebaseInteractor.AuthStateListener,
        IFirebaseInteractor.OnCreateUserListener, IFirebaseInteractor.OnSigninUserListener {

    private ILoginView mView;

    public SetupPresenter(ILoginView view){
        mView = view;
    }

    @Override
    public void onCreate(){
        FirebaseInteractor.getInstance().addAuthStateListener(this);
    }

    @Override
    public void onStart() {
        if (mView != null){
            mView.showCreatingUser();
            FirebaseInteractor.getInstance().createUserAccount(this);
        }
    }

    @Override
    public void onStop() {
        FirebaseInteractor.getInstance().removeAuthStateListener(this);
    }

    @Override
    public void onDestroy() {
        mView = null;
    }

    @Override
    public void onSignIn() { // Unnecessary to handle here
    }

    @Override
    public void onSignOut() {
        if (mView != null){
            mView.showUserOffline();
        }
    }

    @Override
    public void onCreateUserSuccess() {
        if (mView != null){
            mView.showUserSigninSuccess();
        }
    }

    @Override
    public void onCreateUserFailure() {
        if (mView != null){
            mView.showErrorCreatingUser();
        }
    }

    @Override public void onUserAlreadyExists(){
        if (mView != null){
            mView.showUserSigningIn();
            FirebaseInteractor.getInstance().signInUser(this);
        }
    }

    @Override
    public void onSigninUserSuccess() {
        if (mView != null){
            mView.showUserSigninSuccess();
        }
    }

    @Override
    public void onSigninUserFailure() {
        if (mView != null){
            mView.showErrorSigningin();
        }
    }
}
