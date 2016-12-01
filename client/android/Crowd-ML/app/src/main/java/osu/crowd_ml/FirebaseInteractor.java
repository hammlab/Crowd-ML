package osu.crowd_ml;

/**
 * Created by tylerzeller on 11/28/16.
 */
public class FirebaseInteractor {
    private static FirebaseInteractor ourInstance = new FirebaseInteractor();

    public static FirebaseInteractor getInstance() {
        return ourInstance;
    }

    private FirebaseInteractor() {
    }
}
