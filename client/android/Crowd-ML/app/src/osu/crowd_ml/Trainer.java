package osu.crowd_ml;

import java.util.List;

/**
 * Created by tylerzeller on 5/10/17.
 */

public interface Trainer {

    double[] train(final int numIterations);

    List<Double> getNoisyGrad();

    Trainer setIter(final int t);

    Trainer setWeights(double[] weights);

    Trainer setParams(final Parameters params);

    void destroy();

    //void interrupt();
}
