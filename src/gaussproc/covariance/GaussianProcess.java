package gaussproc.covariance;

/**
 * Interface for extracting values and variance from Gaussian processes at a given point in input space.
 * @author Nathaniel
 */
public interface GaussianProcess {
    
    /**
     * Computes the mean value of the Gaussian process at position <code>x</code> in input space.
     * @param x The position in input space at which the Gaussian process is to be evaluated.
     * @return The mean value of the Gaussian process at position <code>x</code> in input space.
     */
    public double getValue(double[] x);
    
    /**
     * Computes the variance of the Gaussian process at position <code>x</code> in input space.
     * @param x The position in input space at which the variance is to be evaluated.
     * @return The variance of the Gaussian process at position <code>x</code> in input space.
     */
    public double getVariance(double[] x);
}