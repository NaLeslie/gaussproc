package gaussproc.covariance;

import java.util.Arrays;
import linear_algebra.Operations;
import linear_algebra.SingularMatrixException;

/**
 * Contains method for creating and evaluating a Gaussian process with a squared exponential covariance function.
 * @author Nathaniel
 */
public class SquaredExponential implements GaussianProcess{
    
    /**
     * Creates a Gaussian process based on a squared exponential covariance function.
     * Hyperparameters controlling the covariance function are:
     * <ul>
     * <li>Covariance amplitude</li>
     * <li>Dimension-specific covariance length scale</li>
     * <li>Measurement variance</li>
     * </ul>
     * These hyperparameters are optimized using Gauss-Newton iterations from a variety of starting points.
     * @param y The list y-values to be modelled by the Gaussian process as a list of points. {@code y[point_index]}
     * @param X The list of x-values to be modelled by the Gaussian process as a list of vectors.
     * Indexed by: <code>X[point_index][dimension_index]</code>
     */
    public SquaredExponential(double[] y, double[][] X){

        //Initialize hyperparameters
        int dimension = X[0].length;
        ell = new double[dimension];

        double[] t0s = new double[]{-4, -2, 0};
        double[] tis = new double[]{-6, -4, -2, -1};
        double[] tss = new double[]{0.00001, 0.001, 0.1};
        
        for(int i = 0; i < dimension; i ++){
            ell[i] = tis[0];
        }
        ell_zero = t0s[0];
        sigma = tss[0];
        optimizeHyperparameters(X, y);
        
        int numparams = ell.length+2;
        double[] hyperparameters = new double[numparams];
        hyperparameters[0] = ell_zero;
        for(int i = 0; i < ell.length; i++){
            hyperparameters[i+1] = ell[i];
        }
        hyperparameters[numparams - 1] = sigma;
        
        double[] best_hyperparameters = new double[numparams];
        System.arraycopy(hyperparameters, 0, best_hyperparameters, 0, numparams);
        
        double best_liklihood = logProbability(X, y, hyperparameters);
        
        for(double t0:t0s){
            for(double ti: tis){
                for(double ts:tss){
                    for(int i = 0; i < dimension; i ++){
                        ell[i] = ti;
                    }
                    ell_zero = t0;
                    sigma = ts;
                    optimizeHyperparameters(X, y);
                    hyperparameters[0] = ell_zero;
                    for(int i = 0; i < ell.length; i++){
                        hyperparameters[i+1] = ell[i];
                    }
                    hyperparameters[numparams - 1] = sigma;
                    double liklihood = logProbability(X, y, hyperparameters);
                    if(liklihood > best_liklihood){
                        best_liklihood = liklihood;
                        System.arraycopy(hyperparameters, 0, best_hyperparameters, 0, numparams);
                    }
                }
            }
        }
        this.X = X;
        this.y = y;
        
        System.arraycopy(best_hyperparameters, 0, hyperparameters, 0, numparams);
        
        ell_zero = best_hyperparameters[0];
        for(int i = 0; i < ell.length; i++){
            ell[i] = best_hyperparameters[i+1];
        }
        sigma = best_hyperparameters[numparams - 1];
        
        System.out.println("\n\nFINAL:");
        System.out.println(Arrays.toString(hyperparameters));
        System.out.println("log(liklihood): " + best_liklihood);
        
        this.Kinv = Operations.invertHermitian(constructK(X, hyperparameters));
//        System.out.println("\n\nKinv:");
//        LinearAlgebra.printMatrix(this.Kinv);
    }
    
    /**
     * Evaluates the value of this Gaussian process at position x.
     * @param x The position at which the Gaussian process is to be evaluated as a vector.
     * @return the value of this Gaussian process at position x.
     */
    @Override
    public double getValue(double[] x){
        int numparams = ell.length+2;
        double[] hyperparameters = new double[numparams];
        //build hyperparameters and shift vector
        hyperparameters[0] = ell_zero;
        for(int i = 0; i < ell.length; i++){
            hyperparameters[i+1] = ell[i];
        }
        hyperparameters[numparams - 1] = sigma;
        //INVERT Kinv!!!!!
        double[] kXxstar = construct_kXxstar(x, hyperparameters);
        double[] Kiy = Operations.multiply(Kinv, y);
        return Operations.innerProduct(kXxstar, Kiy);
    }
    
    /**
     * Evaluates the variance of this Gaussian process at position x.
     * @param x the position at which to evaluate the variance as a vector.
     * @return The variance of this Gaussian process at position x.
     */
    @Override
    public double getVariance(double[] x){
        int numparams = ell.length+2;
        double[] hyperparameters = new double[numparams];
        //build hyperparameters and shift vector
        hyperparameters[0] = ell_zero;
        for(int i = 0; i < ell.length; i++){
            hyperparameters[i+1] = ell[i];
        }
        hyperparameters[numparams - 1] = sigma;
        double[] kXxstar = construct_kXxstar(x, hyperparameters);
        double[] KikXxstar = Operations.multiply(Kinv, kXxstar);
        double subtract = Operations.innerProduct(kXxstar, KikXxstar);
        double autocovariance = covariance(x, x, hyperparameters);
        return autocovariance-subtract;
    }
    
    /**
     * Determines the covariance between <code>x</code> and every point in {@link #X} (kXxstar).
     * @param x The position for which kXxstar is to be evaluated <code>x[dimension_index]</code>.
     * @param hyperparameters The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return The covariance between <code>x</code> and {@link #X} as a vector kXxstar[point_index].
     */
    private double[] construct_kXxstar(double[] x, double[] hyperparameters){
        int numpts = X.length;
        int dim = X[0].length;
        double[] kXxstar = new double[numpts];
        for(int i = 0; i < numpts; i++){
            double[] a = new double[dim];
            for(int m = 0; m < dim; m++){
                a[m] = X[i][m];
            }
            kXxstar[i] = covariance(a, x, hyperparameters);
        }
        return kXxstar;
    }
    
    /**
     * Computes the covariance between points {@code a} and {@code b} following a squared exponential
     * @param a Point {@code a} as a vector
     * @param b Point {@code b} as a vector
     * @param hyperparams The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return The covariance between points {@code a} and {@code b}.
     */
    private double covariance(double[] a, double[] b, double[] hyperparams){
        double sum = 0;
        for(int i = 0; i < a.length; i++){
            double diff = a[i]-b[i];
            double term = diff*diff*Math.exp(hyperparams[i+1]);
            sum += term;
        }
        return Math.exp(hyperparams[0]-sum);
    }
    
    /**
     * Constructs the covariance matrix (contains the covariance between every point in {@code X}.
     * @param X The points as a tensor {@code X[point_index][dimension_index]}
     * @param hyperparams The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return The covariance matrix as a tensor {@code Kxx[point_index][point_index]}.
     */
    private double[][] constructKxx(double[][] X, double[] hyperparams){
        int numpts = X.length;
        int dim = X[0].length;
        double[][] kxx = new double[numpts][numpts];
        for(int i = 0; i < numpts; i++){
            double[] a = new double[dim];
            for(int m = 0; m < dim; m++){
                a[m] = X[i][m];
            }
            for(int j = i; j < numpts; j++){
                double[] b = new double[dim];
                for(int m = 0; m < dim; m++){
                    b[m] = X[j][m];
                }
                //covariance is symmetric
                double cov = covariance(a, b, hyperparams);
                kxx[i][j] = cov;
                kxx[j][i] = cov;
            }
        }
        return kxx;
    }
    
    /**
     * Evaluates the derivative of {@code (Kinv(X,X)+sigma^2*I)} with respect to the {@code index}-th hyperparameter
     * @param index the hyperparameter index.
     * <p>index = 0 --> ell_zero</p>
     * <p>index > 0 ; dim => index --> ell[index - 1]</p>
     * <p>index = dim+1 --> sigma</p>
     * @param X The points as a tensor {@code X[point_index][dimension_index]}
     * @param hyperparams The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return the derivative of {@code (Kinv(X,X)+sigmaI)} with respect to the {@code index}-th hyperparameter
     */
    private double[][] dKdTheta(int index, double[][] X, double[] hyperparams){
        if(index > 0 && index <= ell.length){
            int numpts = X.length;
            int dim = X[0].length;
            double[][] dkxxdl = new double[numpts][numpts];
            for(int i = 0; i < numpts; i++){
                double[] a = new double[dim];
                for(int m = 0; m < dim; m++){
                        a[m] = X[i][m];
                    }
                for(int j = i; j < numpts; j++){
                    double[] b = new double[dim];
                    for(int m = 0; m < dim; m++){
                        b[m] = X[j][m];
                    }
                    //covariance is symmetric
                    double cov = covariance(a, b, hyperparams);
                    double diff = (a[index - 1] - b[index - 1]);
                    double diffsq = -diff*diff*Math.exp(hyperparams[index]);
                    dkxxdl[i][j] = cov*diffsq;
                    dkxxdl[j][i] = cov*diffsq;
                }
            }
            return dkxxdl;
        }
        else if(index == 0){
            int numpts = X.length;
            int dim = X[0].length;
            double[][] dkxxda = new double[numpts][numpts];
            for(int i = 0; i < numpts; i++){
                double[] a = new double[dim];
                for(int m = 0; m < dim; m++){
                        a[m] = X[i][m];
                    }
                for(int j = i; j < numpts; j++){
                    double[] b = new double[dim];
                    for(int m = 0; m < dim; m++){
                        b[m] = X[j][m];
                    }
                    //covariance is symmetric
                    double cov = covariance(a, b, hyperparams);
                    dkxxda[i][j] = cov;
                    dkxxda[j][i] = cov;
                }
            }
            return dkxxda;
        }
        else if(index == ell.length + 1){
            int numpts = X.length;
            double[][] dkdsigma = new double[numpts][numpts];
            for(int i = 0; i < numpts; i++){
                dkdsigma[i][i] = 2*hyperparams[ell.length + 1];
                for(int j = i+1; j < numpts; j++){
                    dkdsigma[i][j] = 0;
                    dkdsigma[j][i] = 0;
                }
            }
            return dkdsigma;
        }
        else{
            throw new ArrayIndexOutOfBoundsException("index was too large to refer to a hyperparameter.");
            
        }
    }
    
    /**
     * Constructs the covariance matrix + sigma<sup>2</sup><b>I</b>
     * @param X The points as a tensor {@code X[point_index][dimension_index]} 
     * @param hyperparams The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return The full covariance matrix
     */
    private double[][] constructK(double[][] X, double[] hyperparams){
        double[][] K = constructKxx(X, hyperparams);
        double sig = hyperparams[hyperparams.length-1];
        for(int i = 0; i < K.length; i++){
            K[i][i] += sig*sig;
        }
        return K;
    }
    
    /**
     * Uses Cholesky decomposition and back-substitution to invert K as calculated by {@link #constructK(double[][], double[]) } O(n<sup>3</sup>)
     * @param X The points as a tensor {@code X[point_index][dimension_index]} 
     * @param hyperparams The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return The inverse of {@link #constructK(double[][], double[]) }
     */
    private double[][] invertK(double[][] X, double[] hyperparams){
        return Operations.invertHermitian(constructK(X, hyperparams));
    }
    
    /**
     * Optimizes the hyperparameters for this Gaussian process
     * @param X The points as a tensor {@code X[point_index][dimension_index]}
     * @param y The list y-values to be modelled by the Gaussian process as a list of points. {@code y[point_index]}
     */
    private void optimizeHyperparameters(double[][] X, double[] y){
        int numparams = ell.length+2;
        double[] residuals = new double[numparams];
        double[][] jacobian = new double[numparams][numparams];
        double[] shift = new double[numparams];
        
        double[] hyperparams = new double[numparams];
        double[] perturbedparams = new double[numparams];
        double[] temporaryhyperparams = new double[numparams];
        
        //build hyperparameters and shift vector
        hyperparams[0] = ell_zero;
        for(int i = 0; i < ell.length; i++){
            hyperparams[i+1] = ell[i];
        }
        hyperparams[numparams - 1] = sigma;
        
        for(int i = 0; i < numparams; i++){
            shift[i] = 0;
        }
        
        //build residuals
        double[][] K_inverse = invertK(X, hyperparams);

//        System.out.println("\n\nK-1:");
//        GaussProcMain.printMatrix(K_inverse);
        
        for(int i = 0; i < numparams; i++){
            residuals[i] = -logProbabilityDerivative(X, y, i, K_inverse, hyperparams);
        }
        
        int iterations = 10;
        double last_l2;
        double l2 = Operations.l2norm_square(residuals);
        double last_logprob;
        double logprob = logProbability(X, y, hyperparams);
        do{
            last_l2 = l2;
            last_logprob = logprob;
            //update hyperparameters (note shift is zero prior to the first iteration)
            for(int i = 0; i < numparams; i++){
                hyperparams[i] += shift[i];
            }
            
//            System.out.println(Arrays.toString(hyperparams));
            
            
            
//            System.out.println("log marginal liklihood: " + last_logprob);
            
            //create perturbations of 5% or 1E-5 whichever is highest.
            for(int i = 0; i < numparams; i++){
                perturbedparams[i] = 1.01*hyperparams[i];
                if(Math.abs(0.01*hyperparams[i]) < 0.00005){
                    perturbedparams[i] = 0.00005 + hyperparams[i];
                }
            }

            //build the Jacobian
            for(int i = 0; i < numparams; i++){
                double[] perturb = new double[numparams];
                System.arraycopy(hyperparams, 0, perturb, 0, numparams);
                perturb[i] = perturbedparams[i];
                double run = perturbedparams[i]-hyperparams[i];
                double[][] perturbed_K_inv = invertK(X, perturb);
                
//                System.out.println("\n\nK-1 (" + i + "):");
//                LinearAlgebra.printMatrix(perturbed_K_inv);
                
                for(int j = 0; j < numparams; j++){
                    double rise = residuals[j]  + logProbabilityDerivative(X, y, j, perturbed_K_inv, perturb);
                    jacobian[j][i] = rise / run;
                }
            }
//            System.out.println("Jacobian " + iterations + ":");
//            LinearAlgebra.printMatrix(jacobian);
            try{
            double[][] jacobian_inv = Operations.invert(jacobian);
            shift = Operations.multiply(jacobian_inv, residuals);
            }
            catch(SingularMatrixException e){
//                System.out.println(Arrays.toString(hyperparams));
            }
            
            
            for(int i = 0; i < numparams; i++){
                temporaryhyperparams[i] = hyperparams[i] + shift[i];
            }
            
            //build residuals
            
            K_inverse = invertK(X, temporaryhyperparams);
            

            for(int i = 0; i < numparams; i++){
                residuals[i] = -logProbabilityDerivative(X, y, i, K_inverse, temporaryhyperparams);
            }
            l2 = Operations.l2norm_square(residuals);
            logprob = logProbability(X, y, temporaryhyperparams);
//            System.out.println("l2norm: " + l2);
            iterations --;
        }while(iterations > 0 && l2 < last_l2 && logprob > last_logprob);
        
//        System.out.println("\n\nFINAL:");
        if(l2 < last_l2 && logprob > last_logprob){
            ell_zero = temporaryhyperparams[0];
            System.arraycopy(temporaryhyperparams, 1, ell, 0, numparams-2);
            sigma = temporaryhyperparams[temporaryhyperparams.length - 1];
//            System.out.println(Arrays.toString(temporaryhyperparams));
//            System.out.println("L2: " + l2);
//            System.out.println("log(liklihood): " + logprob);
        }
        else{
            ell_zero = hyperparams[0];
            System.arraycopy(hyperparams, 1, ell, 0, numparams-2);
            sigma = hyperparams[hyperparams.length - 1];
//            System.out.println(Arrays.toString(hyperparams));
//            System.out.println("L2: " + last_l2);
//            System.out.println("log(liklihood): " + last_logprob);
        }
        
    }
    
    /**
     * Computes the derivative of the log liklihood that the data is modelled by the Gaussian process with respect to the {@code index}-th hyperparameter
     * @param X The points as a tensor {@code X[point_index][dimension_index]}
     * @param y The list y-values to be modelled by the Gaussian process as a list of points. {@code y[point_index]}
     * @param index The hyperparameter for which the derivative is to be computed.
     * @param K_inverse The inverse of the covariance matrix passed-in as a tensor since this method will be called many times for the same covariance matrix.
     * @param hyperparams The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return The derivative of the log liklihood with respect to the {@code index}-th hyperparameter
     */
    private double logProbabilityDerivative(double[][] X, double[] y, int index, double[][] K_inverse, double[] hyperparams){
        double[] K_inv_y = Operations.multiply(K_inverse, y);
        double[][] outerprod = Operations.outerProduct(K_inv_y, K_inv_y);
        double[][] left_factor = Operations.subtract(outerprod, K_inverse);

        double[][] product = Operations.multiply(left_factor, dKdTheta(index, X, hyperparams));
        double trace = Operations.trace(product);
        return 0.5*trace;
    }
    
    /**
     * Computes the log liklihood that the data matches the Gaussian process
     * @param X The points as a tensor {@code X[point_index][dimension_index]}
     * @param y The list y-values to be modelled by the Gaussian process as a list of points. {@code y[point_index]}
     * @param hyperparams The hyperparameters to be used to control the squared-exponential covariance.
     * <ul>
     * <li><code>hyperparams[0]</code> = l0</li>
     * <li><code>hyperparams[i]</code> = li (0 &lt; <code>i</code> &lt; <code>hyperparams.length-1</code>)</li>
     * <li><code>hyperparams[hyperparams.length-1]</code> = sigma</li>
     * </ul>
     * @return The log liklihood that the data matches the Gaussian process
     */
    private double logProbability(double[][] X, double[] y, double[] hyperparams){
        double[][] K_inv = invertK(X, hyperparams);
        double[] Kiy = Operations.multiply(K_inv, y);
        double yKiy = Operations.innerProduct(y, Kiy);
        double detK = Operations.determinantHermitian(constructK(X, hyperparams));
        double n = y.length;
        double twicelogprob = 0.0 - yKiy - Math.log(detK) - n*Math.log(2.0*Math.PI);
        return 0.5*twicelogprob;
    }
    
    /**
     * The inverse of the covariance matrix
     */
    private double[][] Kinv;
    
    /**
     * The log length scales for the covariance along each dimension
     */
    private double[] ell;
    
    /**
     * The log of the amplitude hyperparameter
     */
    private double ell_zero;
    
    /**
     * The variance hyperperameter
     */
    private double sigma;
    
    /**
     * The position data used to produce this Gaussian process {@code X[point_index][dimension_index]}
     */
    private double[][] X;
    
    /**
     * The signal data used to produce this Gaussian process {@code y[point_index]}
     */
    private double[] y;
}
