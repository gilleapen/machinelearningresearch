/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.core;

/**
 *
 * @author huangdongshan
 */
public class ExpDistance extends NormalizableDistance implements TechnicalInformationHandler {
    private double sigma=0.43;
    public ExpDistance(Instances data) {
        super(data);
    }

    public ExpDistance() {
    }

    public static long getSerialVersionUID() {
        return serialVersionUID;
    }

    public double getSigma() {
        return sigma;
    }

    public void setSigma(double sigma) {
        this.sigma = sigma;
    }

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected double updateDistance(double currDist, double diff) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public String getRevision() {
        String extract = RevisionUtils.extract("v1.0");
        return extract;
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    /** for serialization */
    private static final long serialVersionUID = -2173635135622930167L;

    @Override
    public double distance(Instance first, Instance second) {
        double distance = 0.0;
        double temp = 0.0;
        temp = Math.sqrt(distance(first, second, Double.POSITIVE_INFINITY));
        distance = Math.exp(temp / (-1 * sigma));

        return distance;
    }
}
