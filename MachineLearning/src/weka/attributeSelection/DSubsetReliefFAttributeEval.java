package weka.attributeSelection;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

public class DSubsetReliefFAttributeEval
        extends ASEvaluation
        implements SubsetEvaluator,
        OptionHandler,
        TechnicalInformationHandler {

    /** The training instances */
    private Instances m_trainInstances;
    /**
     * The number of instances to sample when estimating attributes
     * default == -1, use all instances
     */
    private int m_sampleM;
    /** The number of nearest hits/misses */
    private int m_Knn;
    private int m_sigma;
    /** Weight by distance rather than equal weights */
    private boolean m_weightByDistance;
    /** Random number seed used for sampling instances */
    private int m_seed;
    private int m_numAttribs;
    private int m_classIndex;            //类别属性的属性索引号
    private int m_numInstances;
    private boolean m_isNumericClass;
    private int[] m_stored;
    private double[][][] m_karray;
    private int[] m_index;
    private double[] m_worst;
    private double[] m_minArray;
    private double[] m_maxArray;
    private int m_numClasses;
    private int m_totalInstances;
    private double[] m_classProbs;

    /**
     * Constructor
     */
    public DSubsetReliefFAttributeEval() {
        resetOptions();
    }

    @Override
    public void buildEvaluator(Instances data) throws Exception {

        // can evaluator handle data?
        getCapabilities().testWithFail(data);

        m_trainInstances = new Instances(data);
        m_trainInstances.deleteWithMissingClass();
        m_classIndex = m_trainInstances.classIndex();
        m_numAttribs = m_trainInstances.numAttributes();
        m_numInstances = m_trainInstances.numInstances();
        m_isNumericClass = m_trainInstances.attribute(m_classIndex).isNumeric();

        if (!m_isNumericClass) {
            m_numClasses = m_trainInstances.attribute(m_classIndex).numValues();
        } else {
            m_numClasses = 1;
        }

        m_karray = new double[m_numClasses][m_Knn][2];

        if (!m_isNumericClass) {
            m_classProbs = new double[m_numClasses];

            for (int i = 0; i < m_numInstances; i++) {
                m_classProbs[(int) m_trainInstances.instance(i).value(m_classIndex)]++;
            }

            for (int i = 0; i < m_numClasses; i++) {
                m_classProbs[i] /= m_numInstances;
            }
        }
        m_worst = new double[m_numClasses];
        m_index = new int[m_numClasses];
        m_stored = new int[m_numClasses];
        m_minArray = new double[m_numAttribs];
        m_maxArray = new double[m_numAttribs];

        for (int i = 0; i < m_numAttribs; i++) {
            m_minArray[i] = m_maxArray[i] = Double.NaN;
        }

        for (int i = 0; i < m_numInstances; i++) {
            updateMinMax(m_trainInstances.instance(i));
        }

        if ((m_sampleM > m_numInstances) || (m_sampleM < 0)) {
            m_totalInstances = m_numInstances;
        } else {
            m_totalInstances = m_sampleM;
        }

        //throw new UnsupportedOperationException("Not supported yet.");
    }

    public double evaluateSubset(BitSet subset) throws Exception {
        double weight = 0;
        int z;
        Random r = new Random(m_seed);
        // process each instance, updating attribute weights
        for (int i = 0; i < m_totalInstances; i++) {   //m_totalInstances为抽样次数
            if (m_totalInstances == m_numInstances) {
                z = i;
            } else {
                z = r.nextInt() % m_numInstances;
            }
            if (z < 0) {
                z *= -1;
            }                                          //z为本次抽取的数据记录索引号

            if (!(m_trainInstances.instance(z).isMissing(m_classIndex))) {
                // first clear the knn and worst index stuff for the classes
                for (int j = 0; j < m_numClasses; j++) {
                    m_index[j] = m_stored[j] = 0;

                    for (int k = 0; k < m_Knn; k++) {
                        m_karray[j][k][0] = m_karray[j][k][1] = 0;
                    }
                }
                findKHitMiss(z, subset);
                double[] diff = calSubsetDiff(z, subset);
                int cl = (int) m_trainInstances.instance(z).value(m_classIndex);

                double sameClassDiff = diff[cl];
                double diffClassDiff = calDiffClassDiff(z, diff);

                //weight = weight - sameClassDiff / m_totalInstances + diffClassDiff / m_totalInstances;
                weight = weight + diffClassDiff / m_totalInstances;

                if (Double.isNaN(sameClassDiff) || sameClassDiff > 1) {
                    System.out.println("sameClassDiff=" + sameClassDiff + "\n");
                }
                if (Double.isNaN(diffClassDiff) || diffClassDiff > 1) {
                    System.out.println("diffClassDiff=" + diffClassDiff + "\n");
                }

            }
        }

        //throw new UnsupportedOperationException("Not supported yet.");
        if (Double.isNaN(weight) || weight > 1) {
            System.out.println("weight=" + weight + "\n");
        }
        return 1 - Math.abs(weight);
    }

    /**
     * Returns a string describing this attribute evaluator
     * @return a description of the evaluator suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "SubsetReliefFAttributeEval :\n\nEvaluates the worth of an attribute by "
                + "repeatedly sampling an instance and considering the value of the "
                + "given attribute for the nearest instance of the same and different "
                + "class. Can operate on both discrete and continuous class data.\n\n"
                + "For more information see:\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns the capabilities of this evaluator.
     *
     * @return            the capabilities of this evaluator
     * @see               Capabilities
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Returns an enumeration describing the available options.
     * @return an enumeration of all the available options.
     **/
    public Enumeration listOptions() {
        Vector newVector = new Vector(4);
        newVector.addElement(new Option("\tSpecify the number of instances to\n"
                + "\tsample when estimating attributes.\n"
                + "\tIf not specified, then all instances\n"
                + "\twill be used.", "M", 1, "-M <num instances>"));
        newVector.addElement(new Option("\tSeed for randomly sampling instances.\n"
                + "\t(Default = 1)", "D", 1, "-D <seed>"));
        newVector.addElement(new Option("\tNumber of nearest neighbours (k) used\n"
                + "\tto estimate attribute relevances\n"
                + "\t(Default = 10).", "K", 1, "-K <number of neighbours>"));
        newVector.addElement(new Option("\tWeight nearest neighbours by distance", "W", 0, "-W"));
        newVector.addElement(new Option("\tSpecify sigma value (used in an exp\n"
                + "\tfunction to control how quickly\n"
                + "\tweights for more distant instances\n"
                + "\tdecrease. Use in conjunction with -W.\n"
                + "\tSensible value=1/5 to 1/10 of the\n"
                + "\tnumber of nearest neighbours.\n"
                + "\t(Default = 2)", "A", 1, "-A <num>"));
        return newVector.elements();
    }

    public void setOptions(String[] options) throws Exception {
        String optionString;
        resetOptions();
        setWeightByDistance(Utils.getFlag('W', options));
        optionString = Utils.getOption('M', options);

        if (optionString.length() != 0) {
            setSampleSize(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption('D', options);

        if (optionString.length() != 0) {
            setSeed(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption('K', options);

        if (optionString.length() != 0) {
            setNumNeighbours(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption('A', options);

        if (optionString.length() != 0) {
            setWeightByDistance(true); // turn on weighting by distance
            setSigma(Integer.parseInt(optionString));
        }
    }

    /**
     * Gets the current settings of ReliefFAttributeEval.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {
        String[] options = new String[9];
        int current = 0;

        if (getWeightByDistance()) {
            options[current++] = "-W";
        }

        options[current++] = "-M";
        options[current++] = "" + getSampleSize();
        options[current++] = "-D";
        options[current++] = "" + getSeed();
        options[current++] = "-K";
        options[current++] = "" + getNumNeighbours();

        if (getWeightByDistance()) {
            options[current++] = "-A";
            options[current++] = "" + getSigma();
        }

        while (current < options.length) {
            options[current++] = "";
        }

        return options;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        TechnicalInformation additional;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Kenji Kira and Larry A. Rendell");
        result.setValue(Field.TITLE, "A Practical Approach to Feature Selection");
        result.setValue(Field.BOOKTITLE, "Ninth International Workshop on Machine Learning");
        result.setValue(Field.EDITOR, "Derek H. Sleeman and Peter Edwards");
        result.setValue(Field.YEAR, "1992");
        result.setValue(Field.PAGES, "249-256");
        result.setValue(Field.PUBLISHER, "Morgan Kaufmann");

        additional = result.add(Type.INPROCEEDINGS);
        additional.setValue(Field.AUTHOR, "Igor Kononenko");
        additional.setValue(Field.TITLE, "Estimating Attributes: Analysis and Extensions of RELIEF");
        additional.setValue(Field.BOOKTITLE, "European Conference on Machine Learning");
        additional.setValue(Field.EDITOR, "Francesco Bergadano and Luc De Raedt");
        additional.setValue(Field.YEAR, "1994");
        additional.setValue(Field.PAGES, "171-182");
        additional.setValue(Field.PUBLISHER, "Springer");

        additional = result.add(Type.INPROCEEDINGS);
        additional.setValue(Field.AUTHOR, "Marko Robnik-Sikonja and Igor Kononenko");
        additional.setValue(Field.TITLE, "An adaptation of Relief for attribute estimation in regression");
        additional.setValue(Field.BOOKTITLE, "Fourteenth International Conference on Machine Learning");
        additional.setValue(Field.EDITOR, "Douglas H. Fisher");
        additional.setValue(Field.YEAR, "1997");
        additional.setValue(Field.PAGES, "296-304");
        additional.setValue(Field.PUBLISHER, "Morgan Kaufmann");

        return result;
    }

    /**
     * Reset options to their default values
     */
    protected void resetOptions() {
        m_trainInstances = null;
        m_sampleM = -1;
        m_Knn = 10;
        m_sigma = 2;
        m_weightByDistance = false;
        m_seed = 1;
    }

    /**
     * Set the nearest neighbour weighting method
     *
     * @param b true nearest neighbours are to be weighted by distance.
     */
    public void setWeightByDistance(boolean b) {
        m_weightByDistance = b;
    }

    /**
     * Set the number of instances to sample for attribute estimation
     *
     * @param s the number of instances to sample.
     */
    public void setSampleSize(int s) {
        m_sampleM = s;
    }

    /**
     * Set the random number seed for randomly sampling instances.
     *
     * @param s the random number seed.
     */
    public void setSeed(int s) {
        m_seed = s;
    }

    /**
     * Set the number of nearest neighbours
     *
     * @param n the number of nearest neighbours.
     */
    public void setNumNeighbours(int n) {
        m_Knn = n;
    }

    /**
     * Sets the sigma value.
     *
     * @param s the value of sigma (> 0)
     * @throws Exception if s is not positive
     */
    public void setSigma(int s)
            throws Exception {
        if (s <= 0) {
            throw new Exception("value of sigma must be > 0!");
        }

        m_sigma = s;
    }

    /**
     * Get whether nearest neighbours are being weighted by distance
     *
     * @return m_weightByDiffernce
     */
    public boolean getWeightByDistance() {
        return m_weightByDistance;
    }

    /**
     * Get the number of instances used for estimating attributes
     *
     * @return the number of instances.
     */
    public int getSampleSize() {
        return m_sampleM;
    }

    /**
     * Get the seed used for randomly sampling instances.
     *
     * @return the random number seed.
     */
    public int getSeed() {
        return m_seed;
    }

    /**
     * Get the number of nearest neighbours
     *
     * @return the number of nearest neighbours
     */
    public int getNumNeighbours() {
        return m_Knn;
    }

    /**
     * Get the value of sigma.
     *
     * @return the sigma value.
     */
    public int getSigma() {
        return m_sigma;
    }

    /**
     * Find the K nearest instances to supplied instance if the class is numeric,
     * or the K nearest Hits (same class) and Misses (K from each of the other
     * classes) if the class is discrete.
     *
     * @param instNum the index of the instance to find nearest neighbours of
     */
    private void findKHitMiss(int instNum, final BitSet subset) {
        int i, j;
        int cl;
        double ww;
        double temp_diff = 0.0;
        Instance thisInst = m_trainInstances.instance(instNum);

        for (i = 0; i < m_numInstances; i++) {
            if (i != instNum) {
                Instance cmpInst = m_trainInstances.instance(i);
                temp_diff = calSubsetNormDistance(instNum, i, subset);

                // class of this training instance or 0 if numeric
                if (m_isNumericClass) {
                    cl = 0;                //如果是类别是数值型，则令类别号均为0
                } else {
                    cl = (int) m_trainInstances.instance(i).value(m_classIndex);
                }

                // add this diff to the list for the class of this instance
                //若第cl类数据当前计算出的距离数目未达到m_Knn
                if (m_stored[cl] < m_Knn) {                   //m_stored[cl]为第cl类数据当前已经计算出的距离数目
                    m_karray[cl][m_stored[cl]][0] = temp_diff; //m_karray[cl][m_stored[cl]]为第cl类数据当前计算的距离
                    m_karray[cl][m_stored[cl]][1] = i;        //其0号数据存放距离值temp_diff，1号数据存放当前数据记录的索引号
                    m_stored[cl]++;                          //增添一个距离记录，将距离数目加1

                    // note the worst diff for this class
                    //标记出第cl类数据目前计算出的最大距离
                    for (j = 0, ww = -1.0; j < m_stored[cl]; j++) {
                        if (m_karray[cl][j][0] > ww) {
                            ww = m_karray[cl][j][0];
                            m_index[cl] = j;                  //标记m_karray中第j个距离为cl类数据的最大距离
                        }
                    }

                    m_worst[cl] = ww;                   //标记第cl类数据的最大距离值为ww
                } else //若第cl类数据当前计算出的距离数目已经达到m_Knn
                /* if we already have stored knn for this class then check to
                see if this instance is better than the worst */ {
                    if (temp_diff < m_karray[cl][m_index[cl]][0]) {        //若当前距离小于当前最大距离值，则将最大距离替换为当前距离
                        m_karray[cl][m_index[cl]][0] = temp_diff;
                        m_karray[cl][m_index[cl]][1] = i;

                        for (j = 0, ww = -1.0; j < m_stored[cl]; j++) {     //重新标记最大距离
                            if (m_karray[cl][j][0] > ww) {
                                ww = m_karray[cl][j][0];
                                m_index[cl] = j;
                            }
                        }

                        m_worst[cl] = ww;
                    }
                }
            }
        }
    }

    /**
     * Calculates the distance between two instances
     *
     * @param first the first instance
     * @param second the second instance
     * @return the distance between the two given instances, between 0 and 1
     */
    private double distance(Instance first, Instance second) {

        double distance = 0;
        int firstI, secondI;

        for (int p1 = 0, p2 = 0;
                p1 < first.numValues() || p2 < second.numValues();) {
            if (p1 >= first.numValues()) {
                firstI = m_trainInstances.numAttributes();
            } else {
                firstI = first.index(p1);
            }
            if (p2 >= second.numValues()) {
                secondI = m_trainInstances.numAttributes();
            } else {
                secondI = second.index(p2);
            }
            if (firstI == m_trainInstances.classIndex()) {
                p1++;
                continue;
            }
            if (secondI == m_trainInstances.classIndex()) {
                p2++;
                continue;
            }
            double diff;
            if (firstI == secondI) {
                diff = difference(firstI,
                        first.valueSparse(p1),
                        second.valueSparse(p2));
                p1++;
                p2++;
            } else if (firstI > secondI) {
                diff = difference(secondI,
                        0, second.valueSparse(p2));
                p2++;
            } else {
                diff = difference(firstI,
                        first.valueSparse(p1), 0);
                p1++;
            }
            distance += diff * diff;
            //distance += diff;
        }

        //return Math.sqrt(distance / m_NumAttributesUsed);
        return Math.sqrt(distance / m_trainInstances.numAttributes());
        //return distance;
    }

    /**
     * Computes the difference between two given attribute
     * values.
     */
    private double difference(int index, double val1, double val2) {

        switch (m_trainInstances.attribute(index).type()) {
            case Attribute.NOMINAL:

                // If attribute is nominal
                if (Instance.isMissingValue(val1)
                        || Instance.isMissingValue(val2)) {
                    return (1.0 - (1.0 / ((double) m_trainInstances.attribute(index).numValues())));
                } else if ((int) val1 != (int) val2) {
                    return 1;
                } else {
                    return 0;
                }
            case Attribute.NUMERIC:

                // If attribute is numeric
                if (Instance.isMissingValue(val1)
                        || Instance.isMissingValue(val2)) {
                    if (Instance.isMissingValue(val1)
                            && Instance.isMissingValue(val2)) {
                        return 1;
                    } else {
                        double diff;
                        if (Instance.isMissingValue(val2)) {
                            diff = norm(val1, index);
                        } else {
                            diff = norm(val2, index);
                        }
                        if (diff < 0.5) {
                            diff = 1.0 - diff;
                        }
                        return diff;
                    }
                } else {
                    return Math.abs(norm(val1, index) - norm(val2, index));
                }
            default:
                return 0;
        }
    }

    /**
     * Normalizes a given value of a numeric attribute.
     *
     * @param x the value to be normalized
     * @param i the attribute's index
     * @return the normalized value
     */
    private double norm(double x, int i) {
        if (Double.isNaN(m_minArray[i])
                || Utils.eq(m_maxArray[i], m_minArray[i])) {
            return 0;
        } else {
            return (x - m_minArray[i]) / (m_maxArray[i] - m_minArray[i]);
        }
    }

    /**
     * Updates the minimum and maximum values for all the attributes
     * based on a new instance.
     *
     * @param instance the new instance
     */
    private void updateMinMax(Instance instance) {
        //    for (int j = 0; j < m_numAttribs; j++) {
        try {
            for (int j = 0; j < instance.numValues(); j++) {
                if ((instance.attributeSparse(j).isNumeric())
                        && (!instance.isMissingSparse(j))) {
                    if (Double.isNaN(m_minArray[instance.index(j)])) {
                        m_minArray[instance.index(j)] = instance.valueSparse(j);
                        m_maxArray[instance.index(j)] = instance.valueSparse(j);
                    } else {
                        if (instance.valueSparse(j) < m_minArray[instance.index(j)]) {
                            m_minArray[instance.index(j)] = instance.valueSparse(j);
                        } else {
                            if (instance.valueSparse(j) > m_maxArray[instance.index(j)]) {
                                m_maxArray[instance.index(j)] = instance.valueSparse(j);
                            }
                        }
                    }
                }
            }
        } catch (Exception ex) {
            System.err.println(ex);
            ex.printStackTrace();
        }
    }

    /**
     * Return a description of the ReliefF attribute evaluator.
     *
     * @return a description of the evaluator as a String.
     */
    @Override
    public String toString() {
        StringBuffer text = new StringBuffer();

        if (m_trainInstances == null) {
            text.append("ReliefF feature evaluator has not been built yet\n");
        } else {
            text.append("\tReliefF Ranking Filter");
            text.append("\n\tInstances sampled: ");

            if (m_sampleM == -1) {
                text.append("all\n");
            } else {
                text.append(m_sampleM + "\n");
            }

            text.append("\tNumber of nearest neighbours (k): " + m_Knn + "\n");

            if (m_weightByDistance) {
                text.append("\tExponentially decreasing (with distance) "
                        + "influence for\n"
                        + "\tnearest neighbours. Sigma: "
                        + m_sigma + "\n");
            } else {
                text.append("\tEqual influence nearest neighbours\n");
            }
        }

        return text.toString();
    }

    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.23 $");
    }

    /**
     *
     * @param index  抽样数据记录的索引号
     * @param subset  评价子集的标记数组
     * @return
     */
    private double[] calSubsetDiff(int index, final BitSet subset) {
        double[] diff = new double[m_numClasses];

        int cl = (int) m_trainInstances.instance(index).value(m_classIndex);
        int k = m_stored[cl];
        if (k != m_Knn) {
            System.out.println("Knn 过大!  k=" + k);
        }

        for (int c = 0; c < m_numClasses; c++) {
            for (int i = 0; i < k; i++) {
                diff[c] = diff[c] + distance(m_trainInstances.instance(index), m_trainInstances.instance((int) m_karray[c][i][1]));
            }
            diff[c] = diff[c] / k;

        }

        return diff;
        //throw new UnsupportedOperationException("Not yet implemented");
    }

    private double calSubsetNormDistance(int thisIdx, int cmpIdx, final BitSet subset) {

        if (thisIdx == cmpIdx) {
            return 0;
        }

        double dist = 0;
        int subSize = subset.size();
        Instance thisInstance = m_trainInstances.instance(thisIdx);
        Instance cmpInstance = m_trainInstances.instance(cmpIdx);

        for (int i = 0; i < subSize; i++) {
            if (subset.get(i)) {
//                double x = norm(thisInstance.value(i), i);
//                double y = norm(cmpInstance.value(i), i);
//                dist = dist + (x - y) * (x - y);
                double d = difference(i, thisInstance.value(i), cmpInstance.value(i));
                dist = dist + d * d;
            }
        }

        dist = dist / subSize;
        dist = Math.sqrt(dist);

        return dist;
    }

    private double[] calDiffClassProbs(int index) {
        double[] p = new double[m_numClasses];
        int cl = (int) m_trainInstances.instance(index).value(m_classIndex);
        for (int c = 0; c < m_numClasses; c++) {
            if (c != cl) {
                p[c] = m_classProbs[c] / (1 - m_classProbs[cl]);
            }
        }
        p[cl] = 1;
        return p;
        //throw new UnsupportedOperationException("Not yet implemented");
    }

    private double calDiffClassDiff(int index, final double[] diff) {
        double result = 0;
        double[] p = calDiffClassProbs(index);
        int cl = (int) m_trainInstances.instance(index).value(m_classIndex);
        for (int c = 0; c < m_numClasses; c++) {
            if (c != cl) {
                result = result + p[c] * diff[c];
            }
        }
        return result;
        //throw new UnsupportedOperationException("Not yet implemented");
    }
}
