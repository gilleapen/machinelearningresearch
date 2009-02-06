/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.collective.meta;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.classifiers.collective.CollectiveRandomizableMultipleClassifiersCombiner;
import weka.classifiers.collective.util.CollectiveHelper;
import weka.core.Instance;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.RevisionUtils.Type;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 *
 * @author huangdongshan
 */
public class CollectiveSemiSupervisedLearning
        extends CollectiveRandomizableMultipleClassifiersCombiner
        implements Summarizable {
    //信任度最高的前10条记录

    Classifier[] cls;
    protected int m_NumberConfident = 10;
    protected int m_CutOff;

    protected void initializeMembers() {
        super.initializeMembers();
        Classifier[] cls = new Classifier[2];
        cls[0] =  new weka.classifiers.trees.J48();
        cls[1] =  new weka.classifiers.trees.J48();
        setClassifiers(cls);
        m_NumberConfident = 10;
    }

    @Override
    //设置参数的时候一定要用setNumberConfident
    public void setOptions(String[] options) throws Exception {
        String tempstr;


        tempstr = Utils.getOption("conf", options);
        if (tempstr.length() != 0) {
            setNumberConfident(Integer.parseInt(tempstr));
        } else {
            setNumberConfident(10);
        }

        String tmpStr;
        tmpStr = Utils.getOption("cut-off", options);
        if (tmpStr.length() != 0) {
            setCutOff(Integer.parseInt(tmpStr));
        } else {
            setCutOff(0);
        }
        super.setOptions(options);

        // fix the classifiers
        if (getClassifiers().length < 2) {
            System.out.println("Less than two classifiers - defaulting to two J48 instances!");
            cls = new Classifier[2];
            cls[0] =  new weka.classifiers.trees.J48();
            cls[1] =  new weka.classifiers.trees.J48();
            setClassifiers(cls);
        }
    }

    @Override
    public String[] getOptions() {
        Vector result;
        String[] options;
        result = new Vector();
        result.add("-conf");
        result.add("" + getNumberConfident());
        result.add("-cut-off");
        result.add("" + getCutOff());
        options = super.getOptions();
        for (int i = 0; i < options.length; i++) {
            result.add(options[i]);
        }
        return (String[]) result.toArray(new String[result.size()]);

    }

    public Enumeration listOptions() {
        Vector result = new Vector();

        result.addElement(new Option(
                "\tthe number of the most confident\n" + "\t(default 10)",
                "conf", 1, "-conf <num>"));
        result.addElement(new Option(
                "\tNumber of folds after which to stop the adding of folds.\n" + "\t0 means no cut-off at all. (default 0)",
                "cut-off", 1, "-cut-off <num>"));
        Enumeration en = super.listOptions();
        while (en.hasMoreElements()) {
            result.addElement(en.nextElement());
        }

        return result.elements();
    }

    public void setNumberConfident(int value) {
        if (value >= 0) {
            m_NumberConfident = value;
        } else {
            System.out.println("NumberConfident must be 1 (no cut-off) or greater than 1 " + " (provided: " + value + ")!");
        }
    }

    public int getNumberConfident() {
        return m_NumberConfident;
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public int getCutOff() {
        return m_CutOff;
    }

    public void setCutOff(int value) {
        if (value >= 0) {
            m_CutOff = value;
        } else {
            System.out.println("Cut-Off must be 0 (no cut-off) or greater than 0 " + " (provided: " + value + ")!");
        }
    }

    @Override
    protected double[] getDistribution(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected void buildClassifier() throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected void build() throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    /*
     * getRevison
     */

    public String getRevision() {
        return RevisionUtils.extract("￥Revision: 1.0 ￥");
    }

    public String toSummaryString() {
        return CollectiveHelper.generateMD5(Utils.joinOptions(getOptions()));
    }

    public static void main(String[] args) {
        runClassifier(new CollectiveSemiSupervisedLearning(), args);
    }
}
