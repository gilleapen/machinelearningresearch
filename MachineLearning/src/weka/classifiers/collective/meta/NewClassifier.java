/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.classifiers.collective.meta;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 *
 * @author huangdongshan
 */
public class NewClassifier  extends Classifier implements OptionHandler{
protected int m_NumberConfident = 10;
    public Enumeration listOptions() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void setOptions(String[] options) throws Exception {
        String tempstr;


        tempstr = Utils.getOption("conf", options);
        if (tempstr.length() != 0) {
            setNumberConfident(Integer.parseInt(tempstr));
        } else {
            setNumberConfident(10);
        }
    }

    public String[] getOptions() {
        Vector result;
        String[] options;
        result = new Vector();
        result.add("-conf");
        result.add("" + getNumberConfident());
        return (String[]) result.toArray(new String[result.size()]);
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

    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

}
