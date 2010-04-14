/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.attributeSelection;

import java.util.ArrayList;
import java.util.Enumeration;
import weka.core.Instances;
import weka.core.OptionHandler;

/**
 *
 * @author Administrator
 */
public class SemiReliefAttributeEval extends ASEvaluation implements SemiASEvaluation,
        AttributeEvaluator, OptionHandler{

    @Override
    public void buildEvaluator(Instances data) throws Exception {

        Instances labeledData = null;
        Instances unlabledData = null;


        buildEvaluator(labeledData,  unlabledData);
         throw new UnsupportedOperationException("Not supported yet.");
    }

    public void buildEvaluator(Instances labeledData, Instances unlabledData) throws Exception {

        ArrayList labeledPair = null;
        buildEvaluator(labeledPair,  unlabledData);
        throw new UnsupportedOperationException("Not supported yet.");

    }
    private void buildEvaluator(ArrayList labeledPair, Instances unlabledData)throws Exception{}

    public double evaluateAttribute(int attribute) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public Enumeration listOptions() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void setOptions(String[] options) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public String[] getOptions() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

}
