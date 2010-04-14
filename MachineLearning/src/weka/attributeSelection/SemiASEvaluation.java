/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.attributeSelection;

import weka.core.Instances;

/**
 *
 * @author Administrator
 */
public interface SemiASEvaluation {
    /**
     *
     * @param labeledData
     * @param unlabledData
     * @throws Exception
     */
    public abstract void buildEvaluator(Instances labeledData, Instances unlabledData) throws Exception;

}
