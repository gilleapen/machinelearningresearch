/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import weka.core.Instances;

/**
 *用来将训练集分成labelData和unlabeldData;
 * @author Administrator
 */
public interface SplitTrainSet {

    /**
     * 分开数据集
     */
    void splitTrainSet();
    void splitTrainSet(int numLabledData,boolean randomSplit);
    void splitTrainSet(double percentLabeled,boolean randomSplit);
     /**
     * 获得标记的数据
     * @return
     */
    Instances getLabledData();

    /**
     * 获得未标记的数据
     * @return
     */
    Instances getUnLabledData();
}
