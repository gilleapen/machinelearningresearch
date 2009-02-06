/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.collective.meta;

import weka.classifiers.collective.CollectiveRandomizableSingleClassifierEnhancer;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

/**
 *
 * @author huangdongshan
 */
public class SemiBoost extends CollectiveRandomizableSingleClassifierEnhancer {
  //初始距离函数为欧拉距离公式
    DistanceFunction distanceFunction = new EuclideanDistance();
    //标记的数目，初始值是10；
    int labledNumber=10;
    Instances labledData=null;

    public void setLabledNumber(int labledNumber) {
        this.labledNumber = labledNumber;
    }

    public int getLabledNumber() {
        return labledNumber;
    }
    public void setDistanceFunction(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
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

        Instances trainSet = getTrainingSet();
        Instances testSet = getTestSet();
        //测试集和训练集实例的总数
        int numInstances = trainSet.numInstances() + testSet.numInstances();
        //合并训练集和测试集
        Instances unpropossedInstances = Instances.mergeInstances(trainSet, testSet);

    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

//获得训练集与测试集的相似度矩阵；
    private Matrix getSimilarityMatrix(Instances instances) {
        //
        Matrix simiMatrix;
        int numInstances=instances.numInstances();
        simiMatrix = new Matrix(numInstances, numInstances);
        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numInstances; j++) {

                double tempValue = distanceFunction.distance(
                        instances.instance(i), instances.instance(j));
                simiMatrix.set(i, j, tempValue);
            }
        }
        return simiMatrix;
    }
}
