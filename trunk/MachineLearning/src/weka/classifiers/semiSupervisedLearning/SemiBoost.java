/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.rules.ZeroR;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.ExpDistance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.matrix.Matrix;

/**
 *
 * @author Administrator
 */
public class SemiBoost extends RandomizableIteratedSingleClassifierEnhancer implements WeightedInstancesHandler,
        TechnicalInformationHandler {

    protected Classifier m_ZeroR;
    /** Use SemiBoost with reweighting? */
    protected boolean m_UseResampling;
    /** 训练池*/
    protected Instances trainPool;
    /** 已经标记的数据*/
    protected Instances labledDataSet;
    /** 未标记的数据*/
    protected Instances UnlabledDataSet;
    /** 已经标记的数目*/
    protected int m_numLabledData = 10;
    /** 随机分训练集*/
    protected boolean m_randomSplit = false;
    /** The number of successfully generated base classifiers. */
    protected int m_NumIterationsPerformed;
    /**相似矩阵*/
    protected Matrix m_simiMatrix;
    DistanceFunction distanceFunction = new ExpDistance();
   //计算p,q值
    double m_pValue=0.0;
    double m_qValue=0.0;

    public void setRandomSplit(boolean randomSplit) {
        m_randomSplit = randomSplit;
    }

    public boolean getRandomSplit() {
        return m_randomSplit;
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Pavan Kumar Mallapragada and Rong Jin and Anil k, Jain");
        result.setValue(Field.YEAR, "2008");
        result.setValue(Field.TITLE, "SemiBoost:Boosting for Semi-supervised learning");
        result.setValue(Field.JOURNAL, "IEEE Tanscation on Pattern Analysis and Machine Intelligence");
        result.setValue(Field.VOLUME, "No");
        result.setValue(Field.NUMBER, "no");
        result.setValue(Field.PAGES, "no");
        return result;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.0 $");
    }

    @Override
    /**
     *获得分类器的处理能力，主要从类的数据类型来考虑的
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAllClasses();
        result.disableAllClassDependencies();
        //Nominal 类属性
        if (super.getCapabilities().handles(Capability.NOMINAL_CLASS));
        result.enable(Capability.NOMINAL_CLASS);
        if (super.getCapabilities().handles(Capability.BINARY_CLASS)) {
            result.enable(Capability.BINARY_CLASS);
        }
        return result;
    }

    @Override
    public void buildClassifier(Instances traindata) throws Exception {
        super.buildClassifier(traindata);
        //检测训练数据的类型
        getCapabilities().testWithFail(traindata);
        traindata = new Instances(traindata);
        traindata.deleteWithMissingClass();//删除类缺失的实例
        // only class? -> build ZeroR model
        if (traindata.numAttributes() == 1) {
            System.err.println(
                    "Cannot build model (only class attribute present in data!), " + "using ZeroR model instead!");
            m_ZeroR = new weka.classifiers.rules.ZeroR();
            m_ZeroR.buildClassifier(traindata);
            return;
        } else {
            m_ZeroR = null;
        }
        //如果不需要重抽样
        if (!getUseResampling() && (m_Classifier instanceof WeightedInstancesHandler)) {
            buildClassifierWithWeights(traindata);
        } else {
            buildClassifierUsingResampling(traindata);
        }

    }

    /**
     * 获得训练集与测试集的相似度矩阵；
     * @param instances
     * @return
     */
    private void computeSimilarityMatrix(Instances instances) {
        //
        Matrix simiMatrix;
        int numInstances = instances.numInstances();
        simiMatrix = new Matrix(numInstances, numInstances);
        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numInstances; j++) {
                double tempValue = distanceFunction.distance(
                        instances.instance(i), instances.instance(j));
                simiMatrix.set(i, j, tempValue);
            }
        }
     m_simiMatrix=simiMatrix;
       
    }
    protected Matrix getSimilarityMatrix(){
        return m_simiMatrix;
    }

    /**
     *
     * @param traindata
     */
    protected void buildClassifierWithWeights(Instances traindata) throws Exception {
        //将训练数据分成有标记的和没有标记的数据
        SemiBoostSplitTrainSet splitTrain = new SemiBoostSplitTrainSet(traindata);
        Random randomInstance = new Random(m_Seed);
        Matrix simiMatrix;
        splitTrain.setRandom(randomInstance);
        splitTrain.splitTrainSet(m_numLabledData, m_randomSplit);
        Evaluation evaluation;
        labledDataSet = splitTrain.getLabledData();
        UnlabledDataSet = splitTrain.getUnLabledData();
        //计算训练集(包括有标记的数据和没有标记的数据)的相似矩阵
        computeSimilarityMatrix(traindata);
        //每次训练的训练数据,数量会递增，不断从未标记的实例中提取实例加入到其中
        trainPool = new Instances(labledDataSet, 0, labledDataSet.numInstances());
        for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length;
                m_NumIterationsPerformed++) {
            if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable) {
                ((Randomizable) (m_Classifiers[m_NumIterationsPerformed])).setSeed(randomInstance.nextInt());
                m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainPool);

                evaluation = new Evaluation(UnlabledDataSet);
                evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed], UnlabledDataSet);
            }
        }

    }
    /**
     * 计算p value
     */
    protected void computePvalue(){

    }
    /**
     * 计算q 值
     */
    protected void computeQvalue(){
    }
    /**
     * 计算a 值
     */
    protected void computeAlpha(){
    }
    protected void buildClassifierUsingResampling(Instances data) {
    }

    public void setUseResampling(boolean useResampling) {
        m_UseResampling = useResampling;
    }

    public boolean getUseResampling() {
        return m_UseResampling;
    }

    public void setNumLabledData(int numLabledData) {
        m_numLabledData = numLabledData;
    }

    public int getNumLabledData() {
        return m_numLabledData;
    }

    @Override
    public void setOptions(String[] options) throws Exception {

        setDebug(Utils.getFlag('D', options));
    }

    /**
     *
     * @return
     */
    @Override
    public String[] getOptions() {

        String[] options;
        if (getDebug()) {
            options = new String[1];
            options[0] = "-D";
        } else {
            options = new String[0];
        }
        return options;
    }
}
