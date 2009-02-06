/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.matrix.Matrix;
import java.lang.Math;
import java.util.Enumeration;
import weka.core.Instance;

/**
 *2008-12-16
 * @author huangdongshan
 */
public class MultiSemiAdaBoost extends RandomizableIteratedSingleClassifierEnhancer
        implements WeightedInstancesHandler, Sourcable {

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
    protected int m_numLabledData = 90;
    /** 随机分训练集*/
    protected boolean m_randomSplit = false;
    /** The number of successfully generated base classifiers. */
    protected int m_NumIterationsPerformed;
    /**Beta系数*/
    private double[] m_Betas;
    /**只权重分类错误的*/
    public boolean m_onlyWeightIncorrect = false;
    /**
     * 每次挑选信任度实例的条数，默认值为10
     */
    private int numberIncreaseLabelbed = 10;
    private YValueMatrix yValueMatrix;
    /**标记的百分比*/
    private double m_percentLabeled = 10.0;
    /**信任度百分比*/
    double m_percentConfidce = 10.0;

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

    public void setPercentConfidce(double m_percentConfidce) {
        this.m_percentConfidce = m_percentConfidce;
    }

    public double getPercentConfidce() {
        return m_percentConfidce;
    }

    public boolean getUseResampling() {
        return m_UseResampling;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 2.00 $");
    }

    /**
     * 获得源代码？
     */
    public String toSource(String className) throws Exception {
        return null;
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

    /**
     * 重抽样建立分类器
     *
     */
    private void buildClassifierUsingResampling(Instances traindata) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
//

    /**
     *通过权重来建立分类器
     * @param traindata
     */
    protected void buildClassifierWithWeights(Instances traindata) throws Exception {
        //将训练数据分成有标记的和没有标记的数据
        SemiBoostSplitTrainSet splitTrain = new SemiBoostSplitTrainSet(traindata);
        Random randomInstance = new Random(m_Seed);
        Matrix simiMatrix;
        splitTrain.setRandom(randomInstance);
        //记录每次迭代的系数
        m_Betas = new double[m_Classifiers.length];
        //根据训练数据类别数目生成一个映射矩阵
        yValueMatrix = new YValueMatrix(traindata.numClasses());
        //没次训练后的错误率
        double epsilon = 0.0;
        //训练错误的样本需重新加权重；
        double reweight = 0.0;
        //对分类真确的实力加权的权重
        double reweightcorrect = 0.0;
        //对分类错误的实力加权的权重
        double reweightincorrect = 0.0;
        double alpha = 0.0;
        //类别数目为K类
        double K = (double) traindata.numClasses();
        //终止条件当错误率>(k-1)/2k 时候Batea就为负了
        double stop = (double) (K - 1) / (double) (2 * K);
        //按标记的训练数目按百分比将traindata 分成labledDataSet和UnlabledDataSet
        splitTrain.splitTrainSet(m_percentLabeled, m_randomSplit);
        Evaluation evalUnlabledSet, evalTrainPool;
        
        labledDataSet=splitTrain.getLabledData();
        int numlabledData=labledDataSet.numInstances();
        double[][] distribution;
        for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length;
                m_NumIterationsPerformed++) {
            //如果是随机的才设随机数，
            if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable) {
                ((Randomizable) (m_Classifiers[m_NumIterationsPerformed])).setSeed(randomInstance.nextInt());
            }
            //在抽取样本；
            trainPool = splitTrain.reSampleLabledData(m_NumIterationsPerformed+1, numlabledData);
            //未标记的实例，数量会减少；
            UnlabledDataSet = splitTrain.getUnLabledData();
            //建立该轮训练分类器
            m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainPool);
            //对训练池中的数据进行评价，当m_NumIterationsPerformed=0
            //trainPool中的数据都是实际的label，没有伪label
            evalTrainPool = new Evaluation(trainPool);
            evalTrainPool.evaluateModel(m_Classifiers[m_NumIterationsPerformed], trainPool);
            epsilon = evalTrainPool.errorRate();
            //错误率超过stop 就跳出循环。
            if (Utils.grOrEq(epsilon, stop)) {
                //如果是第一轮，还可以试试，呵呵
//                if (m_NumIterationsPerformed == 0) {
//                    m_NumIterationsPerformed = 1;
//                }
                //否则就终止
                break;
            }
            double temp1 = (K - 1) * (K - 1) / K;
            double logtemp2 = Math.log((1 - epsilon) / epsilon);
            double logtemp3 = Math.log((K - 1) / (K + 1));
            //计算每一次迭代的alpha值公式：(log((1-err)/err)+log((k-1)/(k+1)))
            alpha = (logtemp2 + logtemp3);
            //计算每一次迭代的Beta值公式：(K-1)^2/(K))*alpha
            m_Betas[m_NumIterationsPerformed] = temp1 * alpha;
            //终止条件当错误率>(k-1)/2k 时候Batea就为负了,或为零的时候


            //  reweight = (1 - epsilon) / epsilon;
            //分类正确的加权公式=exp(-(K-1)/K*alpha)
            reweightcorrect = Math.exp(-(K - 1) / K * alpha);
            //分类错误的加权公式=exp(alpha/K);
            reweightincorrect = Math.exp(alpha / K);
            //这个地方应该对分类正确和错误都加权
            setWeights(trainPool, reweightcorrect, reweightincorrect);

            //==以下准备对unlabeledset进行评价，挑选信任度最高的N条数据加到trainpool中
            evalUnlabledSet = new Evaluation(UnlabledDataSet);
            //UnlabledDataSet的每一条实力的类属分布
            distribution = evalUnlabledSet.getDistributions(m_Classifiers[m_NumIterationsPerformed], UnlabledDataSet);

            Distribution[] allMostConfidenceInfor = getConfidenceInfor(distribution);
            splitTrain.addToLabledDataSet(m_percentConfidce, allMostConfidenceInfor);

        }

    }

    /**
     * 返回给定实例的类属性概率
     * @param instance 给定的实例
     * @return
     * @throws java.lang.Exception
     */
    public double[] distributionForInstance(Instance instance)
            throws Exception {
        // default model?

        if (m_ZeroR != null) {
            return m_ZeroR.distributionForInstance(instance);
        }
        //没有默认的模型
        if (m_NumIterationsPerformed == 0) {
            throw new Exception("No model built");
        }
        //如果只有一轮就直接输出第一个分类器的结果
        if (m_NumIterationsPerformed == 1) {
            return m_Classifiers[0].distributionForInstance(instance);
        }
        //把每次的结果叠加起来
        int numClasses = instance.numClasses();
        double[] sums = new double[numClasses];
        //存最终的类属概率
        double[] result = new double[numClasses];
        //==========修改前代码================================
//        for (int i = 0; i < m_NumIterationsPerformed; i++) {
//
//            double[] gvalue = yValueMatrix.getRowVector((int) m_Classifiers[i].classifyInstance(instance));
//            //     double[] lastDistribution= m_Classifiers[i-1].distributionForInstance(instance);
//            for (int j = 0; j < numClasses; j++) {
//                sums[j] += m_Betas[i] * gvalue[j];
//            }
//        }
        //==========修改前代码================================
        for (int i = 0; i < m_NumIterationsPerformed; i++) {

            for (int j = 0; j < numClasses; j++) {
                sums[(int) m_Classifiers[i].classifyInstance(instance)] += m_Betas[i];
            }
        }


//为统一weka的统计信息求法：特意将sums最大的值的的类属概率设置为0.6，其他类属概率平摊
        Utils.normalize(sums);
        int maxindex = Utils.maxIndex(sums);
        for (int i = 0; i < numClasses; i++) {
            //主要突出这个最大的。。
            if (i == maxindex) {
                result[i] = 0.6;
            } else {
                result[i] = 0.4 / (double) (numClasses - 1);
            }
        }
          return result;
      //  return sums;
    }


    /**
     * 根据分布，获得所有unlabeld中的信任度信息
     * 每个实例的最大类号，以及该实例的位置索引
     * @param number
     * @param distribution
     * @return
     */
    private Distribution[] getConfidenceInfor(double[][] distribution) {



        int row = distribution.length;
        int column = distribution[0].length;
        Distribution[] ConfidenceInfor = new Distribution[row];
        double[] allMostConfidenceValue = new double[row];
        //结构体变量
        Distribution[] allMostConfidenceImfor = new Distribution[row];
        for (int i = 0; i < row; i++) {

            double maxValue = distribution[i][0];
            int maxClass = 0;
            for (int j = 0; j < column; j++) {
                if (distribution[i][j] > maxValue) {
                    maxValue = distribution[i][j];
                    maxClass = j;
                }
            }
            allMostConfidenceImfor[i] = new Distribution(i, maxClass, maxValue);
        }
        Distribution temp = new Distribution(0, 0, -5);
        for (int i = 0; i < distribution.length; i++) {
            double max = allMostConfidenceImfor[i].m_value;
            for (int j = i + 1; j < row; j++) {
                if (allMostConfidenceImfor[j].m_value > max) {
                    temp = allMostConfidenceImfor[j];
                    allMostConfidenceImfor[j] = allMostConfidenceImfor[i];
                    allMostConfidenceImfor[i] = temp;
                    max = allMostConfidenceImfor[j].m_value;
                }
            }
        }

        return allMostConfidenceImfor;
    }

    /**
     * 根据分布，获得信任度最高的number个实力的信息：包括
     * 每个实例的最大类号，以及该实例的位置索引
     * @param number
     * @param distribution
     * @return
     */
    private Distribution[] getSomeMostConfidenceInfor(int number, double[][] distribution) {


        //信任度数目超过了
        if (number > distribution.length) {
            return null;
        } else {
            int row = distribution.length;
            int column = distribution[0].length;
            Distribution[] someMostConfidenceInfor = new Distribution[number];
            double[] allMostConfidenceValue = new double[row];
            //结构体变量
            Distribution[] allMostConfidenceImfor = new Distribution[row];
            for (int i = 0; i < row; i++) {

                double maxValue = distribution[i][0];
                int maxClass = 0;
                for (int j = 0; j < column; j++) {
                    if (distribution[i][j] > maxValue) {
                        maxValue = distribution[i][j];
                        maxClass = j;
                    }
                }
                allMostConfidenceImfor[i] = new Distribution(i, maxClass, maxValue);
            }
            Distribution temp = new Distribution(0, 0, -5);
            for (int i = 0; i < number; i++) {
                double max = allMostConfidenceImfor[i].m_value;
                for (int j = i + 1; j < row; j++) {
                    if (allMostConfidenceImfor[j].m_value > max) {
                        temp = allMostConfidenceImfor[j];
                        allMostConfidenceImfor[j] = allMostConfidenceImfor[i];
                        allMostConfidenceImfor[i] = temp;
                        max = allMostConfidenceImfor[j].m_value;
                    }
                }
            }
            for (int i = 0; i < number; i++) {
                someMostConfidenceInfor[i] = allMostConfidenceImfor[i];
            }
            return someMostConfidenceInfor;
        }
    }

    /**
     * 对训练池中分类错误和真确的分别权重
     * @param trainPool
     * @param reweightcorrect
     * @param reweightincorrect
     */
    private void setWeights(Instances trainPool, double reweightcorrect, double reweightincorrect)
            throws Exception {
        double newSumOfWeight, oldSumOfWeights;
        //原来训练池中数据的权重总和
        oldSumOfWeights = trainPool.sumOfWeights();
        Enumeration enu = trainPool.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance instance = (Instance) enu.nextElement();
            boolean correct = Utils.eq(m_Classifiers[m_NumIterationsPerformed].classifyInstance(instance),
                    instance.classValue());
            if (!correct) {
                instance.setWeight(reweightincorrect);//设置分错了的权值
            }
            if (!m_onlyWeightIncorrect && correct)//不只对分类错误的进行权重，还需对正确的类权重
            {
                instance.setWeight(reweightcorrect);//设置分类正确的权值
            }
        }
        //09-01-03下午注释
//        newSumOfWeight = trainPool.sumOfWeights();
//        enu = trainPool.enumerateInstances();
//        while (enu.hasMoreElements()) {
//            Instance instance = (Instance) enu.nextElement();
//            double oldweight = instance.weight();
//            //将权重归一化
//            instance.setWeight(oldweight * oldSumOfWeights / newSumOfWeight);
//        }
    }

    /**
     *
     * 设置权重
     */
    private void setWeights(Instances trainPool, double reweight) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void setOptions(String[] options) throws Exception {

        setDebug(Utils.getFlag('D', options));
    }

    public void setUseResampling(boolean useResampling) {
        m_UseResampling = useResampling;
    }

    public void setNumLabledData(int numLabledData) {
        m_numLabledData = numLabledData;
    }

    public int getNumLabledData() {
        return m_numLabledData;
    }

    /**
     * 设置标记的百分比
     * @param m_percentLabeled
     */
    public void setPercentLabeled(double m_percentLabeled) {
        this.m_percentLabeled = m_percentLabeled;
    }

    public double getPercentLabeled() {
        return m_percentLabeled;
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

    private void computeSimilarityMatrix(Instances traindata) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public static void main(String[] argv) {
        runClassifier(new MultiSemiAdaBoost(), argv);
    }
}
