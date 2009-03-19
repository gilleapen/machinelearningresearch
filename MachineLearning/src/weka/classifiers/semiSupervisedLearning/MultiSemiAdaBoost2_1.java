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
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;

/**
 *2009-01-03
 * @author huangdongshan
 */
public class MultiSemiAdaBoost2_1 extends RandomizableIteratedSingleClassifierEnhancer
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
    public boolean m_onlyWeightIncorrect = true;
    /**标记的百分比*/
    private double m_percentLabeled = 10.0;
    private int m_kNN = 1;
    private double m_labebledWeights = 0.8;

//    public void setKNN(int m_kNN) {
//        this.m_kNN = m_kNN;
//    }
//
//    public int getKNN() {
//        return m_kNN;
//    }
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
        if (!getUseResampling() /*&& (m_Classifier instanceof WeightedInstancesHandler)*/) {
            buildClassifierWithWeights(traindata);
        } else {
            buildClassifierUsingResampling(traindata);
        }

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

    /**
     * 设置标记的百分比
     * @param m_percentLabeled
     */
    public void setLabebledWeights(double m_labebledWeights) {
        this.m_labebledWeights = m_labebledWeights;
    }

    public double getLabebledWeights() {
        return m_labebledWeights;
    }

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
        double stop = (double) (K - 1) / (double) (K);
        //按标记的训练数目按百分比将traindata 分成labledDataSet和UnlabledDataSet
        splitTrain.splitTrainSet(m_percentLabeled, m_randomSplit);
        Evaluation evalUnlabledSet, evalTrainPool;
        //利用K紧邻给为标记的训练数据进行伪标记；
//        IBk knn = new IBk();
//        knn.setKNN(m_kNN);
        IB1 knn = new IB1();
        //标记的数据
        labledDataSet = splitTrain.getLabledData();
        //为标记的数据设定初始权值b/l
        // setWeights(trainPool, belta / (double) trainPool.numInstances());

        //每次训练的训练数据,数量会递增，不断从未标记的实例中提取实例加入到其中
        trainPool = new Instances(labledDataSet, 0,
                labledDataSet.numInstances());

        //未标记的实例
        UnlabledDataSet = splitTrain.getUnLabledData();

        //利用KNN方法预先进行类别标记
        Enumeration e = UnlabledDataSet.enumerateInstances();
        //利用标记的数据建立KNN分类器。
        knn.buildClassifier(trainPool);
        while (e.hasMoreElements()) {
            Instance instance = (Instance) e.nextElement();
            instance.setClassValue(knn.classifyInstance(instance));
            //添加到训练已经标记的样本中中去
            trainPool.add(instance);
        }
        trainPool.compactify();
        UnlabledDataSet.compactify();
        labledDataSet.compactify();
        // 把labled的权重加重，把unlabled的权重减少
        Enumeration labled = labledDataSet.enumerateInstances();
        while (labled.hasMoreElements()) {
            Instance l = (Instance) labled.nextElement();
            l.setWeight(m_labebledWeights);
        }

        Enumeration unlabled = UnlabledDataSet.enumerateInstances();
        while (unlabled.hasMoreElements()) {
            Instance u = (Instance) unlabled.nextElement();
            u.setWeight(1.0 - m_labebledWeights);
        }


        for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length;
                m_NumIterationsPerformed++) {
            //如果是随机的才设随机数，
            if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable) {
                ((Randomizable) (m_Classifiers[m_NumIterationsPerformed])).setSeed(randomInstance.nextInt());
            }

            //建立该轮训练分类器
            m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainPool);
            //对训练池中的数据进行评价，当m_NumIterationsPerformed=0
            //trainPool中的数据都是实际的label，没有伪label
            evalTrainPool = new Evaluation(trainPool);
            evalTrainPool.evaluateModel(m_Classifiers[m_NumIterationsPerformed], trainPool);
            epsilon = evalTrainPool.errorRate();
            //错误率超过stop 或者UnlabeldDataSet中没有足够的数据，就跳出循环。
            if (Utils.grOrEq(epsilon, stop) /*|| Utils.eq(epsilon, 0)*/) {
                //如果是第一轮，还可以试试，呵呵
                if (m_NumIterationsPerformed == 0) {
                    m_NumIterationsPerformed = 1;
                }
                //否则就终止
                break;
            }

            double logtemp2 = Math.log((1 - epsilon) / epsilon);
            double logtemp3 = Math.log((K - 1));
            //计算每一次迭代的alpha值公式：(log((1-err)/err)+log((k-1)/(k+1)))
            alpha = (logtemp2 + logtemp3);
            //计算每一次迭代的Beta值公式：(K-1)^2/(K))*alpha
            double temp1 = (double) (K - 1) * (double) (K - 1) / (double) K;
            m_Betas[m_NumIterationsPerformed] = temp1 * alpha;
            //  reweight = (1 - epsilon) / epsilon;
            //分类正确的加权公式=exp(alpha)
            reweightcorrect = Math.exp(alpha);
            //分类错误的加权公式=exp(alpha/K);
            reweightincorrect = Math.exp(alpha);
            //这个地方应该对分类正确和错误都加权
            setWeights(trainPool, reweightcorrect, reweightincorrect);

        }

    }

    /**
     * 返回给定实例的类属性概率
     * @param instance 给定的实例
     * @return
     * @throws java.lang.Exception
     */
    @Override
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
        newSumOfWeight = trainPool.sumOfWeights();
        enu = trainPool.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance instance = (Instance) enu.nextElement();
            double oldweight = instance.weight();
            //将权重归一化
            instance.setWeight(oldweight * oldSumOfWeights / newSumOfWeight);
        }
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

    public static void main(String[] argv) {
        runClassifier(new MultiSemiAdaBoost(), argv);
    }
}
