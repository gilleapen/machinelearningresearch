/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import java.util.Enumeration;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.Sourcable;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.classifiers.lazy.IB1;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;

/**
 *
 * @author Administrator
 */
public class ASSEMBLE extends RandomizableIteratedSingleClassifierEnhancer
        implements WeightedInstancesHandler, Sourcable {

    private ZeroR m_ZeroR;
    private int m_NumIterationsPerformed;
    private int m_WeightThreshold;
    private int MAX_NUM_RESAMPLING_ITERATIONS;
    /**标记的百分比*/
    private double m_percentLabeled = 10.0;
    /** 随机分训练集*/
    protected boolean m_randomSplit = false;
    private Instances UnlabledDataSet;
    private double[] m_weight;

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
            buildClassifierWithWeights(traindata, 1.0, 0.9);
        } else {
            buildClassifierUsingResampling(traindata, 1.0, 0.9);
        }

    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public String toSource(String className) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     *通过权重来建立分类器
     * @param traindata
     */
    protected void buildClassifierWithWeights(Instances traindata, double alpha, double belta) throws Exception {
        //将训练数据分成有标记的和没有标记的数据
        SemiBoostSplitTrainSet splitTrain = new SemiBoostSplitTrainSet(traindata);
        Random randomInstance = new Random(m_Seed);
        splitTrain.setRandom(randomInstance);
        //创建一个KNN近邻算法
        Classifier nn1 = new IB1();
        //记录每次迭代的系数
        m_weight = new double[m_Classifiers.length];

        //每次训练后的错误率
        double epsilon = 0.0;
        double[][] distribution;
        Evaluation evalUnlabledSet, evalTrainPool;
        Instances trainPool;
        Instances labeledDataSet;
        Instances tempInstanctces;


        //按标记的训练数目按百分比将traindata 分成labledDataSet和UnlabledDataSet
        splitTrain.splitTrainSet(m_percentLabeled, m_randomSplit);
        //每次训练的训练数据,数量会递增，不断从未标记的实例中提取实例加入到其中
        trainPool = splitTrain.getLabledData();
        //为标记的数据设定初始权值b/l
        //  setWeights(trainPool, belta / (double) trainPool.numInstances());
        //标记的数据
        labeledDataSet = new Instances(splitTrain.getLabledData(), 0,
                splitTrain.getLabledData().numInstances());

        //未标记的实例，
        UnlabledDataSet = splitTrain.getUnLabledData();
        //为未标记的数据设定初始权值（1-b）/l
        //    setWeights(UnlabledDataSet, (1.0 - belta) / (double) UnlabledDataSet.numInstances());
        //利用KNN方法预先进行类别标记
        Enumeration e = UnlabledDataSet.enumerateInstances();
        //利用标记的数据建立KNN分类器。
        nn1.buildClassifier(trainPool);
        while (e.hasMoreElements()) {
            Instance instance = (Instance) e.nextElement();
            instance.setClassValue(nn1.classifyInstance(instance));
            //添加到训练已经标记的样本中中去
            trainPool.add(instance);
        }
        trainPool.compactify();
        UnlabledDataSet.compactify();
        labeledDataSet.compactify();
        //trainPool中的数据都是实际的label，有伪label 用KNN方法
        //建立该轮训练分类器
        m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainPool);

        for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length-1;
                ) {
            //如果是随机的才设随机数，
            if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable) {
                ((Randomizable) (m_Classifiers[m_NumIterationsPerformed])).setSeed(randomInstance.nextInt());
            }

            evalTrainPool = new Evaluation(trainPool);
            evalTrainPool.evaluateModel(m_Classifiers[m_NumIterationsPerformed], trainPool);
            epsilon = evalTrainPool.errorRate();
            //错误率超过stop 或者UnlabeldDataSet中没有足够的数据，就跳出循环。
            if (Utils.grOrEq(epsilon, 0.5)/* || Utils.eq(epsilon, 0)*/) {
                //如果是第一轮，还可以试试，呵呵
                if (m_NumIterationsPerformed == 0) {
                    m_NumIterationsPerformed = 1;
                }
                //否则就终止
                break;
            }
            //基分类器的权值
            m_weight[m_NumIterationsPerformed] = 0.5 * Math.log((1 - epsilon) / epsilon);
            //得出yi
            Enumeration en = UnlabledDataSet.enumerateInstances();
            //保存D值;
            double[] D_value = new double[UnlabledDataSet.numInstances()];
            int index = 0;
            double sumD_value = 0.0;


            while (en.hasMoreElements()) {
                Instance instance = (Instance) en.nextElement();
                //利用前面几次迭代已经训练好的分类器进行类标记
                double[] distribut = distributionForInstance(instance, m_NumIterationsPerformed);
                //重新设定类标记
                int y = Utils.maxIndex(distribut);
                instance.setClassValue(y);
                //计算新的权值
                D_value[index] = alpha * Math.exp(-(double) y * distribut[y]);
                sumD_value += D_value[index];
                index++;
            }

            for (int i = 0; i < index; i++) {
                D_value[i] = D_value[i] / sumD_value;
            }
            //  重新设定权值
            setWeights(UnlabledDataSet, D_value);

            //为合并标记数据和未标记数据临时拿来用;
            Instances copylabeledDataSet = new Instances(labeledDataSet, 0,
                    labeledDataSet.numInstances());
            Enumeration enu = UnlabledDataSet.enumerateInstances();
            while (enu.hasMoreElements()) {
                Instance instance = (Instance) enu.nextElement();
                copylabeledDataSet.add(instance);
            }
            copylabeledDataSet.compactify();
            //抽取下轮训练数据
            Instances nextTrainPool = resample(copylabeledDataSet, m_NumIterationsPerformed, labeledDataSet.numInstances());
            //建立下一轮的分类器
            m_NumIterationsPerformed=m_NumIterationsPerformed+1;
            m_Classifiers[m_NumIterationsPerformed].buildClassifier(nextTrainPool);
         // copylabeledDataSet = null;

        }


    }

    /**
     * 随机抽样
     * @param data 抽样的数据集
     * @param seed 种子
     * @param numLabled 数目
     * @return 抽取的样本集
     */
    public Instances resample(Instances data, int seed, int numLabled) {
        Instances newData;
        int i;
        int index;
        Instance instance;
        Random random;
        random = new Random(seed);
        newData = new Instances(data, numLabled);

        if (data.numInstances() > 0) {
            for (i = 0; i < numLabled; i++) {
                index = random.nextInt(numLabled);
                instance = data.instance(index);
                newData.add(instance);
            }
        }

        return newData;
    }

    private void buildClassifierUsingResampling(Instances traindata, double d, double d0) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    /**
     * 根据权值矩阵设置权值
     * @param training
     * @param reweight
     */
    private void setWeights(Instances data, double[] reweight) {
        Enumeration e = data.enumerateInstances();
        int index = 0;
        while (e.hasMoreElements()) {
            Instance instance = (Instance) e.nextElement();
            instance.setWeight(reweight[index]);
            index++;
        }
    }

    /**最终分类器
     * 返回给定实例的类属性概率
     * @param instance 给定的实例
     * @return
     * @throws java.lang.Exception
     */
    public double[] distributionForInstance(Instance instance)
            throws Exception {
        return distributionForInstance(instance, m_NumIterationsPerformed);
    }

    /**中间多个分类器的和累加 注意这个跟原来有些不样
     * 返回给定实例的类属性概率
     * @param instance 给定的实例
     * @return
     * @throws java.lang.Exception
     */
    public double[] distributionForInstance(Instance instance, int IterationsPerformed)
            throws Exception {
        // default model?
        double[]dfirstDistribution=m_Classifiers[0].distributionForInstance(instance);
        if (m_ZeroR != null) {
            return m_ZeroR.distributionForInstance(instance);
        }
        // 注意这个跟原来有些不样
         if (IterationsPerformed == 0)
         {
             return dfirstDistribution;
         }
       
        //把每次的结果叠加起来
        int numClasses = instance.numClasses();

        //存最终的类属概率
        double[] result = dfirstDistribution;

        for (int i = 0; i < IterationsPerformed; i++) {

            double[] temp = m_Classifiers[i].distributionForInstance(instance);
            for (int j = 0; j < numClasses; j++) {
                result[j] += temp[j] * m_weight[i];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < numClasses; j++) {
            sum += result[j];
        }
        Utils.normalize(result, sum);
        return result;

    }

    private boolean getUseResampling() {
        return false;
    }

    private void setWeights(Instances training, double reweight) {
        Enumeration e = training.enumerateInstances();

        while (e.hasMoreElements()) {
            Instance instance = (Instance) e.nextElement();
            instance.setWeight(reweight);
        }
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
     * 设置标记的百分比
     * @param m_percentLabeled
     */
    public void setPercentLabeled(double m_percentLabeled) {
        this.m_percentLabeled = m_percentLabeled;
    }

    public double getPercentLabeled() {
        return m_percentLabeled;
    }
}
