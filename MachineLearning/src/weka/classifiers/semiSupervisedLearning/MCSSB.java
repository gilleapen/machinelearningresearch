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
import weka.core.EuclideanDistance;
import weka.core.ExpDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 *
 * @author Administrator
 */
public class MCSSB extends RandomizableIteratedSingleClassifierEnhancer
        implements WeightedInstancesHandler, Sourcable {

    private ZeroR m_ZeroR;
//    private double[] m_alpha;
    /**标记的百分比*/
    private double m_percentLabeled = 10.0;
    /** 随机分训练集*/
    protected boolean m_randomSplit = false;
    private Instances labledDataSet;
    private Instances UnlabledDataSet;
    /** The number of successfully generated base classifiers. */
    protected int m_NumIterationsPerformed;
    /*从抽样的百分比*/
    private double m_percentSample = 20.0;
    /*系数*/
    double C = 10000;
    //存储伪标记
    double[][] ylabele;
    /*基分类器其h(x)的系数*/
    private double[] m_WeightAlpha;
    protected boolean m_UseResampling = false;

    public void setPercentSample(int percentSample) {
        m_percentSample = percentSample;
    }

    public double getPercentSample() {
        return m_percentSample;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Hanmed Valizadegan, RongJin and Anil K.Jain");
        result.setValue(Field.TITLE, "Semi-Supervised Boosting for Multi-Class Classification");
        result.setValue(Field.BOOKTITLE, "The 19th European Conference on Machine Learning (ECML 2008 )");
        result.setValue(Field.YEAR, "2008");
        result.setValue(Field.PAGES, "NO");
        result.setValue(Field.PUBLISHER, "ECML");
        result.setValue(Field.ADDRESS, "Antwerp, Belgium");

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

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public String toSource(String className) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    private void buildClassifierUsingResampling(Instances traindata) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    /**
     * 计算b的值
     * @param y
     * @return
     */
    private double[][] computbValue(double y[][]) {
        int row = y.length;
        int m = y[0].length;
        double[][] b = new double[row][m];
        for (int i = 0; i < row; i++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += Math.exp(y[i][k]);
            }
            for (int k = 0; k < m; k++) {
                b[i][k] = Math.exp(y[i][k]) / sum;
            }
        }
        return b;

    }

    /**
     * 计算未标记中Z矩阵的值
     * @param b
     * @return
     */
    private double[][] computZvalue(double b[][]) {
        int row = b.length;
        int m = b[0].length;
        double[][] Z = new double[row][row];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < row; j++) {
                for (int k = 0; k < m; k++) {
                    Z[i][j] += b[i][k] * b[j][k];
                }
            }
        }
        return Z;
    }

    /**
     * 计算T矩阵，三维矩阵
     * @param b
     * @param Z
     * @return T
     */
    private double[][][] computTvalue(double b[][], double Z[][]) {
        int row = b.length;
        int m = b[0].length;
        double[][][] T = new double[row][row][m];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < row; j++) {
                for (int k = 0; k < m; k++) {
                    T[i][j][k] = b[i][k] * b[j][k] / Z[i][j];
                }
            }
        }
        return T;
    }

    /**
     * 计算相似矩阵
     * @param data
     * @return S
     */
    private double[][] computSvalue(Instances firstInstances, Instances secondInstances) {
        int first = firstInstances.numInstances();
        int second = secondInstances.numInstances();
        int m = secondInstances.numClasses();
        double[][] S = new double[first][second];
        EuclideanDistance distance = new EuclideanDistance(firstInstances);
        for (int i = 0; i < first; i++) {
            for (int j = 0; j < second; j++) {
                S[i][j] = distance.distance(firstInstances.instance(i), secondInstances.instance(j));
            }
        }
        return S;

    }

    /**
     * 计算phi
     * @param b
     * @param yUnlabele
     * @param ylabele
     * @return
     */
    private double[][][] computPhivalue(double b[][], double yUnlabele[][],
            double ylabele[][]) {
        int numL = ylabele.length;
        int numUnL = yUnlabele.length;

        int m = b[0].length;
        double[][][] Phi = new double[numUnL][numL][m];
        for (int i = 0; i < numUnL; i++) {
            for (int j = 0; j < numL; j++) {
                for (int k = 0; k < m; k++) {
                    double sumtemp = 0;
                    for (int l = 0; l < m; l++) {
                        sumtemp += 1.0 / b[i][l];
                    }
                    Phi[i][j][k] = ylabele[j][k] * b[i][k] * sumtemp - yUnlabele[i][k] / b[i][k];
                }
            }
        }
        return Phi;

    }

    /**
     * 计算未标记数据的alpha值
     * @param unlabeledData
     * @param y
     * @return
     */
    private double[][] computAlpha(Instances unlabeledData, double y[][]) {
        int row = y.length;
        int m = y[0].length;
        double[][] alpha = new double[row][m];
        double[][] S = computSvalue(unlabeledData, unlabeledData);
        double[][] b = computbValue(y);
        double[][] Z = computZvalue(b);
        double[][][] T = computTvalue(b, Z);
        int num = unlabeledData.numInstances();
        for (int i = 0; i < num; i++) {

            for (int k = 0; k < m; k++) {
                double tempsum = 0.0;
                for (int j = 0; j < num; j++) {
                    tempsum += S[i][j] * (b[i][k] - T[i][j][k]) / Z[i][j];
                }
                alpha[i][k] = tempsum;
            }

        }

        return alpha;

    }

    private double[][] computBelta(Instances unlabeledData, Instances labeledData,
            double ylabele[][], double yUnlabele[][]) {
        int numUnL = unlabeledData.numInstances();
        int numL = labeledData.numInstances();
        int m = labeledData.numClasses();
        double belta[][] = new double[numUnL][m];
        double[][] b = computbValue(yUnlabele);
        double[][][] Phi = computPhivalue(b, yUnlabele, ylabele);
        double[][] S = computSvalue(unlabeledData, labeledData);
        double tempsum = 0.0;
        for (int i = 0; i < numUnL; i++) {
            for (int k = 0; k < m; k++) {
                tempsum = 0.0;
                for (int j = 0; j < numL; j++) {
                    tempsum += 0.5 * S[i][j] * Phi[i][j][k];
                }
                belta[i][k] = tempsum;
            }
        }
        return belta;
    }

    /**
     * 设置系数
     * @param C
     */
    public void setC(double C) {
        this.C = C;
    }

    public double getC() {
        return C;
    }

    /**最终分类器
     * 返回给定实例的类属性概率
     * @param instance 给定的实例
     * @return
     * @throws java.lang.Exception
     */
    public double[] distributionForInstance(Instance instance)
            throws Exception {
        // default model?
        double[] dfirstDistribution = m_Classifiers[0].distributionForInstance(instance);
        if (m_ZeroR != null) {
            return m_ZeroR.distributionForInstance(instance);
        }
        // 注意这个跟原来有些不样
        if (m_NumIterationsPerformed == 0) {
            return dfirstDistribution;
        }

        //把每次的结果叠加起来
        int numClasses = instance.numClasses();

        //存最终的类属概率
        double[] result = dfirstDistribution;

        for (int i = 1; i < m_NumIterationsPerformed; i++) {

            double[] temp = m_Classifiers[i].distributionForInstance(instance);
            for (int j = 0; j < numClasses; j++) {
                result[j] += temp[j] * m_WeightAlpha[i];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < numClasses; j++) {
            sum += result[j];
        }
        Utils.normalize(result, sum);
        return result;
    }

    /**
     * 调试程序用
     * @param alpha
     * @param belta
     * @return
     */
    private double[][] AlaphaBelta(double[][] alpha, double[][] belta) {
        int num = alpha.length;
        int m = alpha[0].length;
        double[][] result = new double[num][m];
        for (int i = 0; i < num; i++) {
            for (int k = 0; k < m; k++) {
                result[i][k] = alpha[i][k] + C * belta[i][k];
            }

        }
        return result;
    }

    private void buildClassifierWithWeights(Instances traindata) throws Exception {
        //将训练数据分成有标记的和没有标记的数据
        SemiBoostSplitTrainSet splitTrain = new SemiBoostSplitTrainSet(traindata);
        Random randomInstance = new Random(m_Seed);
        splitTrain.setRandom(randomInstance);
        //记录每次迭代的系数
        m_WeightAlpha = new double[m_Classifiers.length];
//        //记录每次迭代的系数
//        m_alpha = new double[m_Classifiers.length];
        //按标记的训练数目按百分比将traindata 分成labledDataSet和UnlabledDataSet
        splitTrain.splitTrainSet(m_percentLabeled, m_randomSplit);
        labledDataSet = splitTrain.getLabledData();
        UnlabledDataSet = splitTrain.getUnLabledData();
        int numlabledData = labledDataSet.numInstances();
        int numUnlabledData = UnlabledDataSet.numInstances();
        //从未标记的数据集中按比例抽取S个样本
        int s = Math.max(20, (int) traindata.numInstances() / 5);
        //使之符合实际大小
        UnlabledDataSet.compactify();
        labledDataSet.compactify();
        //训练一个初始分类器h0(x)
        m_Classifiers[0].buildClassifier(labledDataSet);
        Enumeration enun = UnlabledDataSet.enumerateInstances();
        //未标记数据的分布，并设置伪y
        double[][] yUnlabele = new double[UnlabledDataSet.numInstances()][UnlabledDataSet.numClasses()];
        int index = 0;
        while (enun.hasMoreElements()) {
            Instance instance = (Instance) enun.nextElement();
            double classvalue = m_Classifiers[0].classifyInstance(instance);
            instance.setClassValue(classvalue);
            for (int j = 0; j < instance.numClasses(); j++) {
                if ((int) classvalue == j) {
                    yUnlabele[index][j] = 1;
                } else {
                    yUnlabele[index][j] = 0;
                }
            }

            index++;
        }
        //将有标记的y设置为成员变量
        setylable(labledDataSet);

        for (m_NumIterationsPerformed = 1; m_NumIterationsPerformed < m_Classifiers.length;) {
            //如果是随机的才设随机数，
            if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable) {
                ((Randomizable) (m_Classifiers[m_NumIterationsPerformed])).setSeed(randomInstance.nextInt());
            }
            double[][] alpha = computAlpha(UnlabledDataSet, yUnlabele);
            double[][] belta = computBelta(UnlabledDataSet, labledDataSet, ylabele, yUnlabele);
            double [][]test= AlaphaBelta(alpha, belta);
            setUnlabeledWeight(alpha, belta);
            //从未标记的数据集中按比例抽取S个样本

            Instances sampleInstance = resample(UnlabledDataSet, m_NumIterationsPerformed, s);
            //back up the lableledInstances sets.
            Instances copylabledInstances = new Instances(labledDataSet, 0, labledDataSet.numInstances());
            //合并标记的数据和S个带有伪标记的数据集
            Enumeration en = sampleInstance.enumerateInstances();
            while (en.hasMoreElements()) {
                Instance instance = (Instance) en.nextElement();
                copylabledInstances.add(instance);
            }
            //建立分类器h(x)
            m_Classifiers[m_NumIterationsPerformed].buildClassifier(copylabledInstances);
            //计算h(k)
            double h[][] = coumputhValue(m_Classifiers[m_NumIterationsPerformed], UnlabledDataSet);
            double weightAlpha = computWeightAlpha(labledDataSet, UnlabledDataSet, h);
            //如果权重为负数，就直接终止
            if (weightAlpha < 0) {
                break;
            }
            m_WeightAlpha[m_NumIterationsPerformed] = weightAlpha;
        }
    }

    private void setylable(Instances labledDataSet) {

        Enumeration enun = labledDataSet.enumerateInstances();
        ylabele = new double[labledDataSet.numInstances()][labledDataSet.numClasses()];
        int index2 = 0;
        while (enun.hasMoreElements()) {
            Instance instance = (Instance) enun.nextElement();
            int classvalue = (int) instance.classValue();
            for (int j = 0; j < instance.numClasses(); j++) {
                if (classvalue == j) {
                    ylabele[index2][j] = 1;
                } else {
                    ylabele[index2][j] = 0;
                }
            }
            index2++;
        }

    }

    /**
     * 计算基分类器的权重
     * @param labledDataSet
     * @param UnlabledDataSet
     * @param h
     * @return
     */
    private double computWeightAlpha(Instances labledDataSet, Instances UnlabledDataSet,
            double h[][]) throws Exception {
        double weightAlpha = 0.0;
        int numL = labledDataSet.numInstances();
        int numUnL = UnlabledDataSet.numInstances();
        int m = UnlabledDataSet.numClasses();
        //============计算Au====
        double Au = 0.0;
        //未标记数据的相似矩阵
        double Sunlable[][] = computSvalue(UnlabledDataSet, UnlabledDataSet);
        double b[][] = computbValue(h);
        double Zu[][] = computZvalue(b);

        for (int i = 0; i < numUnL; i++) {
            for (int j = 0; j < numUnL; j++) {
                double temp = 0.0;
                for (int k = 0; k < m; k++) {
                    temp += h[i][k] * b[i][k];
                }
                Au += Sunlable[i][j] / Zu[i][j] * temp;
            }
        }
        //============计算Al==========
        double[][] S_lableandUnlabe = computSvalue(labledDataSet, UnlabledDataSet);
        double Al = 0.0;
        double sum = 0.0;
        double temp = 0.0;
        double temp2 = 0.0;
        for (int i = 0; i < numL; i++) {
            for (int j = 0; j < numUnL; j++) {
                sum = 0.0;
                temp = 0.0;
                for (int k = 0; k < m; k++) {
                    temp = ylabele[i][k] / b[j][k];
                    temp2 = 0.0;
                    for (int l = 0; l < m; l++) {
                        temp2 += (b[j][l]) * h[j][l];
                    }
                    sum += temp * temp2;
                }
                Al += 0.5 * S_lableandUnlabe[i][j] * sum;
            }
        }
        //============计算Bu==========
        double Bu = 0.0;
        double sumtemp = 0.0;
        double[][][] T = computTvalue(b, Zu);
        for (int i = 0; i < numUnL; i++) {
            for (int j = 0; j < numUnL; j++) {
                sumtemp = 0.0;
                for (int k = 0; k < m; k++) {
                    sumtemp += h[i][k] * T[i][j][k];
                }
                Bu += Sunlable[i][j] / Zu[i][j] * sumtemp;
            }
        }
        //============计算Bl==========
        double Bl = 0.0;
        double sumtemp1 = 0.0;
        for (int i = 0; i < numL; i++) {
            for (int j = 0; j < numUnL; j++) {
                sumtemp1 = 0.0;
                for (int k = 0; k < m; k++) {
                    sumtemp1 += ylabele[i][k] * h[j][k] / b[j][k];
                }
                Bl += 0.5 * S_lableandUnlabe[i][j] * sumtemp1;
            }
        }
//******计算weightAlpha*****************************************
        weightAlpha = 0.25 * Math.log((Bu + C * Bl) / (Au + C * Al));

        return weightAlpha;

    }

    /**
     * 计算h(k)
     * @param classifier 当前的分类器器
     * @param data unlabled数据
     * @return
     * @throws java.lang.Exception
     */
    private double[][] coumputhValue(Classifier classifier, Instances data) throws Exception {
        int num = data.numInstances();
        int m = data.numClasses();
        double[][] h = new double[num][m];
        Enumeration en = data.enumerateInstances();
        int index = 0;
        while (en.hasMoreElements()) {
            Instance instance = (Instance) en.nextElement();
            double temp[] = classifier.distributionForInstance(instance);
            int maxindex = Utils.maxIndex(temp);
            for (int i = 0; i < m; i++) {
                if (i == maxindex) {
                    h[index][i] = 1;
                } else {
                    h[index][i] = 0;
                }
            }
            index++;
        }
        return h;
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

    /**
     * 随机抽样
     * @param data 抽样的数据集
     * @param seed 种子
     * @param 比例数目
     * @return 抽取的样本集
     */
    public Instances resample(Instances data, int seed, double percentSample) {
        Instances newData;
        int i;
        int index;
        Instance instance;
        Random random;
        random = new Random(seed);
        int numLabled = (int) ((double) data.numInstances() * (percentSample / 100.0));
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

    /**
     * 设置未标记数据的权重和伪类标记
     * @param alpha
     * @param belta
     */
    private void setUnlabeledWeight(double[][] alpha, double[][] belta) {
        Enumeration enun = UnlabledDataSet.enumerateInstances();
        double[][] temp = new double[alpha.length][alpha[0].length];
        int index = 0;
        double oldSumOfWeights, newSumOfWeights;

        oldSumOfWeights = UnlabledDataSet.sumOfWeights();

        while (enun.hasMoreElements()) {
            Instance instance = (Instance) enun.nextElement();
            for (int j = 0; j < alpha[0].length; j++) {
                temp[index][j] = alpha[index][j] + C * belta[index][j];
            }
            //将最大的类的索引号设为类标记
            int maxClass = Utils.maxIndex(temp[index]);
            instance.setClassValue(maxClass);
            //重新设置权值
            instance.setWeight(temp[index][maxClass] * instance.weight());
            index++;

        }
        // Renormalize weights
        newSumOfWeights = UnlabledDataSet.sumOfWeights();
        enun = UnlabledDataSet.enumerateInstances();
        while (enun.hasMoreElements()) {
            Instance instance = (Instance) enun.nextElement();
            instance.setWeight(instance.weight() * oldSumOfWeights / newSumOfWeights);
        }

    }

//    public double getPercentLabeled() {
//        return m_percentLabeled;
//    }
//
//    public void setPercentLabeled(double m_percentLabeled) {
//        this.m_percentLabeled = m_percentLabeled;
//    }
    private boolean getUseResampling() {
        return m_UseResampling;
    }
}
