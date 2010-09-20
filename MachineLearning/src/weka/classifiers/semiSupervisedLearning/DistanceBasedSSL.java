/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import weka.classifiers.semiSupervisedLearning.DijkstraGraph.Dijkstra;
//import weka.classifiers.semiSupervisedLearning.;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.collective.CollectiveRandomizableClassifier;
import weka.classifiers.collective.util.Splitter;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.semiSupervisedLearning.DijkstraGraph.KnnGraph;
import weka.classifiers.semiSupervisedLearning.DijkstraGraph.KnnInfor;
import weka.classifiers.semiSupervisedLearning.dijkstra.Side;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *  基于直接用距离计算 半监督学习
 * @author huangdongshan
 */
public class DistanceBasedSSL extends CollectiveRandomizableClassifier implements
        TechnicalInformationHandler {

    /** Whether to normalize/standardize/neither */
    protected int m_filterType;
    protected DistanceBasedSSL m_Parent;
    private CostDistanceBasedSSLInstances m_Data;
    /** copy of the original training dataset */
    protected Instances m_TrainsetNew;
    /** copy of the original test dataset */
    protected Instances m_TestsetNew;
    int k = 5;
    //define some constants
    public static final double INF = Double.MAX_VALUE; //infinity
    private double weightGraph[][];
    private double PairsweightGraph[][];
    /*保存结果*/
    Matrix m_resultMatrix;
    private double m_sigma = 0.01;
private DistanceFunction  m_DistanceFunction=new EuclideanDistance();
    /**
     * performs initialization of members
     */
    @Override
    protected void initializeMembers() {
        super.initializeMembers();

        m_TrainsetNew = null;
        m_TestsetNew = null;

        m_filterType = SMO.FILTER_NORMALIZE;

        m_Data = null;
        m_resultMatrix = null;
        weightGraph = null;
    }

    /**
     * 得到一个实例的最终类分布
     * @param instance
     * @return
     * @throws java.lang.Exception
     */
    @Override
    protected double[] getDistribution(Instance instance) throws Exception {
        int index;
        double[] result;
        int numClass;
        index = m_Data.indexOf(instance);
        result = null;
        numClass = m_Data.getTrainSet().numClasses();
        if (index > -1) {
            result = new double[numClass];
            double classvalue = m_resultMatrix.get(index, 0);
            if (classvalue == -1.0) {
                System.out.println("不能按我们的方法找到合适的最终类标记");
            } else {
                int classindex = (int) classvalue;
                result[classindex] = 1.0;
            }

        } else {
            throw new Exception("Cannot find instance: " + instance + "\n" + " -> pos=" + index + " = " + m_Data.get(StrictMath.abs(index)));
        }
        return result;
    }

    public void setK(int k) {
        this.k = (k + 1);
    }

    /**
     *  k个邻居，为建立邻接图
     * @return
     */
    public int getK() {
        return k;
    }

    /**
     * 设置sigma 扩大代价函数的差异
     * @param sigma
     */
    public void setSigma(double sigma) {
        m_sigma = sigma;
    }

    public double getSigma() {
        return m_sigma;
    }

    /**
     * 初始化权值矩阵
     */
    void initWeightGraph(int NUM_VERTICES) {
        weightGraph = new double[NUM_VERTICES][NUM_VERTICES];
        for (int i = 0; i < NUM_VERTICES; i++) {
            for (int j = 0; j < NUM_VERTICES; j++) {
                weightGraph[i][j] = INF;
            }
        }
    }

    /**
     * 构建邻域图，实际上已经是有向图了
     * @param trainNew 数据集
     * @param k 近邻数目
     * @throws java.lang.Exception
     */
    void createKNNweightGraph(Instances trainNew, int[] nodes, int k) throws Exception {

        int numInst = trainNew.numInstances();
        //初始化权值矩阵
        initWeightGraph(numInst);
        // int numCls = trainNew.numClasses();
        //利用KNN寻找训练样本点的K个近邻
//        IBk knn = new IBk();
//        knn.setKNN(k);
//        knn.buildClassifier(trainNew);
        KnnGraph kgraph = new KnnGraph(trainNew, k);
        for (int i = 0; i < numInst; i++) {
            Instance inst = trainNew.instance(i);
            kgraph.computeKnn(inst,m_DistanceFunction);
            //得到邻域的索引和相应距离          
            KnnInfor knn[] = kgraph.getKnnInfor();
            //非对称图，对角线为0
            for (int j = 0; j < k; j++) {
                int index = knn[j].getIndex();
                double weight = knn[j].getDistance();
                weightGraph[i][index] = weight;

            }
        }

    }
public void setOptions(String[] options) throws Exception {
        String tmpStr;
        String classname;
        String[] spec;
        super.setOptions(options);

        tmpStr = Utils.getOption("distance", options);
        if (tmpStr.length() != 0) {
            spec = Utils.splitOptions(tmpStr);
            if (spec.length == 0) {
                throw new Exception("Invalid DistanceFunction specification string.");
            }
            classname = spec[0];
            spec[0] = "";

            setDistanceFunction((DistanceFunction) Utils.forName(DistanceFunction.class, classname, spec));
        } else {
            setDistanceFunction(new EuclideanDistance());
        }
} /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        Vector result;
        String[] options;
        int i;
        result = new Vector();
        options = super.getOptions();
        for (i = 0; i < options.length; i++) {
            result.add(options[i]);
        }

        result.add("-distance");
        result.add(getSpecification(getDistanceFunction()));



        return (String[]) result.toArray(new String[result.size()]);
    }

//    /**
//     * 计算自适应的sigma =最小距离。
//     * @param weightGraph
//     * @return
//     */
//    double computeAdaptSigma(double[][] weightGraph) {
//        double sigma = 0.01;
//        double min = INF;
//        for (int i = 0; i < weightGraph.length; i++) {
//            for (int j = 0; j < weightGraph[0].length; j++) {
//                if (weightGraph[i][j] < min && weightGraph[i][j] != 0) {
//                    min = weightGraph[i][j];
//                }
//            }
//        }
//        //把最小值作为sigma
//        if (min > 0 && min < INF) {
//            sigma = min;
//        }
//        return sigma;
//    }

    /**
     * 
     * 将邻域图两点的距离改成两个点各自邻域点内距离之和的差
     * 保持局部特性。
     * @param weightGraph
     */
    void createPairsweightGraph(double weightGraph[][]) {
        int rownum = weightGraph.length;
        int colunum = weightGraph[0].length;
        for (int i = 0; i < rownum; i++) {
            double sumdistancei = 0.0;
            int countknni = 1;
            for (int k = 0; k < rownum; k++)//i的邻域距离累加
            {
                if (weightGraph[i][k] != INF)//实际上在这里就是i的邻域点
                {
                    sumdistancei += weightGraph[i][k];
                    countknni++;
                }
            }
            //将距离求平均
            sumdistancei = sumdistancei / (double) countknni;
            for (int j = 0; j < colunum; j++) {

                double sumdistancj = 0.0;
                int countknnj = 1;
                for (int m = 0; m < rownum; m++) {
                    if (weightGraph[j][m] != INF)//实际上在这里就是j的邻域点
                    {
                        sumdistancj += weightGraph[j][m];
                        countknnj++;
                    }


                }
                sumdistancj = sumdistancj / (double) countknnj;
                double pairsdistance = Math.abs(sumdistancj - sumdistancei);
                PairsweightGraph[i][j] = sumdistancj;

            }
        }

    }

    @Override
    protected void buildClassifier() throws Exception {

        int numlabled = 0;
        //String fileString="C:\\test.txt"
        FileWriter writer = new FileWriter("C:\\test.txt", true);
        Dijkstra dijkstra = new Dijkstra(weightGraph);
    //   double sigma = computeAdaptSigma(weightGraph);
        //扩大代价距离的影响，设置sigma
//        dijkstra.setSigma(m_sigma);
      //  dijkstra.setSigma(sigma);
        //对未标记样本中的每一数据寻找到标记样本数据集合的最短路径。
        for (int unLabel = 0; unLabel < m_TrainsetNew.numInstances(); unLabel++) {
            if (m_TrainsetNew.instance(unLabel).classIsMissing())//未标记数据
            {
                int source = unLabel;//源点未标记数据
                //某一个未标记样本到所有其它标记样本之间的最短距离。
                double mincostdistance = Double.POSITIVE_INFINITY;
                //未标记样本的最终类别
                double finalClass = -1.0;
                dijkstra.dijkstra(source);
                for (int Label = 0; Label < m_TrainsetNew.numInstances(); Label++) {
                    if (!m_TrainsetNew.instance(Label).classIsMissing()) {
                        int target = Label; //标记数据
                        //dijkstra.printShortestPath(source, target);
                        //有路径才写入文件

                        if (dijkstra.writePath(source, target, writer, m_TrainsetNew)) {
                            //获得目的节点的类标记
                            double targetClass = m_TrainsetNew.instance(target).classValue();
                            String str = Double.toString(targetClass);
                            // writer.write("Class " + str + "\n");
                            //获得最短路径的距离和
                            double tempcost = dijkstra.getPathDistance();
                            // 找最短路径集中的最短的路径和代价和
                            if (tempcost < mincostdistance) {
                                mincostdistance = tempcost;
                                finalClass = targetClass;
                            }
                        }

                    }
                }
                if (finalClass == -1.0) {
                    //
                    System.out.println("该未标记数据与任意标记数据都不可达！");
                    m_resultMatrix.set(unLabel, 0, -1.0);
                } else {
                    //System.out.println("该未标记的样本最终类别为：" + finalClass);
                    m_resultMatrix.set(unLabel, 0, finalClass);
                }

            }

        }
        writer.close();
    }

    /**
     * 获得该实例的类分布情况
     * @param instance
     * @return
     * @throws java.lang.Exception
     */
//    protected double[] getDistribution(Instance instance) throws Exception{
//
// }
    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);

        return result;
    }

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     *
     * @throws java.lang.Exception
     */
    public void init() throws Exception {
        //创建图；
        //建立用来寻找最短路劲的图
        int numInst = m_TrainsetNew.numInstances();
        int numClass = m_TrainsetNew.numClasses();
        int[] nodes = new int[numInst];

        for (int i = 0; i < numInst; i++) {
            nodes[i] = i;
        }
        createKNNweightGraph(m_TrainsetNew, nodes, k);
        //直接将未标记样本的终类别存入结果中,初始类别都为-1.0
        m_resultMatrix = new Matrix(numInst, 1, -1.0);

    }

    @Override
    protected void build() throws Exception {
        init();
        buildClassifier();
    }
  /**
     * Returns the tip text for this property
     *
     * @return 		tip text for this property suitable for
     * 			displaying in the explorer/experimenter gui
     */
    public String distanceFunctionTipText() {
        return "The distance function to use for finding neighbours " + "(default: weka.core.EuclideanDistance). ";
    }

    /**
     * returns the distance function currently in use
     *
     * @return		the current distance function
     */
    public DistanceFunction getDistanceFunction() {
        return m_DistanceFunction;
    }

    /**
     * sets the distance function to use
     *
     * @param value	the distance function to use
     */
    public void setDistanceFunction(DistanceFunction value) {
        m_DistanceFunction = value;
    }

    /**
     * Gets how the training data will be transformed. Will be one of
     * SMO.FILTER_NORMALIZE, SMO.FILTER_STANDARDIZE, SMO.FILTER_NONE.
     *
     * @return the filtering mode
     */
    public SelectedTag getFilterType() {
        return new SelectedTag(m_filterType, SMO.TAGS_FILTER);
    }

    /**
     * Sets how the training data will be transformed. Should be one of
     * SMO.FILTER_NORMALIZE, SMO.FILTER_STANDARDIZE, SMO.FILTER_NONE.
     *
     * @param value the new filtering mode
     */
    public void setFilterType(SelectedTag value) {
        if (value.getTags() == SMO.TAGS_FILTER) {
            m_filterType = value.getSelectedTag().getID();
        }
    }

    /**
     * Returns the tip text for this property
     *
     * @return 		tip text for this property suitable for
     * 			displaying in the explorer/experimenter gui
     */
    public String filterTypeTipText() {
        return "Determines how/if the data will be transformed.";
    }

    /**
     * generates copies of the original datasets and also builds a relation
     * (hashtable) between each original instance and new instance. This is
     * necessary to retrieve the determined class value in the
     * <code>getDistribution(Instance)</code> method.
     *
     * @see   #getDistribution(Instance)
     * @throws Exception if anything goes wrong
     */
    protected void generateSets() throws Exception {
        super.generateSets();
//利用父类的成员变量
        m_Data =
                new CostDistanceBasedSSLInstances(this, m_Trainset, m_Testset);
        //貌似是合并后的数据确实
        m_TrainsetNew =
                m_Data.getTrainSet();
        m_TestsetNew =
                null;
    }

    /**
     * splits the train set into train and test set if no test set was provided,
     * according to the set SplitFolds.
     *
     * @see Splitter
     * @see #getSplitFolds()
     * @see #getInvertSplitFolds()
     * @throws Exception if anything goes wrong with the Filter
     */
    protected void splitTrainSet() throws Exception {
        Splitter splitter;

        splitter =
                new Splitter(m_Trainset);
        splitter.setSplitFolds(getSplitFolds());
        splitter.setInvertSplitFolds(getInvertSplitFolds());
        splitter.setVerbose(getVerbose());

        m_Trainset =
                splitter.getTrainset();
        m_Testset =
                splitter.getTestset();
    }
    // protected class
    /* ********************* other classes ************************** */

    /**
     * Stores the relation between unprocessed instance and processed instance.
     *
     * @author FracPete (fracpete at waikato dot ac dot nz)
     */
    protected class CostDistanceBasedSSLInstances
            implements Serializable {

        /** for serialization */
        private static final long serialVersionUID = 1975979462375468594L;
        /** the parent algorithm (used to get the parameters) */
        protected DistanceBasedSSL m_Parent = null;
        /** the unprocessed instances */
        protected Instance[] m_Unprocessed = null;
        /** the new training set */
        protected Instances m_Trainset = null;
        /** for finding instances again (used for classifying) */
        protected InstanceComparator m_Comparator = new InstanceComparator(false);
        /** The filter used to make attributes numeric. */
        protected NominalToBinary m_NominalToBinary;
        /** The filter used to standardize/normalize all values. */
        protected Filter m_Filter = null;
        /** The filter used to get rid of missing values. */
        protected ReplaceMissingValues m_Missing;

        /**
         * initializes the object
         *
         * @param parent      the parent algorithm
         * @param train       the train instances
         * @param test        the test instances
         * @throws Exception  if something goes wrong
         */
        public CostDistanceBasedSSLInstances(DistanceBasedSSL parent, Instances train, Instances test)
                throws Exception {

            super();

            m_Parent = parent;

            // set up filters
            m_Missing = new ReplaceMissingValues();
            //只要训练数据的输入结构，
            m_Missing.setInputFormat(train);

            m_NominalToBinary = new NominalToBinary();
            m_NominalToBinary.setInputFormat(train);

            if (getParent().getFilterType().getSelectedTag().getID() == SMO.FILTER_STANDARDIZE) {
                m_Filter = new Standardize();
                m_Filter.setInputFormat(train);
            } //采用SMO过滤有什么作用？
            else if (getParent().getFilterType().getSelectedTag().getID() == SMO.FILTER_NORMALIZE) {
                m_Filter = new Normalize();
                m_Filter.setInputFormat(train);
            } else {
                m_Filter = null;
            }

            // build sorted array (train + test)
            m_Unprocessed = new Instance[train.numInstances() + test.numInstances()];
            for (int i = 0; i < train.numInstances(); i++) {
                m_Unprocessed[i] = train.instance(i);
            }
            for (int i = 0; i < test.numInstances(); i++) {
                m_Unprocessed[train.numInstances() + i] = test.instance(i);
            }
            //把所有的数据集所有的数据都进行了排序！
            Arrays.sort(m_Unprocessed, m_Comparator);

            // filter data，这个时候trainset才有值
            m_Trainset = new Instances(train, 0);
            //成员的训练集是所有的train和测试数据之和？
            for (int i = 0; i < m_Unprocessed.length; i++) {
                m_Trainset.add(m_Unprocessed[i]);
            }
            //这回事真的进行数据过滤
            m_Missing.setInputFormat(m_Trainset);
            m_Trainset = Filter.useFilter(m_Trainset, m_Missing);

            m_NominalToBinary.setInputFormat(m_Trainset);
            m_Trainset = Filter.useFilter(m_Trainset, m_NominalToBinary);

            if (m_Filter != null) {
                m_Filter.setInputFormat(m_Trainset);
                m_Trainset = Filter.useFilter(m_Trainset, m_Filter);
            }
        }

        /**
         * 返回算法父类
         * @return
         */
        private DistanceBasedSSL getParent() {
            return m_Parent;
        }

        /**
         * returns the train set (with the processed instances)
         *
         * @return		the train set
         */
        public Instances getTrainSet() {
            return m_Trainset;
        }

        /**
         * returns the number of stored instances
         *
         * @return		the number of instances
         */
        public int size() {
            return m_Trainset.numInstances();
        }

        /**
         * returns the index of the given (unprocessed) Instance, -1 in case it
         * can't find the instance
         *
         * @param inst	the instance to return the index for
         * @return		the index for the instance, -1 if not found
         */
        public int indexOf(Instance inst) {
            return Arrays.binarySearch(m_Unprocessed, inst, m_Comparator);
        }

        /**
         * returns the processed instance for the given index, null if not within
         * bounds.
         *
         * @param index	the index of the instance to retrieve
         * @return		null if index out of bounds, otherwise the instance
         */
        public Instance get(int index) {
            if ((index >= 0) && (index < m_Trainset.numInstances())) {
                return m_Trainset.instance(index);
            } else {
                return null;
            }
        }

        /**
         * returns the processed version of the unprocessed instance in the new
         * training set, null if it can't find the instance
         * @param inst      the unprocessed instance to retrieve the processed one
         *                  for
         * @return          the processed version of the given instance
         * @see             #getTrainSet()
         */
        public Instance get(Instance inst) {
            return get(indexOf(inst));
        }
    }
}
