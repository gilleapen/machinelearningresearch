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
import weka.classifiers.collective.CollectiveRandomizableClassifier;
import weka.classifiers.collective.util.Splitter;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.semiSupervisedLearning.DijkstraGraph.KnnGraph;
import weka.classifiers.semiSupervisedLearning.DijkstraGraph.KnnInfor;
import weka.classifiers.semiSupervisedLearning.dijkstra.Side;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *  基于距离代价函数的 半监督学习
 * @author huangdongshan
 */
public class CostDistanceBasedSSL extends CollectiveRandomizableClassifier implements
        TechnicalInformationHandler {

    /** Whether to normalize/standardize/neither */
    protected int m_filterType;
    protected CostDistanceBasedSSL m_Parent;
    private CostDistanceBasedSSLInstances m_Data;
    /** copy of the original training dataset */
    protected Instances m_TrainsetNew;
    /** copy of the original test dataset */
    protected Instances m_TestsetNew;
    int k = 5;
    //define some constants
    public static final double INF = Double.MAX_VALUE; //infinity
    private double weightGraph[][];

    @Override
    protected double[] getDistribution(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
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
     * 构建邻域图
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
         KnnGraph kgraph= new KnnGraph(trainNew,  k);
        for (int i = 0; i < numInst; i++) {
            Instance inst = trainNew.instance(i);
            kgraph.computeKnn(inst);
            //得到邻域的索引和相应距离          
            KnnInfor knn[]=kgraph.getKnnInfor();
            //非对称图，对角线为0
            for (int j = 0; j < k; j++) {
                int index=knn[j].getIndex();
                double weight=knn[j].getDistance();
                weightGraph[i][index] = weight;
        
            }
        }

    }



    @Override
    protected void buildClassifier() throws Exception {
        //创建图；
        //建立用来寻找最短路劲的图
        int numInst = m_TrainsetNew.numInstances();
        int[] nodes = new int[numInst];

        for (int i = 0; i < numInst; i++) {
            nodes[i] = i;
        }

        createKNNweightGraph(m_TrainsetNew, nodes, k);

        Enumeration e = m_TrainsetNew.enumerateInstances();
        int numlabled = 0;
        FileWriter writer = new FileWriter("C:\\test.txt", true);
        Dijkstra dijkstra = new Dijkstra(weightGraph);
        //对未标记样本中的每一数据寻找到标记样本数据集合的最短路径。
        for (int unLabel = 0; unLabel < m_TrainsetNew.numInstances(); unLabel++) {
            if (m_TrainsetNew.instance(unLabel).classIsMissing())//未标记数据
            {
                int source = unLabel;//源点未标记数据
                for (int Label = 0; Label < m_TrainsetNew.numInstances(); Label++) {
                    if (!m_TrainsetNew.instance(Label).classIsMissing()) {
                        int target = Label; //标记数据
                        //dijkstra.printShortestPath(source, target);
                        dijkstra.writePath(source, target, writer);
                       double targetClass=m_TrainsetNew.instance(Label).classValue();
                        String str=Double.toString(targetClass);
                        writer.write("targetClass"+str+"\n");
                    }
                }

            }
           
        }
         writer.close();
    }

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

    @Override
    protected void build() throws Exception {

        buildClassifier();
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

    /**
     * performs initialization of members
     */
    protected void initializeMembers() {
        super.initializeMembers();

        m_TrainsetNew = null;
        m_TestsetNew = null;

        m_filterType = SMO.FILTER_NORMALIZE;
        m_Data = null;

    }
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
        protected CostDistanceBasedSSL m_Parent = null;
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
        public CostDistanceBasedSSLInstances(CostDistanceBasedSSL parent, Instances train, Instances test)
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
            //    Arrays.sort(m_Unprocessed, m_Comparator);

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
        private CostDistanceBasedSSL getParent() {
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
