/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import weka.classifiers.collective.CollectiveRandomizableClassifier;
import weka.classifiers.functions.SMO;
import weka.core.ExpDistance;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Semi-Supervised Robust Path Spectral Clustering
 * @author Administrator
 */
public class RobustPathSpectralClustering extends CollectiveRandomizableClassifier implements
        TechnicalInformationHandler {

    private int m_filterType;
    private RobustPathSpectralClusteringInstances m_Data;
    /** copy of the original training dataset */
    protected Instances m_TrainsetNew;
    /** copy of the original test dataset */
    protected Instances m_TestsetNew;
    protected ExpDistance expDis = new ExpDistance();
    private double sigma = 0.43;
    /** 在S集合中找最相似的*/
    private double maxSimilarity = 0.0;
    /** 在D集合中找最不相似的*/
    private double minSimilarity = -1.0;
    /**保存结果*/
    private Matrix m_resultMatrix;

    public void setExpDis(ExpDistance expDis) {
        this.expDis = expDis;
    }

    public double getSigma() {
        return sigma;
    }

    @Override
    protected double[] getDistribution(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * 在相似集合S中寻找相似度最大相似度。
     * S集合实际由给定的类标记中相同类别的数据对组成（x,y）,
     * @param trainInstances
     * @return
     */
    private void computeMaxMinSimilarity(Instances trainInstances) {

        int num = trainInstances.numInstances();



        int indexClass = trainInstances.classIndex();
        for (int i = 0; i < num; i++) {
            Instance first = trainInstances.instance(i);
            for (int j = 0; j < num; j++) {
                Instance second = trainInstances.instance(j);
                //如果两个数据都是标记的数据,并且两个数据的类标记一样,且不是同一个样本，S集合
                if ((!first.classIsMissing()) & !(second.classIsMissing()) & (first.classValue() == second.classValue()) & (i != j)) {
                    if (maxSimilarity < expDis.distance(first, second)) {
                        maxSimilarity = expDis.distance(first, second);
                    }
                }
                //如果两个数据都是标记的数据,并且两个数据的类标记不一一样,且不是同一个样本，D集合
                if ((!first.classIsMissing()) & !(second.classIsMissing()) & (first.classValue() != second.classValue()) & (i != j)) {
                    if (minSimilarity > expDis.distance(first, second)) {
                        minSimilarity = expDis.distance(first, second);
                    }
                }
            }
        }
    }
    Matrix computeMatrixY(Matrix MatrixL,int dim){

        int row=MatrixL.getRowDimension();
        Matrix MatrixY=new Matrix(row,dim);
        //获取特征向量和特征值
        EigenvalueDecomposition eigDec=MatrixL.eig();
        Matrix eigV=eigDec.getV();
        double[] eigvalues=eigDec.getRealEigenvalues();

        return MatrixY;
    }
    Matrix computeMatrixL(Matrix similarityMatrix, Matrix MatrixD)
    {
       int num=similarityMatrix.getColumnDimension();
        Matrix MatrixL=new Matrix(num,num);
        Matrix Temp=null;
        Temp=MatrixD.sqrt().inverse();
        MatrixL =Temp.times(similarityMatrix).times(Temp);
        return MatrixL;
    }
    /**
     * 计算D矩阵
     * @param similarityMatrix
     * @return
     */
Matrix computeMatrixD(Matrix similarityMatrix){
    int num=similarityMatrix.getColumnDimension();
    Matrix MatrixD=new Matrix(num, num);
    double sum=0.0;
    for(int i=0;i<num;i++){
           sum = 0.0;
        for(int j=0;j<num;j++)
        {
            sum+=similarityMatrix.get(i, j);
        }
           MatrixD.set(i, i, sum);
    }
    return  MatrixD;
}
    /**
     * 定义训练集中的相似矩阵
     * @param trainInstances 包含有标记的样本和未标记的样本
     * @throws Exception
     */
    private void createSimilarityMatrix(Instances trainInstances) throws Exception {

        int num = trainInstances.numInstances();
        expDis.setSigma(sigma);
        //在S和D集合中分别计算相似度最大和最小值
        computeMaxMinSimilarity(trainInstances);
        Matrix similarityMatrix = new Matrix(num, num);
        int indexClass = trainInstances.classIndex();
        for (int i = 0; i < num; i++) {
            Instance first = trainInstances.instance(i);
            for (int j = 0; j < num; j++) {
                Instance second = trainInstances.instance(j);
                //如果两个数据都是未标记的数据
                if ((!first.classIsMissing()) & !(second.classIsMissing())) {
                    if (first.classValue() == second.classValue())//同一类
                    {
                        similarityMatrix.set(i, j, maxSimilarity);
                    } else {
                        similarityMatrix.set(i, j, minSimilarity);//不同类
                    }
                } //不在以标记点中，（即不在相似S和不相似D集合中）
                else {
                    if (i == j) {
                        double dis = 0.0;
                        similarityMatrix.set(i, j, dis);

                    } else {
                        double dis = expDis.distance(first, second);
                        similarityMatrix.set(i, j, dis);
                    }
                }
            }
        }
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
        createSimilarityMatrix(m_TrainsetNew);
        //直接将未标记样本的终类别存入结果中,初始类别都为-1.0
        m_resultMatrix = new Matrix(numInst, 1, -1.0);

    }

    @Override
    protected void buildClassifier() throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected void build() throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
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

    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * Stores the relation between unprocessed instance and processed instance.
     * 将一些数据进行预处理，比如，去掉一些标记，噪声等等
     * @author FracPete (fracpete at waikato dot ac dot nz)
     */
    protected class RobustPathSpectralClusteringInstances
            implements Serializable {

        /** for serialization */
        private static final long serialVersionUID = 1975979462375468594L;
        /** the parent algorithm (used to get the parameters) */
        protected RobustPathSpectralClustering m_Parent = null;
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
        public RobustPathSpectralClusteringInstances(RobustPathSpectralClustering parent, Instances train, Instances test)
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
        private RobustPathSpectralClustering getParent() {
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
         * @return the number of instances
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
