/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import java.util.Enumeration;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Administrator
 */
public class SemiBoostSplitTrainSet implements SplitTrainSet {

    public SemiBoostSplitTrainSet(Instances m_trainDataSet) {
        this.m_trainDataSet = m_trainDataSet;
    }
    /** 已经标记的数目*/
    protected int m_numLabledData = 30;
    //随机数
    protected Random m_randomInstance = new Random(1);
    /**给定的总的训练数据*/
    private Instances m_trainDataSet;
    /** 已经标记的数据*/
    private Instances labledDataSet;

    /** 已经标记的数据一个备份，这个可以不断增加*/
    private Instances copylabledDataSet;
    /** 未标记的数据*/
    private Instances UnlabledDataSet;
    /**标记的百分比*/
    private double m_percentLabeled = 10.0;


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

    public Instances getLabledData() {
        return labledDataSet;
    }
    public Instances reSampleLabledData(int seed, int numLabled){
     {
        Instances newLabeledData;
        int i;
        int index;
        Instance instance;
        Random random;
        random = new Random(seed);
     
        newLabeledData = new Instances(copylabledDataSet, numLabled);

        if (copylabledDataSet.numInstances() > 0) {
            for (i = 0; i < numLabled; i++) {
                index = random.nextInt(numLabled);
                instance = copylabledDataSet.instance(index);
                newLabeledData.add(instance);
            }
        }

        return newLabeledData;
    }
    }

    public Instances getUnLabledData() {
        return UnlabledDataSet;
    }

    public void splitTrainSet() {
    }

    public void setRandom(Random randomInstance) {
        m_randomInstance = randomInstance;
    }

    private void splitData(int numLabledData) {

        m_numLabledData = numLabledData;
        // number of unlabeled data;
        int unlabledSize = m_trainDataSet.numInstances() - m_numLabledData;
        if (m_trainDataSet.classAttribute().isNominal()) {
            int numClass = m_trainDataSet.numClasses();
            Instances[] subSets = new Instances[numClass];
            for (int i = 0; i < numClass; i++) {
                //给每个类别初始化
                subSets[i] = new Instances(m_trainDataSet, 5);
            }

            Enumeration e = m_trainDataSet.enumerateInstances();
            //new标记数据集合
            labledDataSet = new Instances(m_trainDataSet, m_trainDataSet.numInstances());
            //new未标记数据集合
            UnlabledDataSet = new Instances(m_trainDataSet, m_trainDataSet.numInstances());
            while (e.hasMoreElements()) {
                Instance inst = (Instance) e.nextElement();
                if (inst.classIsMissing()) {
                    subSets[numClass].add(inst);
                } else {
                    subSets[(int) inst.classValue()].add(inst);
                }
            }
            for (int i = 0; i < numClass; i++) {
                subSets[i].compactify();
            }

            for (int i = 0; i < numClass; i++) {
                //需添加到labledDataSet中每一类实例的数目，注意实际标记的总数可能会减少
                int sublabeledSize = (int) (m_numLabledData / numClass);
                for (int j = 0; j < sublabeledSize; j++) {
                    labledDataSet.add(subSets[i].instance(j));
                }
                //剩下的做未标记数据
                for (int k = sublabeledSize; k < subSets[i].numInstances(); k++) {
                    UnlabledDataSet.add(subSets[i].instance(k));
                }
                //释放内存
                subSets[i] = null;
            }
            //申请空间与实际空间匹配
            labledDataSet.compactify();
            UnlabledDataSet.compactify();

        } else {
            System.out.println("类标记为Numeric target !");
        }
    }

    /**
     * 根据标记数目和是否随机，把训练数据集分成标记的数据集和未标注的数据集；
     * @param numLabledData
     * @param randomSplit
     */
    public void splitTrainSet(int numLabledData, boolean randomSplit) {


        if (randomSplit) {
            m_trainDataSet.randomize(m_randomInstance);
            splitData(numLabledData);
            labledDataSet.randomize(m_randomInstance);
            UnlabledDataSet.randomize(m_randomInstance);
        } else {
            splitData(numLabledData);
        }
    }

    /**
     * 按百分比将训练数据分成标记数据和未标记数据。
     * @param percentLabeled 标记数据所占的百分比
     */
    private void splitData(double percentLabeled) {
        m_percentLabeled = percentLabeled;
        // number of unlabeled data;
        int labledSize = Utils.round(m_trainDataSet.numInstances() * m_percentLabeled / 100);
        int unlabledSize = m_trainDataSet.numInstances() - labledSize;
        if (m_trainDataSet.classAttribute().isNominal()) {
            int numClass = m_trainDataSet.numClasses();
            Instances[] subSets = new Instances[numClass];
            for (int i = 0; i < numClass; i++) {
                //给每个类别初始化
                subSets[i] = new Instances(m_trainDataSet, 5);
            }

            Enumeration e = m_trainDataSet.enumerateInstances();
            //new标记数据集合
            labledDataSet = new Instances(m_trainDataSet, m_trainDataSet.numInstances());
            //new未标记数据集合
            UnlabledDataSet = new Instances(m_trainDataSet, m_trainDataSet.numInstances());
            while (e.hasMoreElements()) {
                Instance inst = (Instance) e.nextElement();
                if (inst.classIsMissing()) {
                    subSets[numClass].add(inst);
                } else {
                    subSets[(int) inst.classValue()].add(inst);
                }
            }
            for (int i = 0; i < numClass; i++) {
                subSets[i].compactify();
            }

            for (int i = 0; i < numClass; i++) {
                //需添加到labledDataSet中每一类实例的数目，注意实际标记的总数可能会减少
                int sublabeledSize = Utils.round(subSets[i].numInstances() * m_percentLabeled / 100);
                for (int j = 0; j < sublabeledSize; j++) {
                    labledDataSet.add(subSets[i].instance(j));
                }
                //剩下的做未标记数据
                for (int k = sublabeledSize; k < subSets[i].numInstances(); k++) {
                    UnlabledDataSet.add(subSets[i].instance(k));
                }
                //释放内存
                subSets[i] = null;
            }
            //申请空间与实际空间匹配
            labledDataSet.compactify();
            UnlabledDataSet.compactify();
           //备份一份后面用
            copylabledDataSet=new Instances(labledDataSet,0,labledDataSet.numInstances());

        } else {
            System.out.println("类标记为Numeric target !");
        }
    }

    /**
     * 按百分比将训练数据分成标记数据和未标记数据。
     * @param percentLabeled 标记数据所占的百分比
     * @param randomSplit 是否将分好后的数据随机打乱顺序。
     */
    public void splitTrainSet(double percentLabeled, boolean randomSplit) {

        if (randomSplit) {
            m_trainDataSet.randomize(m_randomInstance);
            splitData(percentLabeled);
            labledDataSet.randomize(m_randomInstance);
            UnlabledDataSet.randomize(m_randomInstance);
        } else {
            splitData(percentLabeled);
        }

    }

    /**
     * 将unlabedDataSet中的信任度较高的%分比加入到copyLabledDataSet中
     * @param percent
     * @param distribution
     */
    public void addToLabledDataSet(double percent, Distribution[] distribution) {
        //所有的已经按信任度排好序的候选unlabedDataSet总数
        int allsize = distribution.length;
        int numClasses = m_trainDataSet.numClasses();
        copylabledDataSet=new Instances(labledDataSet,0,labledDataSet.numInstances());
        //按类别来统计每类的数量；
        int[] countNumberByClass = new int[numClasses];
        int[] temp=new int[numClasses];
        for (int i = 0; i < allsize; i++) {
            countNumberByClass[distribution[i].m_indexClass]++;
        }
        for (int i = 0; i < numClasses; i++) {
            //需要添加的每一类的数据按percent来控制
            countNumberByClass[i] = (int) (countNumberByClass[i]*percent/100) ;
        }

        for (int i = 0; i < allsize; i++) {
            int indexclass=distribution[i].m_indexClass;
            temp[indexclass]++;
            //添加的每一类数目必须小于控制的数目
            if (temp[indexclass]<countNumberByClass[indexclass]) {
                 Instance instance = UnlabledDataSet.instance(distribution[i].m_indexInstance);
                  //并将类别号给该实例
                instance.setClassValue(distribution[i].m_indexClass);
                copylabledDataSet.add(instance);
            }
        }

    }

    /**
     * 将unlabedDataSet中的numberIncreaseLabelbed加入到LabledDataSet中
     * @param numberIncreaseLabelbed 增加的信任度的条数
     * @param distribution 为Unlabled 数据的分布
     */
    public void addToLabledDataSet(int numberIncreaseLabelbed, Distribution[] distribution) {
        if (distribution == null) {


            System.out.println("信任度规定的个数超过了实际存在的个数");
            return;

        } else {
            //把distribution中剩下的全部拷贝到Labeled的数据中
            if (numberIncreaseLabelbed > distribution.length) {
                numberIncreaseLabelbed = distribution.length;
            }
            //通过位置索引将unlabeled中的一条实例加入到labeledDataSet中
            for (int i = 0; i < numberIncreaseLabelbed; i++) {
                Instance instance = UnlabledDataSet.instance(distribution[i].m_indexInstance);
                //并将类别号给该实例
                instance.setClassValue(distribution[i].m_indexClass);
                labledDataSet.add(instance);
            }

//            //通过位置索引将unlabeledDataSet中的一条实例减去中
//            for (int i = 0; i < numberIncreaseLabelbed; i++) {
//                UnlabledDataSet.delete(distribution[i].m_indexInstance);
//            }
        }
    }
}
