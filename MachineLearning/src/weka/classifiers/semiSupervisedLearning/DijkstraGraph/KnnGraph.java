/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning.DijkstraGraph;

import java.util.Enumeration;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *  为数据集的每一条数据实例寻找k个近邻
 * @author Administrator
 */
public class KnnGraph {

    Instances m_dataset;
    int m_k;
    //存储k个近邻信息
   public  KnnInfor m_kNNInfor[];

    public KnnGraph(Instances dataset, int k) {
        m_dataset = dataset;
        m_k = k;
        m_kNNInfor = new KnnInfor[k];
    }

    /**
     * 计算k近邻
     * @param instance
     */
 public   void computeKnn(Instance instance) {
        Enumeration e = m_dataset.enumerateInstances();
        EuclideanDistance function = new EuclideanDistance();
        int num = m_dataset.numInstances();
        // 将与其他实力的所有点的距离都计算一遍，并保存索引和距离
        KnnInfor allInfor[] = new KnnInfor[num];
        //计算与所有的实例的距离
        int index = 0;// 索引
        while (e.hasMoreElements()) {
            Instance temp = (Instance) e.nextElement();
            double distance = function.distance(instance, temp);
            allInfor[index] = new KnnInfor(index, distance);
            index++;
        }
        //对所有的距离进行排序
        sortAllInfor(allInfor);
        setKnnInfor(allInfor);
    }
    public KnnInfor[] getKnnInfor(){
        return m_kNNInfor;
}
    /**
     * 挑选前K个最小的做近邻
     * @param allInfor
     */
    private void setKnnInfor(KnnInfor allInfor[]) {
        for (int i = 0; i < m_k; i++) {
            m_kNNInfor[i] = allInfor[i];
        }
    }

    /**
     *  对所有的距离进行排序，并保存索引和距离
     * @param allInfor
     */
    private void sortAllInfor(KnnInfor allInfor[]) {
        int size = allInfor.length;
        int base = 0, compare = 0, min = 0;
        for (base = 0; base < size - 1; base++) {
            min = base;
            for (compare = base + 1; compare < size; compare++) {
                if (allInfor[compare].distance < allInfor[min].distance) {
                    min = compare;
                }
            }
            KnnInfor temp = allInfor[min];
            allInfor[min] = allInfor[base];
            allInfor[base] = temp;
        }

    }
}


